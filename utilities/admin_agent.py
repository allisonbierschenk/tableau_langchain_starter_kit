"""
Admin Agent for Tableau Cloud User and Group Management
Uses MCP server for admin operations: add, update, edit, delete users and groups
"""

import os 
import json
import asyncio
from contextlib import AsyncExitStack
from typing import List, Dict, Any, Optional

# LangChain Libraries
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# MCP HTTP client
import httpx
from dotenv import load_dotenv

from utilities.mcp_jsonrpc_client import MCPJsonRpcSseClient
from utilities.identity_mapping import slack_mcp_extra_headers_from_env
from orchestration.mcp_router import MCPRouter
from orchestration.catalog import tool_definitions_for_server
from orchestration.langchain_adapters import create_routed_langchain_tools
from orchestration.cross_mcp_prompt import CROSS_MCP_GOVERNANCE
from orchestration.types import MCPServerId, ToolCatalog
from utilities.mcp_schema_models import build_tool_args_model_from_input_schema
from utilities.tableau_mcp_normalize import (
    coerce_value_to_plain_dict,
    maybe_log_admin_users_schema,
    normalize_tableau_mcp_tool_call,
)
from utilities.pulse_metric_enrich import (
    backfill_pulse_id_names_via_mcp,
    enrich_assistant_text_pulse_metric_names,
    extract_pulse_metric_id_to_name,
    metric_uuids_candidates_in_response,
    scrub_stale_pulse_api_disclaimer,
)

# Load environment variables
load_dotenv()

# Configuration for Admin MCP Server
ADMIN_MCP_SERVER = os.getenv("ADMIN_MCP_SERVER")
if not ADMIN_MCP_SERVER:
    raise ValueError("ADMIN_MCP_SERVER environment variable is required")

SLACK_MCP_SERVER = (os.getenv("SLACK_MCP_SERVER") or "").strip()

# Get site information from environment
TABLEAU_SITE = os.getenv("TABLEAU_SITE", "unknown")
TABLEAU_DOMAIN = os.getenv("TABLEAU_DOMAIN", "Tableau Cloud")
TABLEAU_USER = os.getenv("TABLEAU_USER", "")

# Per-user authentication header (optional, for direct-trust + JWT pattern)
MCP_JWT_SUB_CLAIM_HEADER = os.getenv("MCP_JWT_SUB_CLAIM_HEADER", "")

MAX_ADMIN_ITERATIONS = 8  # Admin operations typically need fewer iterations

# Build admin-specific system prompt with site context
_SITE_CONTEXT = f"""
SITE CONTEXT:
- Tableau Site: {TABLEAU_SITE}
- Tableau Domain: {TABLEAU_DOMAIN}
- Connected via MCP Server: {ADMIN_MCP_SERVER}
""" + (
    f"\n- Slack MCP Server: {SLACK_MCP_SERVER}"
    if SLACK_MCP_SERVER
    else ""
)


def _mcp_native_tool_name(qualified_or_native: str) -> str:
    """LangChain tool names may be qualified (tableau_mcp__…); validations use native MCP names."""
    if "__" in qualified_or_native:
        return qualified_or_native.split("__", 1)[1]
    return qualified_or_native

_CRITICAL_CONVERSATION_RULE = """
⚠️⚠️⚠️ ULTRA-CRITICAL: READ THIS IMMEDIATELY ⚠️⚠️⚠️

**IF THE USER'S MESSAGE IS JUST A ROLE (like "viewer", "explorer", "creator"):**

1. STOP - Do NOT ask for any information
2. Look at the LAST message in conversation_history where role="assistant"
3. Extract ANY email address from that message (pattern: xxx@xxx.xxx)
4. If you find an email → USE IT with the role the user just provided
5. Call add-user-to-site IMMEDIATELY - do NOT ask for the email again

**Example of CORRECT behavior:**
- Turn N: You asked "What site role should be assigned to [email]?"
- Turn N+1: User says "viewer"
- Turn N+2: You extract "[email]" from turn N and call add-user-to-site with name="[email]", siteRole="Viewer"

**NEVER do this:**
- ❌ "What email address?" after user provides the role
- ❌ Asking for information you already mentioned
- ❌ Treating each message as isolated without context

**YOUR PREVIOUS MESSAGES CONTAIN THE CONTEXT YOU NEED!**
"""

ADMIN_SYSTEM_PROMPT = """You are a helpful Tableau Cloud administrator assistant with access to MCP tools.

**How to approach requests:**

1. **Understand the request** - What is the user actually trying to accomplish?

2. **Look at your available tools** - You have tools dynamically loaded from the MCP server. Read their names and descriptions carefully.

3. **Reason about the best approach:**
   - Can you directly answer the question with available tools?
   - Do you need to combine multiple tools?
   - If you can't do exactly what was asked, what's the closest thing you can do?

4. **Be transparent:**
   - If you don't have a direct tool, explain what you CAN do and offer alternatives
   - Don't silently run a different operation than what was asked
   - Ask clarifying questions when helpful

5. **Be conversational** - Talk like ChatGPT or Claude would. Natural, helpful, smart.

You have access to MCP tools for Tableau Cloud administration. The exact tools available are loaded dynamically, so inspect them and use them intelligently.

""" + _SITE_CONTEXT + """

**Key principles:**
- Tools are self-documenting - read their descriptions and parameters
- Multi-step operations are fine (e.g., get user ID first, then update)
- Track conversation context - don't ask for the same info twice
- Present results clearly and concisely
- **Email scope:** For add/remove/update site users, only use email addresses the **user explicitly wrote in their current request** (or the assistant’s immediate prior question you are answering). Do not add extra people from much older messages, bot identity, or unrelated context when confirming or asking for site roles.

**Data sources, workbooks, and content (not only users/groups):** The MCP catalog often includes content and data tools (names vary by server), such as **`list-datasources`**, **`list-workbooks`**, **`query-datasource`**, **`get-datasource-metadata`**, **`search-content`**, etc. For requests like "list all datasources" or "what data sources exist", **search your available tool names and descriptions** for datasource/workbook/query/list patterns and **invoke the matching tool**. Only say a capability is unavailable if **no** such tool appears after you have checked the full list—and then say clearly that the **MCP server did not expose** that tool (e.g. `INCLUDE_TOOLS` / tool groups on the host), not that Tableau lacks the feature.

**Permissions and "who has access":** If the user asks to list users (by name or email) who can reach a workbook or other content, do not stop at naming a group (e.g. "All Users") and inferred capabilities. Use the relevant content-permissions / operations tools to resolve the workbook (or content) ID, list explicit grants (group vs user, capabilities), then list site users and group memberships as needed so you can name users and state whether access is direct, via a named group, project default, or site role. If the result set is large, say so and show a representative sample plus how to narrow the question.

**Tableau Pulse metrics (mandatory):** When listing Pulse subscriptions or “which metrics” a user follows, every item must show **Metric name** (or equivalent title) and never be UUID-only.
- **Incomplete / not allowed:** numbered bullets that only show `Metric ID: <uuid>` plus period/comparison.
- **If subscription payloads are id-only:** follow Tableau Pulse REST patterns (see [Pulse REST](https://help.tableau.com/current/api/rest_api/en-us/REST/rest_api_ref_pulse.htm)): use MCP tools that list metric definitions/metrics in batch or site-wide (e.g. `list-all-pulse-metric-definitions`, or `definitions:batchGet` / `metrics:batchGet` equivalents exposed as tools) to resolve each LUID to its definition or metric display name before answering.
- Prefer the same field names the MCP JSON uses (`metricDefinition.name`, `title`, `specifiedName`, etc.). If tool JSON already contains id+name pairs, your final answer must surface the name for each id.

**Tableau REST payloads (admin-users):** For `add-user-to-site`, follow the MCP `inputSchema` and Tableau REST: use nested `body.user.name` (sign-in email on Tableau Cloud) and `body.user.siteRole` with valid role names (e.g. Viewer, Unlicensed). Do not put `email` or `siteRole` at the top level of `body` unless the schema explicitly allows it.
"""

class AdminMCPHttpClient:
    """HTTP-based MCP client for admin operations"""

    def __init__(self, server_url: str, tableau_username: Optional[str] = None):
        self.server_url = server_url
        self.session = None
        self.tableau_username = tableau_username

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    def _parse_sse_response(self, response_text: str):
        """Parse Server-Sent Events response to extract JSON-RPC result"""
        lines = response_text.strip().split('\n')

        for line in lines:
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if 'result' in data:
                        return data['result']
                except json.JSONDecodeError:
                    continue

        return None

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call an MCP tool"""
        if not self.session:
            raise RuntimeError("Client session not initialized")

        try:
            # Build headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }

            # Add per-user authentication header if configured
            if MCP_JWT_SUB_CLAIM_HEADER and self.tableau_username:
                headers[MCP_JWT_SUB_CLAIM_HEADER] = self.tableau_username
                print(f"🔐 Using per-user auth: {MCP_JWT_SUB_CLAIM_HEADER}={self.tableau_username}")

            response = await self.session.post(
                f"{self.server_url}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    },
                    "id": 1
                },
                headers=headers
            )

            if response.status_code != 200:
                raise Exception(f"MCP tool call failed: {response.status_code}")

            # Parse SSE response
            result = self._parse_sse_response(response.text)
            return result

        except Exception as e:
            print(f"Error calling tool {tool_name}: {str(e)}")
            raise

    async def list_tools(self) -> List[Dict]:
        """List available tools from the MCP server"""
        if not self.session:
            raise RuntimeError("Client session not initialized")

        try:
            # Build headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }

            # Add per-user authentication header if configured
            if MCP_JWT_SUB_CLAIM_HEADER and self.tableau_username:
                headers[MCP_JWT_SUB_CLAIM_HEADER] = self.tableau_username

            response = await self.session.post(
                f"{self.server_url}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": 1
                },
                headers=headers
            )

            if response.status_code != 200:
                raise Exception(f"Failed to list tools: {response.status_code}")

            result = self._parse_sse_response(response.text)
            return result.get('tools', []) if result else []

        except Exception as e:
            print(f"Error listing tools: {str(e)}")
            raise


class AdminLangChainTool(BaseTool):
    """Wrapper to convert MCP tools into LangChain tools for admin operations"""

    name: str
    description: str
    args_schema: type[BaseModel]
    mcp_client: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> str:
        """Sync wrapper (not used)"""
        raise NotImplementedError("Use async version")

    async def _arun(self, **kwargs) -> str:
        """Execute the MCP tool"""
        try:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            clean = normalize_tableau_mcp_tool_call(self.name, clean)
            result = await self.mcp_client.call_tool(self.name, clean)

            # Format result as string for LangChain
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            elif isinstance(result, list):
                return json.dumps(result, indent=2)
            else:
                return str(result)

        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


def create_admin_langchain_tools(mcp_client: AdminMCPHttpClient, mcp_tools: List[Dict]) -> List[BaseTool]:
    """Convert MCP tools to LangChain tools for admin operations (schema-accurate nested models)."""
    langchain_tools = []

    for tool in mcp_tools:
        tool_name = tool.get('name', '')
        tool_description = tool.get('description', 'No description')
        tool_input_schema = tool.get('inputSchema', {})
        ToolArgsModel = build_tool_args_model_from_input_schema(tool_name, tool_input_schema)

        lc_tool = AdminLangChainTool(
            name=tool_name,
            description=tool_description,
            args_schema=ToolArgsModel,
            mcp_client=mcp_client
        )

        langchain_tools.append(lc_tool)

    return langchain_tools


def _nuclear_email_fix(response_text: str, tool_results: List[Dict], current_query: str = "", conversation_history: List[Dict] = None) -> str:
    """
    NUCLEAR OPTION: Fix obvious placeholder emails
    SAFETY: Preserves legitimate multiple emails from multi-user operations
    """
    import re

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Collect ALL correct emails from tool results
    correct_emails = []
    if tool_results:
        for tool_result in tool_results:
            if _tool_result_is_add_user_site(tool_result):
                email = _get_add_user_email_from_tool_args(tool_result.get("arguments") or {})
                if email and email not in correct_emails:
                    correct_emails.append(email)

    merged = _merged_user_text_for_email_guards(current_query, conversation_history)
    if merged.strip():
        query_emails = re.findall(email_pattern, merged)
        unique_query_emails = list(set(query_emails))
        if len(unique_query_emails) > 1:
            print(f"🔍 Multi-email session — skip nuclear email replace: {unique_query_emails}")
            return response_text

    # Find emails in the response
    found_emails = re.findall(email_pattern, response_text)
    unique_found = list(set(found_emails))

    # Check if response has PLACEHOLDER emails (not just wrong emails)
    PLACEHOLDER_DOMAINS = ['example.com', 'test.com']
    has_placeholder = any(any(domain in email.lower() for domain in PLACEHOLDER_DOMAINS) for email in found_emails)

    # Only activate nuclear fix for single-email operations with obvious placeholders
    if has_placeholder and len(correct_emails) == 1:
        print(f"🚨 NUCLEAR FIX: Replacing placeholder with {correct_emails[0]}")
        response_text = re.sub(email_pattern, correct_emails[0], response_text)
    elif len(unique_found) > 1 and len(correct_emails) > 1:
        # Multiple distinct emails in both response and tool calls - this is correct
        print(f"✓ Multi-email response is valid")

    return response_text


def _validate_no_placeholders(response_text: str) -> None:
    """
    Final validation to ensure NO placeholder emails appear in the response
    Raises a loud warning if any are detected
    """
    import re

    # Known placeholder patterns that are BANNED
    banned_patterns = {
        r'john\.?doe@': 'john.doe@',
        r'jane\.?doe@': 'jane.doe@',
        r'user@example': 'user@example',
        r'username@example': 'username@example',
        r'test@example': 'test@example',
        r'@example\.com': '@example.com',
        r'\[.*?email.*?\]': '[email placeholder]',
    }

    violations = []
    for pattern, description in banned_patterns.items():
        if re.search(pattern, response_text, re.IGNORECASE):
            violations.append(description)

    if violations:
        print("🚨" * 20)
        print("🚨 CRITICAL ERROR: PLACEHOLDER EMAIL DETECTED IN RESPONSE!")
        print(f"🚨 Violations found: {violations}")
        print(f"🚨 Response: {response_text[:300]}")
        print("🚨" * 20)


def _is_add_user_to_site_call(tool_name: str, tool_args: dict) -> bool:
    """MCP may expose add-user as its own tool or under admin-users."""
    native = _mcp_native_tool_name(tool_name)
    if native == "add-user-to-site":
        return True
    if native == "admin-users" and tool_args.get("operation") == "add-user-to-site":
        return True
    return False


def _get_add_user_email_from_tool_args(tool_args: dict) -> str:
    body = coerce_value_to_plain_dict(tool_args.get("body"))
    user = coerce_value_to_plain_dict(body.get("user"))
    return (str(user.get("name") or "")).strip()


def _ensure_add_user_body_shape(tool_args: dict) -> None:
    body = coerce_value_to_plain_dict(tool_args.get("body"))
    user = coerce_value_to_plain_dict(body.get("user"))
    body["user"] = user
    tool_args["body"] = body


def _tool_result_is_add_user_site(tool_result: dict) -> bool:
    return _is_add_user_to_site_call(
        tool_result.get("tool", ""),
        tool_result.get("arguments") or {},
    )


def _extract_email_from_history(conversation_history: List[Dict]) -> str:
    """Extract the most recent email mentioned in the conversation that's part of an incomplete operation"""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Search backwards from most recent, skipping completed operations
    found_incomplete_operation = False

    for msg in reversed(conversation_history):
        content = msg.get("content", "")
        role = msg.get("role", "")

        # Skip messages that indicate completed operations
        if role == "assistant" and ("✓" in content or "successfully added" in content.lower() or "has been successfully" in content.lower()):
            # This is a completed operation, skip any emails in this message
            continue

        # Look for emails in assistant follow-ups (wording varies by model)
        if role == "assistant":
            cl = content.lower()
            if "what site role" in cl or ("site role" in cl and re.search(email_pattern, content)):
                emails = re.findall(email_pattern, content)
                if emails:
                    return emails[-1]

        if role == "user" and "add" in content.lower():
            # This is a user request to add someone
            emails = re.findall(email_pattern, content)
            if emails:
                return emails[-1]  # Return the email from the add request

    return None


def _fix_placeholder_in_response(response_text: str, tool_results: List[Dict], current_query: str = "", conversation_history: List[Dict] = None) -> str:
    """
    Replace placeholder emails in the response with actual values
    IMPORTANT: Only fixes obvious placeholders, preserves legitimate multiple emails
    """
    import re

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # STEP 1: Collect ALL actual emails from tool calls
    actual_emails = []
    if tool_results:
        for tool_result in tool_results:
            if _tool_result_is_add_user_site(tool_result):
                email = _get_add_user_email_from_tool_args(tool_result.get("arguments") or {})
                if email and email not in actual_emails:
                    actual_emails.append(email)

    # STEP 2: Find emails in the response
    found_emails = re.findall(email_pattern, response_text)
    unique_found = list(set(found_emails))

    # STEP 3: Only fix if there are PLACEHOLDER emails (example.com, john.doe, etc.)
    # Do NOT replace if the response has multiple legitimate emails
    PLACEHOLDER_PATTERNS = ['example.com', 'john.doe', 'jane.doe', 'test.com', 'user@', 'username@']
    has_placeholder = any(placeholder in response_text.lower() for placeholder in PLACEHOLDER_PATTERNS)

    session_addrs = _extract_session_user_emails(current_query, conversation_history)
    if has_placeholder and len(actual_emails) == 1 and len(session_addrs) <= 1:
        # Single-email session: safe to replace obvious placeholders with the tool email
        print(f"🔧 Fixing placeholder email with: {actual_emails[0]}")
        response_text = re.sub(email_pattern, actual_emails[0], response_text)
    elif has_placeholder and len(session_addrs) > 1:
        print("🔍 Multi-email session — skip blanket placeholder→single-tool-email replace")
    elif len(unique_found) > 1 and len(actual_emails) > 1:
        # Multiple emails in response and multiple tool calls - this is expected, don't touch it
        print(f"✓ Multi-email response preserved: {unique_found}")
    elif has_placeholder:
        print(f"⚠️ Placeholder detected but unclear how to fix (multiple emails)")

    # Replace bracket placeholders only
    placeholder_pattern = r'\[(user-)?email(-address)?\]'
    if actual_emails:
        # If multiple emails, use the first one for bracket replacements
        response_text = re.sub(
            placeholder_pattern,
            actual_emails[0] if len(actual_emails) == 1 else ", ".join(actual_emails),
            response_text,
            flags=re.IGNORECASE
        )

    return response_text


def _augment_query_with_context(query: str, conversation_history: List[Dict]) -> str:
    """
    If the query looks like a simple follow-up answer (e.g., just a role),
    augment it with context from the conversation history
    """
    import difflib
    import re

    query_lower = query.lower().strip()
    query_tok = query_lower.strip()

    # Check if query is just a site role (common follow-up pattern)
    role_keywords = ['viewer', 'explorer', 'creator', 'unlicensed', 'explorercanpublish',
                     'siteadministrator', 'siteadministratorexplorer', 'siteadministratorcreator']

    short_followup = len(query.split()) <= 4
    role_exact = any(role in query_lower for role in role_keywords)
    role_fuzzy = bool(
        re.search(r"\b(viewer|viewers|creator|creators|unlicensed)\b", query_lower)
    ) or bool(re.search(r"\b(site\s*admin\w*|explorercanpublish)\b", query_lower))
    # Typo-tolerant single token (e.g. "explolrers" ~ explorer)
    role_close = False
    if short_followup and len(query.split()) <= 3 and query_tok:
        canon = [
            "viewer", "explorer", "creator", "unlicensed", "explorercanpublish",
            "siteadministratorexplorer", "siteadministratorcreator",
        ]
        role_close = bool(difflib.get_close_matches(query_tok, canon, n=1, cutoff=0.72))
    is_just_role = short_followup and (role_exact or role_fuzzy or role_close)

    if is_just_role and conversation_history:
        # This looks like a follow-up answer to a question
        # Check if the most recent assistant message was asking for a site role

        # First, check if there's an email in the most recent user messages (last 2 turns)
        # This takes priority over old conversation history
        recent_user_emails = []
        user_msg_count = 0
        for msg in reversed(conversation_history):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                if user_content and "add" in user_content.lower():
                    # This is an "add user" request - extract email
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    found_emails = re.findall(email_pattern, user_content)
                    recent_user_emails.extend(found_emails)
                user_msg_count += 1
                if user_msg_count >= 2:  # Only look at last 2 user messages
                    break

        # If we found emails in recent user messages, use those
        if recent_user_emails:
            unique_recent = list(dict.fromkeys(recent_user_emails))  # Preserve order, remove dupes
            if len(unique_recent) == 1:
                augmented_query = f"Add user {unique_recent[0]} with site role {query}"
            else:
                email_list = ", ".join(unique_recent)
                augmented_query = f"Add users {email_list} with site role {query}"
            print(f"🔍 Context from recent user messages - Augmented query: '{augmented_query}'")
            return augmented_query

        # Fallback: check the most recent assistant message (only if no recent user emails found)
        for msg in reversed(conversation_history):
            if (msg.get("role") or "").lower() != "assistant":
                continue
            content = msg.get("content", "")

            # Skip completed operations
            if "✓" in content or "successfully" in content.lower():
                return query  # Not a follow-up, completed operation in between

            cl = content.lower()
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            # Phrasing varies ("What site role...", "Which role..."); require email in message
            if "what site role" in cl or ("site role" in cl and re.search(email_pattern, content)):
                # Extract ALL emails from this question (could be multiple users)
                emails = re.findall(email_pattern, content)
                unique_emails: List[str] = []
                seen = set()
                for email in emails:
                    if email.lower() not in seen:
                        seen.add(email.lower())
                        unique_emails.append(email)

                if unique_emails:
                    if len(unique_emails) == 1:
                        augmented_query = f"Add user {unique_emails[0]} with site role {query}"
                    else:
                        email_list = ", ".join(unique_emails)
                        augmented_query = f"Add users {email_list} with site role {query}"
                    print(f"🔍 Context from assistant question - Augmented query: '{augmented_query}'")
                    return augmented_query

            break  # Only inspect the most recent assistant message

    return query


def _extract_email_from_text(text: str) -> str:
    """Extract any email address from a text string (returns last one)"""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[-1] if emails else None


def _extract_all_emails_from_text(text: str) -> List[str]:
    """Extract ALL unique email addresses from a text string"""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    # Return unique emails in order of appearance
    seen = set()
    unique_emails = []
    for email in emails:
        if email.lower() not in seen:
            seen.add(email.lower())
            unique_emails.append(email)
    return unique_emails


def _merged_user_text_for_email_guards(
    current_query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    max_user_messages: int = 3,
) -> str:
    """
    Current message + a few most recent user turns only.
    Do not scan the entire thread — that pulled unrelated emails into add/remove flows.
    """
    parts = [current_query or ""]
    n = 0
    for m in reversed(conversation_history or []):
        if m.get("role") != "user":
            continue
        parts.append(str(m.get("content") or ""))
        n += 1
        if n >= max_user_messages:
            break
    return "\n".join(parts)


def _extract_session_user_emails(
    current_user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    max_user_messages: int = 3,
) -> List[str]:
    """Emails from current message + last few user turns (narrow window for safety guards)."""
    return _extract_all_emails_from_text(
        _merged_user_text_for_email_guards(current_user_message, conversation_history, max_user_messages)
    )


def normalize_conversation_history(raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Unify shapes from web UI, Slack bots, and OpenAI-style clients:
    - role: Assistant / USER / bot -> assistant / user
    - content vs text vs body
    """
    if not raw:
        return []

    out: List[Dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role_raw = (m.get("role") or m.get("type") or "").strip().lower()
        if role_raw in ("assistant", "bot", "model", "ai"):
            role = "assistant"
        elif role_raw in ("user", "human", "customer", "client"):
            role = "user"
        else:
            continue

        content = m.get("content")
        if content is None:
            content = m.get("text") or m.get("body") or ""
        if isinstance(content, list):
            parts: List[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            content = " ".join(parts)
        else:
            content = str(content)

        item: Dict[str, Any] = {"role": role, "content": content}
        if m.get("tool_calls") is not None:
            item["tool_calls"] = m["tool_calls"]
        out.append(item)
    return out


async def admin_mcp_chat(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Main admin chat function using MCP tools for user and group management

    Args:
        query: User's admin request
        conversation_history: Previous conversation messages

    Returns:
        Dict with response, tool_results, and iterations
    """
    if conversation_history is None:
        conversation_history = []
    conversation_history = normalize_conversation_history(conversation_history)

    print(f"🔐 Admin Agent Started")
    print(f"📋 Original Query: '{query}'")
    print(f"📋 Conversation History Length: {len(conversation_history)}")

    # Augment query first so follow-ups (e.g. role only) carry prior emails in `query`
    original_query = query
    query = _augment_query_with_context(query, conversation_history)

    if query != original_query:
        print(f"✨ Query augmented to: '{query}'")

    # Multi-email scope = augmented query ONLY (do not union whole-thread user emails — that
    # incorrectly included unrelated addresses from old turns in site-role / add-user prompts).
    emails_augmented = _extract_all_emails_from_text(query)
    all_emails_in_query = list(dict.fromkeys(emails_augmented))
    if len(all_emails_in_query) > 1:
        print(f"🔍 MULTI-EMAIL (this turn / augmented): {len(all_emails_in_query)} addresses: {all_emails_in_query}")

    try:
        tableau_username = TABLEAU_USER if MCP_JWT_SUB_CLAIM_HEADER else None
        default_max_iter = 24 if SLACK_MCP_SERVER else MAX_ADMIN_ITERATIONS
        max_iterations = int(os.getenv("MAX_ADMIN_ITERATIONS", str(default_max_iter)))

        async with AsyncExitStack() as stack:
            admin_client = await stack.enter_async_context(
                AdminMCPHttpClient(ADMIN_MCP_SERVER, tableau_username=tableau_username)
            )
            slack_client: Optional[MCPJsonRpcSseClient] = None
            if SLACK_MCP_SERVER:
                slack_client = await stack.enter_async_context(
                    MCPJsonRpcSseClient(
                        SLACK_MCP_SERVER,
                        extra_headers=slack_mcp_extra_headers_from_env() or None,
                    )
                )
                print("💬 Slack MCP client enabled")

            router = MCPRouter.with_tableau_and_optional_slack(admin_client, slack_client)
            print("🔍 Discovering MCP tools...")
            tableau_mcp_tools = await admin_client.list_tools()
            maybe_log_admin_users_schema(tableau_mcp_tools)
            n_tableau = len(tableau_mcp_tools)
            print(
                f"✅ Tableau MCP: {n_tableau} tools "
                f"(if this is unexpectedly low, check INCLUDE_TOOLS / EXCLUDE_TOOLS on the MCP server)"
            )

            langchain_tools = create_admin_langchain_tools(admin_client, tableau_mcp_tools)

            if slack_client:
                slack_mcp_tools = await slack_client.list_tools()
                n_slack = len(slack_mcp_tools)
                print(f"✅ Slack MCP: {n_slack} tools")
                slack_catalog = ToolCatalog(
                    tools=tool_definitions_for_server(MCPServerId.SLACK_MCP, slack_mcp_tools)
                )
                langchain_tools = langchain_tools + create_routed_langchain_tools(router, slack_catalog)

            if not langchain_tools:
                return {
                    "response": "No MCP tools available. Check ADMIN_MCP_SERVER / SLACK_MCP_SERVER.",
                    "tool_results": [],
                    "iterations": 0,
                }

            system_prompt = ADMIN_SYSTEM_PROMPT
            if slack_client:
                system_prompt = ADMIN_SYSTEM_PROMPT + "\n" + CROSS_MCP_GOVERNANCE

            # Initialize LLM with tools
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0,
                streaming=False
            )
            llm_with_tools = llm.bind_tools(langchain_tools)

            # Build message history
            messages = [SystemMessage(content=system_prompt)]

            # Add conversation history WITH tool calls if available
            for msg in conversation_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    # Preserve tool calls if they exist in the message
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        messages.append(AIMessage(content=content, tool_calls=tool_calls))
                    else:
                        messages.append(AIMessage(content=content))

            # Add current query with explicit multi-email context if needed
            query_to_send = query
            if len(all_emails_in_query) > 1:
                # Explicitly inject the email list to force the agent to acknowledge all of them
                email_list_str = "\n".join([f"{i+1}. {email}" for i, email in enumerate(all_emails_in_query)])
                query_to_send = f"""{query}

🚨 SYSTEM DETECTED {len(all_emails_in_query)} EMAILS IN YOUR REQUEST:
{email_list_str}

YOU MUST acknowledge and process ALL {len(all_emails_in_query)} emails listed above. If you ask for site role, list ALL {len(all_emails_in_query)} emails in your question."""
                print(f"📧 Injecting explicit multi-email context into query")

            messages.append(HumanMessage(content=query_to_send))

            # Agent loop
            iterations = 0
            tool_results = []

            while iterations < max_iterations:
                iterations += 1
                print(f"\n🔄 Iteration {iterations}")

                # Get LLM response
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                # Check if tools were called
                if not response.tool_calls:
                    # No more tools to call, return final response
                    print("✅ Admin operation complete")

                    # VALIDATION: Check if this is asking for site role but only mentioning one email when multiple were provided
                    if len(all_emails_in_query) > 1 and "what site role" in response.content.lower():
                        import re
                        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                        emails_in_response = re.findall(email_pattern, response.content)
                        unique_emails_in_response = list(set([e.lower() for e in emails_in_response]))

                        if len(unique_emails_in_response) < len(all_emails_in_query):
                            print(f"🚨 VALIDATION FAILED: Agent only asked about {len(unique_emails_in_response)} emails but query had {len(all_emails_in_query)}")
                            print(f"   Expected emails: {all_emails_in_query}")
                            print(f"   Found in response: {emails_in_response}")

                            # Force a corrected response
                            email_list_str = "\n".join([f"{i+1}. {email}" for i, email in enumerate(all_emails_in_query)])
                            corrected_response = f"""What site role should be assigned to the following users?

{email_list_str}

Available site role options:
- **Creator** - Full access to create and edit content
- **Explorer** - Can view and interact with content
- **ExplorerCanPublish** - Explorer who can also publish content
- **Viewer** - Can only view content
- **SiteAdministratorCreator** - Site admin with Creator license
- **SiteAdministratorExplorer** - Site admin with Explorer license
- **Unlicensed** - No license, cannot access site"""

                            return {
                                "response": corrected_response,
                                "tool_results": tool_results,
                                "iterations": iterations
                            }

                    # Extract tool_calls if present for history preservation
                    response_tool_calls = []
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        response_tool_calls = response.tool_calls

                    # Start with the response
                    final_response = response.content

                    # Legacy single-email normalization: never collapse multiple distinct recipients
                    # into one address (regression when follow-up text has no emails but session had two+).
                    import re

                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails_in_final = re.findall(email_pattern, final_response)
                    unique_in_final = list(dict.fromkeys(emails_in_final))

                    if len(all_emails_in_query) <= 1 and len(unique_in_final) <= 1:
                        actual_email_from_tool = None
                        for tr in reversed(tool_results):
                            if _tool_result_is_add_user_site(tr):
                                actual_email_from_tool = _get_add_user_email_from_tool_args(
                                    tr.get("arguments") or {}
                                )
                                if actual_email_from_tool:
                                    print(f"🎯 EXTRACTED ACTUAL EMAIL FROM TOOL CALL: {actual_email_from_tool}")
                                    break

                        if actual_email_from_tool and emails_in_final:
                            print(f"🔧 Normalizing single-address reply with tool email: {actual_email_from_tool}")
                            final_response = re.sub(email_pattern, actual_email_from_tool, final_response)
                    else:
                        print(
                            f"🔍 Skipping single-email normalization "
                            f"(session={len(all_emails_in_query)} addrs, reply={len(unique_in_final)} distinct)"
                        )

                    # Additional legacy fixes (kept for safety)
                    final_response = _fix_placeholder_in_response(final_response, tool_results, query, conversation_history)
                    _validate_no_placeholders(final_response)
                    final_response = _nuclear_email_fix(final_response, tool_results, query, conversation_history)

                    pulse_map = extract_pulse_metric_id_to_name(tool_results)
                    uuids_in_answer = metric_uuids_candidates_in_response(final_response)
                    more_names = await backfill_pulse_id_names_via_mcp(
                        admin_client,
                        tableau_mcp_tools,
                        final_response,
                        pulse_map,
                    )
                    if more_names:
                        pulse_map = {**pulse_map, **more_names}
                    if pulse_map:
                        enriched = enrich_assistant_text_pulse_metric_names(final_response, pulse_map)
                        if enriched != final_response:
                            print("📊 Enriched reply with Pulse metric names (tool JSON and/or catalog backfill)")
                        final_response = enriched
                    if uuids_in_answer:
                        final_response = scrub_stale_pulse_api_disclaimer(
                            final_response, uuids_in_answer, pulse_map
                        )

                    return {
                        "response": final_response,
                        "tool_results": tool_results,
                        "iterations": iterations,
                        "tool_calls": response_tool_calls  # Include for history
                    }

                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    native_name = _mcp_native_tool_name(tool_name)
                    if isinstance(tool_args, dict) and native_name == "admin-users" and "body" in tool_args:
                        tool_args["body"] = coerce_value_to_plain_dict(tool_args.get("body"))
                    print(f"🔧 Calling MCP tool: {tool_name} (native={native_name})")
                    print(f"   Arguments: {json.dumps(tool_args, indent=2)}")

                    # Log operation type for debugging
                    if native_name == "admin-users":
                        operation = tool_args.get("operation", "unknown")
                        print(f"   Operation: {operation}")
                        if operation == "remove-user-from-site":
                            print(f"   🗑️  Attempting to remove user ID: {tool_args.get('userId', 'MISSING')}")

                    # VALIDATION: Block placeholder emails in tool calls (add-user-to-site or admin-users)
                    if _is_add_user_to_site_call(tool_name, tool_args):
                        user_name = _get_add_user_email_from_tool_args(tool_args)
                        BANNED_EMAILS = [
                            "john.doe@example.com",
                            "jane.doe@example.com",
                            "user@example.com",
                            "user1@example.com",
                            "test@example.com",
                            "username@example.com",
                        ]

                        if any(banned in user_name.lower() for banned in BANNED_EMAILS) or "@example.com" in user_name.lower():
                            print(f"🚨 BLOCKED PLACEHOLDER EMAIL IN TOOL CALL: {user_name}")

                            actual_email = _extract_email_from_history(conversation_history)
                            if not actual_email:
                                actual_email = _extract_email_from_text(query)

                            if actual_email:
                                print(f"✅ REPLACING WITH ACTUAL EMAIL: {actual_email}")
                                _ensure_add_user_body_shape(tool_args)
                                tool_args["body"]["user"]["name"] = actual_email
                            else:
                                print("❌ NO ACTUAL EMAIL FOUND - BLOCKING TOOL CALL")
                                error_msg = "I need a valid email address to add a user. Please provide the email address you want to add."
                                messages.append(
                                    ToolMessage(
                                        content=error_msg,
                                        tool_call_id=tool_call["id"]
                                    )
                                )
                                continue

                    # VALIDATION: Validate user ID format for remove-user-from-site
                    if native_name == "admin-users" and tool_args.get("operation") == "remove-user-from-site":
                        user_id = tool_args.get("userId", "")
                        import re
                        # Tableau user IDs are UUIDs (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
                        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                        if not re.match(uuid_pattern, user_id, re.IGNORECASE):
                            print(f"🚨 INVALID USER ID FORMAT: {user_id}")
                            print(f"   Expected UUID format, got: {type(user_id)} = '{user_id}'")
                            error_msg = f"Invalid user ID format: '{user_id}'. User IDs must be UUIDs (e.g., 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'). Please call get-users-on-site first to get the correct user ID."
                            messages.append(
                                ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call["id"]
                                )
                            )
                            continue

                    # Find and execute the tool
                    tool_result = None
                    for tool in langchain_tools:
                        if tool.name == tool_name:
                            tool_result = await tool.ainvoke(tool_args)
                            break

                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"

                    print(f"   Result: {tool_result[:200]}...")

                    # Enhanced error detection for user removal
                    if native_name == "admin-users" and tool_args.get("operation") == "remove-user-from-site":
                        try:
                            result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                            if isinstance(result_data, dict) and result_data.get("isError"):
                                error_text = str(result_data.get("content", [{}])[0].get("text", "Unknown error"))
                                print(f"   ⚠️ Removal failed: {error_text}")
                                if "404" in error_text or "not found" in error_text.lower():
                                    user_id = tool_args.get("userId", "unknown")
                                    tool_result = json.dumps({
                                        "content": [{
                                            "type": "text",
                                            "text": f"User ID {user_id} not found. This could mean: 1) The user was already removed, 2) The user ID is incorrect, or 3) You don't have permission to remove this user. Please verify the user still exists by calling get-users-on-site again."
                                        }],
                                        "isError": True
                                    })
                        except:
                            pass  # If parsing fails, use original result

                    # Store for response
                    tool_results.append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result
                    })

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        )
                    )

            # Max iterations reached
            print(f"⚠️ Max iterations ({max_iterations}) reached")
            return {
                "response": "Admin operation completed but reached maximum iteration limit. Some actions may be incomplete.",
                "tool_results": tool_results,
                "iterations": iterations
            }

    except Exception as e:
        print(f"❌ Admin Agent error: {str(e)}")
        return {
            "response": f"Admin operation failed: {str(e)}",
            "tool_results": [],
            "iterations": 0
        }
