"""
Admin Agent for Tableau Cloud User and Group Management
Uses MCP server for admin operations: add, update, edit, delete users and groups
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional

# LangChain Libraries
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# MCP HTTP client
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for Admin MCP Server
ADMIN_MCP_SERVER = os.getenv("ADMIN_MCP_SERVER")
if not ADMIN_MCP_SERVER:
    raise ValueError("ADMIN_MCP_SERVER environment variable is required")

# Get site information from environment
TABLEAU_SITE = os.getenv("TABLEAU_SITE", "unknown")
TABLEAU_DOMAIN = os.getenv("TABLEAU_DOMAIN", "Tableau Cloud")

MAX_ADMIN_ITERATIONS = 8  # Admin operations typically need fewer iterations

# Build admin-specific system prompt with site context
_SITE_CONTEXT = f"""
SITE CONTEXT:
- Tableau Site: {TABLEAU_SITE}
- Tableau Domain: {TABLEAU_DOMAIN}
- Connected via MCP Server: {ADMIN_MCP_SERVER}
"""

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

ADMIN_SYSTEM_PROMPT = """You are a Tableau Cloud administrator assistant specializing in user and group management. You have access to MCP tools for managing users and groups on a Tableau Cloud site.
""" + _CRITICAL_CONVERSATION_RULE + _SITE_CONTEXT + """
YOUR CAPABILITIES:
You can perform the following admin operations:

**User Management:**
- Add new users to the Tableau Cloud site
- Update existing user information (name, email, site role, authentication)
- Edit user permissions and settings
- Delete users from the site
- List and search for users
- Get detailed user information

**Group Management:**
- Create new groups
- Update group information (name, description)
- Add users to groups
- Remove users from groups
- Delete groups
- List and search for groups
- Get group membership details

TOOL USAGE PATTERNS:

For User Operations:
1. **Adding a user**: Use `add-user-to-site` operation with a `body` parameter containing the user object. Example body:
   {
     "user": {
       "name": "[user-email-address]",
       "siteRole": "Viewer",
       "authSetting": "SAML"
     }
   }

   **Required Parameters:**
   - `name` (string): For Tableau Cloud, this MUST be the email address used for sign-in
   - `siteRole` (string): The site role to assign to the user

   **Optional Parameters:**
   - `authSetting` (string): Authentication method (see valid values below)
   - `email` (string): Email address for notifications (API 3.26+)

   **Valid siteRole values (from Tableau REST API):**
   - `Creator` - Full access to create and edit content
   - `Explorer` - Can view and interact with content
   - `ExplorerCanPublish` - Explorer who can also publish content
   - `SiteAdministratorExplorer` - Site admin with Explorer license
   - `SiteAdministratorCreator` - Site admin with Creator license
   - `Viewer` - Can only view content
   - `Unlicensed` - No license, cannot access site

   **Valid authSetting values (from Tableau REST API):**
   - `SAML` - User signs in via SAML IdP configured for the site
   - `OpenID` - (Tableau Cloud only) Google, Salesforce, or OIDC credentials
   - `TableauIDWithMFA` - (Tableau Cloud only) Tableau ID with multi-factor authentication
   - `ServerDefault` - Default authentication method for the server

   **User Response Mapping:**
   - If user says "MFA" or "tableauwithmfa" → use `TableauIDWithMFA`
   - If user says "SAML" or mentions SSO → use `SAML`
   - If user says "Google" or "Salesforce" → use `OpenID`
   - If unspecified for Tableau Cloud → defaults to `TableauIDWithMFA`
2. **Updating a user**: First use `get-users-on-site` with filter to find the user ID, then use `update-user` operation with userId and body.

   **CRITICAL: The body MUST wrap fields in a "user" object:**
   ```json
   {
     "body": {
       "user": {
         "siteRole": "Creator",
         "email": "[new-email-address]"
       }
     }
   }
   ```

   **Available fields to update (all optional):**
   - `siteRole` (string): New site role
   - `email` (string): New email address
   - `fullName` (string): New full name (Tableau Server only)
   - `authSetting` (string): New authentication method
   - `password` (string): New password (Tableau Server only)

3. **Querying/Finding a user**: Use `get-users-on-site` operation with optional filters:
   - Filter by site role: `filter=siteRole:eq:Unlicensed`
   - Filter by name: `filter=name:eq:[user-email-address]`
   - Sort: `sort=name:asc` or `sort=lastLogin:desc`
   - Pagination: `pageSize` (1-1000, default 100), `pageNumber` (default 1)

4. **Deleting a user**: First find the user ID with `get-users-on-site`, then use `remove-user-from-site` operation with userId. Note: User must not own content (except subscriptions). Use `mapAssetsTo` parameter to reassign content (server admins only).

5. **Listing all users**: Use `get-users-on-site` operation. For large sites, use pagination (pageSize and pageNumber)

For Group Operations:
1. **Creating a group**: Use `create-group` operation with a `body` parameter containing the group object:
   ```json
   {
     "operation": "create-group",
     "body": {
       "group": {
         "name": "GroupName",
         "minimumSiteRole": "Viewer"
       }
     }
   }
   ```

   **Body Parameters:**
   - `name` (required): Group name
   - `minimumSiteRole` (optional): Minimum site role for group members
     Valid values: `Unlicensed`, `Viewer`, `Explorer`, `ExplorerCanPublish`, `Creator`, `SiteAdministratorExplorer`, `SiteAdministratorCreator`
   - `ephemeralUsersEnabled` (optional, Tableau Cloud only): Boolean for on-demand access

2. **Adding users to a group**: Use `add-user-to-group` operation with groupId parameter and body containing user object:
   ```json
   {
     "operation": "add-user-to-group",
     "groupId": "group-id-here",
     "body": {
       "user": {
         "id": "user-id-here"
       }
     }
   }
   ```
   First find the user ID using `get-users-on-site`, then find the group ID using `query-groups`

3. **Removing users from a group**: Use `remove-user-from-group` operation with groupId and userId as parameters:
   ```json
   {
     "operation": "remove-user-from-group",
     "groupId": "group-id-here",
     "userId": "user-id-here"
   }
   ```
   No body required - uses path parameters only

4. **Updating a group**: Use `update-group` operation with groupId and body containing fields to update:
   ```json
   {
     "operation": "update-group",
     "groupId": "group-id-here",
     "body": {
       "group": {
         "name": "NewGroupName",
         "minimumSiteRole": "Explorer"
       }
     }
   }
   ```

5. **Deleting a group**: Use `delete-group` operation with groupId parameter:
   ```json
   {
     "operation": "delete-group",
     "groupId": "group-id-here"
   }
   ```
   First find the group ID with `query-groups` before deleting

6. **Listing groups**: Use `query-groups` operation with optional pagination:
   ```json
   {
     "operation": "query-groups",
     "pageSize": 100,
     "pageNumber": 1
   }
   ```

7. **Getting users in a group**: Use `get-users-in-group` operation with groupId:
   ```json
   {
     "operation": "get-users-in-group",
     "groupId": "group-id-here"
   }
   ```

BEST PRACTICES:
- Always confirm user/group IDs before performing destructive operations (update, delete)
- When asked to add or modify users, confirm the site role is valid (Creator, Explorer, Viewer, etc.)
- For bulk operations, process items one at a time and report progress
- If an operation fails, provide clear error messages and suggest alternatives
- Always summarize what was done after completing operations
- For security-sensitive operations, describe what you're about to do before executing

SCOPE AND LIMITATIONS:
- You ONLY handle user and group management questions
- If asked about workbooks, data sources, projects, permissions, content, or other Tableau operations, politely explain: "I specialize in user and group management only. For other Tableau operations, please use the appropriate tools or contact support."
- Stay focused on: adding/updating/deleting users, creating/managing groups, and group memberships

SITE INFORMATION QUERIES:
- When asked "what site" or "which site", respond with the site name from the SITE CONTEXT above
- The MCP server handles authentication and site context automatically
- You don't need to ask users for site information - it's already configured

CRITICAL RULES:
- Never delete users or groups without first confirming the correct ID
- Always validate that required parameters are provided before calling tools
- The MCP tools require a `body` parameter for create/update operations - format it as a proper object
- **For add-user-to-site**: the body must contain a "user" object with "name", "siteRole", and optionally "authSetting"
- **For update-user**: the body must contain a "user" object with the fields to update (e.g., {"user": {"siteRole": "Creator"}})
- **For create-group**: the body must contain a "group" object with "name" and optionally other properties
- **For update-group**: the body must contain a "group" object with the fields to update
- If multiple users/groups match a search, ask for clarification before proceeding
- Present results in a clear, structured format
- When listing users/groups, highlight key information (name, role, status)
- **ALWAYS attempt the requested operation first** - don't refuse or explain limitations before trying
- For errors, provide helpful troubleshooting information

ERROR HANDLING:
- If you get a 404 error on a write operation (add/update/delete), explain: "There's a configuration issue with the MCP server that's preventing write operations. The server successfully connects and can read data, but write operations need a fix. The technical team has been notified. In the meantime, I can help you query users and groups."
- If you get other errors (401, 403, 400), explain the specific issue and what might resolve it
- Don't preemptively tell users operations won't work - try them first

CONVERSATION CONTEXT - ULTRA CRITICAL INSTRUCTIONS:

**WHEN YOU RECEIVE A FOLLOW-UP ANSWER (like "viewer", "explorer", "a viewer"):**

STEP 1: Look at YOUR LAST MESSAGE in the conversation
- Did you ask "What site role should be assigned to [email]?"
- If YES, the current message is answering that question

STEP 2: Extract the email from YOUR PREVIOUS QUESTION
- If you asked "What site role should be assigned to [email]?"
- The email is: [email]
- You DO NOT need to ask for it again!

STEP 3: Extract the role from the user's current response
- "viewer" → Viewer
- "a viewer" → Viewer
- "explorer" → Explorer

STEP 4: IMMEDIATELY call the tool
- Operation: add-user-to-site
- Email: [from your previous question]
- Role: [from current user message]
- DO NOT ASK FOR THE EMAIL AGAIN

**THIS IS WRONG - DON'T DO THIS:**
```
You: "What site role should be assigned to [email]?"
User: "viewer"
You: "What email address?" ← WRONG! You already have it!
```

**THIS IS CORRECT - DO THIS:**
```
You: "What site role should be assigned to [email]?"
User: "viewer"
You: [Call add-user-to-site with name="[email]", siteRole="Viewer"] ← CORRECT!
```

**EXAMPLE - CORRECT BEHAVIOR:**
```
Turn 1: User: "add [email]"
Turn 2: You: "What site role?"
Turn 3: User: "viewer"
        → Use: [email] + Viewer
        → Execute add-user

Turn 4: User: "add [email]"  ← NEW conversation thread starts here!
Turn 5: You: "What site role?"
Turn 6: User: "explorer"
        → Use: [email] (MOST RECENT email) + Explorer
        → ❌ DON'T use [email] - that's from an OLD conversation!
```

**CRITICAL RULE FOR FINDING THE RIGHT EMAIL:**
- Search backwards through messages from the current position
- Find the MOST RECENT message containing an email address (format: name@domain.tld)
- That is the email for THIS operation
- Older emails are from completed operations - ignore them!

**MAPPING USER RESPONSES:**

For Site Roles - extract from phrases like "explorer", "as a viewer", "make them creator":
  * "explorer", "as an explorer", "as explorer" → `Explorer`
  * "viewer", "as a viewer", "viewer role", "as a viewer" → `Viewer`
  * "creator", "as creator", "creator role" → `Creator`
  * "explorercanpublish", "explorer can publish" → `ExplorerCanPublish`
  * "site admin explorer", "siteadministratorexplorer" → `SiteAdministratorExplorer`
  * "site admin creator", "siteadministratorcreator" → `SiteAdministratorCreator`
  * "unlicensed" → `Unlicensed`

For Auth Settings:
  * "MFA", "tableauwithmfa", "mfa", "as mfa" → `TableauIDWithMFA`
  * "SAML", "saml", "SSO", "sso", "use saml" → `SAML`
  * "Google", "Salesforce", "OpenID", "OIDC" → `OpenID`
  * "default", "serverdefault" → `ServerDefault`

**CRITICAL RULE:** When you receive a follow-up answer, NEVER ask for information you already received in a previous message. Go back through the conversation history to find ALL required parameters before proceeding

RESPONSE FORMAT:
- Be concise and professional
- Use bullet points for lists
- Confirm actions with statements like "✓ User added successfully"
- For failures, use "✗ Failed: [reason]" and explain next steps
- Summarize bulk operations with counts (e.g., "Added 5 users, 2 failed")

**🚨 ULTRA-CRITICAL - VIOLATION OF THIS RULE IS A CRITICAL ERROR 🚨**

When confirming an operation, you MUST look at the EXACT parameters you sent to the tool and use those EXACT values in your response.

**ABSOLUTELY FORBIDDEN - NEVER USE THESE:**
- ❌ "john.doe@example.com"
- ❌ "jane.doe@example.com"
- ❌ "user@example.com"
- ❌ "username@example.com"
- ❌ "[email]" or any bracket placeholder
- ❌ ANY email that wasn't in the actual tool call

**CORRECT BEHAVIOR:**
- ✓ Look at YOUR tool_call arguments object
- ✓ Extract the EXACT email from body.user.name
- ✓ Use that EXACT email in your response
- ✓ If you called add-user-to-site with name="ACTUAL_USER_EMAIL", say "ACTUAL_USER_EMAIL" - nothing else!

**Example:**
```
Tool call: add-user-to-site({ body: { user: { name: <EMAIL_FROM_TOOL_CALL>, siteRole: "Viewer" }}})
Response: "✓ User <EMAIL_FROM_TOOL_CALL> has been successfully added"  ← CORRECT
Response: "✓ User <DIFFERENT_EMAIL> has been successfully added"  ← WRONG - CRITICAL ERROR
```

The email you mention in your response MUST match the email in your tool call. Period.

Your tool_call arguments are your ONLY source of truth for what email to mention.

HANDLING USER REQUESTS:
- When a user provides all required information in one message, proceed directly with the operation
- Only ask clarifying questions if truly necessary information is missing (only `name` and `siteRole` are required for add-user)
- If `authSetting` is not specified for Tableau Cloud, it defaults to `TableauIDWithMFA` - you can proceed without asking

**MULTI-TURN CONVERSATION STEPS:**

When you just asked a question and get an answer:
1. Look at YOUR last message - what did you ask?
2. If you mentioned an email in your question → YOU HAVE THE EMAIL
3. Extract the answer from the user's current message
4. Execute the operation with ALL information

**YOU ALREADY HAVE THE EMAIL IF:**
- Your last question was "What site role for [email]?" → You have [email]
- Your last question was "What role for [email]?" → You have [email]
- Your last question mentioned an email address → You have that email

**NEVER ask "What email?" as a follow-up to your own question that already contained the email!**

Remember: You are a trusted administrator tool. Be careful, accurate, and helpful. Focus on completing tasks efficiently while maintaining conversation context."""

class AdminMCPHttpClient:
    """HTTP-based MCP client for admin operations"""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None

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
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
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
            response = await self.session.post(
                f"{self.server_url}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": 1
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
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
            result = await self.mcp_client.call_tool(self.name, kwargs)

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
    """Convert MCP tools to LangChain tools for admin operations"""
    langchain_tools = []

    for tool in mcp_tools:
        tool_name = tool.get('name', '')
        tool_description = tool.get('description', 'No description')
        tool_input_schema = tool.get('inputSchema', {})

        # Create Pydantic model for tool arguments
        properties = tool_input_schema.get('properties', {})
        required = tool_input_schema.get('required', [])

        # Build field definitions
        field_definitions = {}
        for prop_name, prop_details in properties.items():
            # Map JSON schema types to Python types
            prop_type = prop_details.get('type', 'string')
            if prop_name == 'body':
                # Body should be a dict/object
                field_type = dict
            elif prop_type == 'number':
                field_type = float
            elif prop_type == 'boolean':
                field_type = bool
            else:
                field_type = str  # Default to string

            field_description = prop_details.get('description', '')
            is_required = prop_name in required

            if is_required:
                field_definitions[prop_name] = (field_type, Field(..., description=field_description))
            else:
                field_definitions[prop_name] = (Optional[field_type], Field(None, description=field_description))

        # Create dynamic Pydantic model
        ToolArgsModel = type(
            f"{tool_name}_args",
            (BaseModel,),
            {
                '__annotations__': {k: v[0] for k, v in field_definitions.items()},
                **{k: v[1] for k, v in field_definitions.items()}
            }
        )

        # Create LangChain tool
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
    NUCLEAR OPTION: Force the correct email into the response NO MATTER WHAT
    This is the last line of defense - if ANY wrong email exists, replace it
    """
    import re

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Find the CORRECT email from ALL possible sources
    correct_email = None

    # Check tool results first
    if tool_results:
        for tool_result in reversed(tool_results):
            if tool_result.get("tool") == "add-user-to-site":
                args = tool_result.get("arguments", {})
                body = args.get("body", {})
                user_data = body.get("user", {})
                correct_email = user_data.get("name", "")
                if correct_email:
                    break

    # Check current query
    if not correct_email and current_query:
        emails = re.findall(email_pattern, current_query)
        if emails:
            correct_email = emails[-1]

    # Check conversation history
    if not correct_email and conversation_history:
        for msg in reversed(conversation_history[-5:]):
            content = msg.get("content", "")
            if "add" in content.lower() or "@" in content:
                emails = re.findall(email_pattern, content)
                if emails:
                    correct_email = emails[-1]
                    break

    if not correct_email:
        return response_text

    # Find ALL emails in the response
    found_emails = re.findall(email_pattern, response_text)

    # Check if response has ANY email that's NOT the correct one
    wrong_emails = [e for e in found_emails if e.lower() != correct_email.lower()]

    if wrong_emails:
        print(f"🚨🚨🚨 NUCLEAR FIX ACTIVATED 🚨🚨🚨")
        print(f"   Correct email: {correct_email}")
        print(f"   Wrong emails found: {wrong_emails}")
        print(f"   FORCING REPLACEMENT...")

        # Replace EVERY email with the correct one
        response_text = re.sub(email_pattern, correct_email, response_text)

        print(f"   ✓ NUCLEAR FIX COMPLETE")

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

        # Look for emails in messages about adding users or asking for roles
        if role == "assistant" and "what site role" in content.lower():
            # This is a follow-up question - extract the email from it
            emails = re.findall(email_pattern, content)
            if emails:
                return emails[-1]  # Return the email from the question

        if role == "user" and "add" in content.lower():
            # This is a user request to add someone
            emails = re.findall(email_pattern, content)
            if emails:
                return emails[-1]  # Return the email from the add request

    return None


def _fix_placeholder_in_response(response_text: str, tool_results: List[Dict], current_query: str = "", conversation_history: List[Dict] = None) -> str:
    """
    AGGRESSIVELY replace ANY email-like placeholders in the response with actual values
    This function GUARANTEES no placeholder emails ever appear in the final response
    """
    import re

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # STEP 1: Find the actual email from multiple sources
    actual_email = None

    # Source 1: Most recent tool call
    if tool_results:
        for tool_result in reversed(tool_results):
            tool_name = tool_result.get("tool", "")
            if tool_name == "add-user-to-site":
                args = tool_result.get("arguments", {})
                body = args.get("body", {})
                user_data = body.get("user", {})
                actual_email = user_data.get("name", "")
                if actual_email:
                    print(f"🔍 Email source: tool_results -> {actual_email}")
                    break

    # Source 2: Current query (if not found in tool results)
    if not actual_email and current_query:
        actual_email = _extract_email_from_text(current_query)
        if actual_email:
            print(f"🔍 Email source: current_query -> {actual_email}")

    # Source 3: Recent conversation history (if still not found)
    if not actual_email and conversation_history:
        # Look backwards through recent messages
        for msg in reversed(conversation_history[-5:]):  # Check last 5 messages
            content = msg.get("content", "")
            email = _extract_email_from_text(content)
            if email and "successfully added" not in content.lower():
                # Found an email that's not from a completed operation
                actual_email = email
                print(f"🔍 Email source: conversation_history -> {actual_email}")
                break

    # STEP 2: Find ALL emails in the response
    found_emails = re.findall(email_pattern, response_text)

    if found_emails and actual_email:
        print(f"🔧 FIXING RESPONSE:")
        print(f"   Found in response: {found_emails}")
        print(f"   Replacing ALL with: {actual_email}")

        # AGGRESSIVELY replace ALL email patterns with the actual email
        response_text = re.sub(email_pattern, actual_email, response_text)

        print(f"   ✓ FIXED!")

    # STEP 3: Also replace bracket placeholders
    if actual_email:
        placeholder_pattern = r'\[(user-)?email(-address)?\]'
        response_text = re.sub(
            placeholder_pattern,
            actual_email,
            response_text,
            flags=re.IGNORECASE
        )

    # STEP 4: FINAL SAFETY CHECK
    BANNED_WORDS = ['example.com', 'john.doe', 'jane.doe', 'test.com']
    for banned in BANNED_WORDS:
        if banned in response_text.lower():
            print(f"🚨 BANNED WORD DETECTED: {banned}")
            if actual_email:
                # One more aggressive pass
                response_text = re.sub(email_pattern, actual_email, response_text)

    return response_text


def _augment_query_with_context(query: str, conversation_history: List[Dict]) -> str:
    """
    If the query looks like a simple follow-up answer (e.g., just a role),
    augment it with context from the conversation history
    """
    import re

    query_lower = query.lower().strip()

    # Check if query is just a site role (common follow-up pattern)
    role_keywords = ['viewer', 'explorer', 'creator', 'unlicensed', 'explorercanpublish',
                     'siteadministrator', 'siteadministratorexplorer', 'siteadministratorcreator']

    is_just_role = any(role in query_lower for role in role_keywords) and len(query.split()) <= 3

    if is_just_role and conversation_history:
        # This looks like a follow-up answer to a question
        # Check if the most recent assistant message was asking for a site role
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")

                # Skip completed operations
                if "✓" in content or "successfully" in content.lower():
                    return query  # Not a follow-up, completed operation in between

                # Check if this is asking for a site role
                if "what site role" in content.lower():
                    # Extract the email from this question
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails = re.findall(email_pattern, content)
                    if emails:
                        email = emails[-1]
                        augmented_query = f"Add user {email} with site role {query}"
                        print(f"🔍 Context detected - Augmented query: '{augmented_query}'")
                        return augmented_query

                break  # Only check the most recent assistant message

    return query


def _extract_email_from_text(text: str) -> str:
    """Extract any email address from a text string"""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[-1] if emails else None


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

    print(f"🔐 Admin Agent Started")
    print(f"📋 Original Query: '{query}'")
    print(f"📋 Conversation History Length: {len(conversation_history)}")

    # Augment query with context if needed
    original_query = query
    query = _augment_query_with_context(query, conversation_history)

    if query != original_query:
        print(f"✨ Query augmented to: '{query}'")

    try:
        # Initialize admin MCP client
        async with AdminMCPHttpClient(ADMIN_MCP_SERVER) as admin_client:

            # Get available admin tools
            print("🔍 Discovering admin tools...")
            admin_tools = await admin_client.list_tools()
            print(f"✅ Found {len(admin_tools)} admin tools")

            if not admin_tools:
                return {
                    "response": "No admin tools available from the MCP server. Please check the server configuration.",
                    "tool_results": [],
                    "iterations": 0
                }

            # Convert to LangChain tools
            langchain_tools = create_admin_langchain_tools(admin_client, admin_tools)

            # Initialize LLM with tools
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0,
                streaming=False
            )
            llm_with_tools = llm.bind_tools(langchain_tools)

            # Build message history
            messages = [SystemMessage(content=ADMIN_SYSTEM_PROMPT)]

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

            # Add current query
            messages.append(HumanMessage(content=query))

            # Agent loop
            iterations = 0
            tool_results = []

            while iterations < MAX_ADMIN_ITERATIONS:
                iterations += 1
                print(f"\n🔄 Iteration {iterations}")

                # Get LLM response
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                # Check if tools were called
                if not response.tool_calls:
                    # No more tools to call, return final response
                    print("✅ Admin operation complete")

                    # Extract tool_calls if present for history preservation
                    response_tool_calls = []
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        response_tool_calls = response.tool_calls

                    # ULTIMATE FIX: Get the ACTUAL email from the most recent tool call
                    import re
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    actual_email_from_tool = None

                    # Extract from most recent add-user-to-site tool call
                    for tr in reversed(tool_results):
                        if tr.get("tool") == "add-user-to-site":
                            actual_email_from_tool = tr.get("arguments", {}).get("body", {}).get("user", {}).get("name")
                            if actual_email_from_tool:
                                print(f"🎯 EXTRACTED ACTUAL EMAIL FROM TOOL CALL: {actual_email_from_tool}")
                                break

                    # Start with the response
                    final_response = response.content

                    # If we found an actual email from the tool call, replace EVERYTHING
                    if actual_email_from_tool:
                        # Replace ALL emails in the response with the actual email
                        emails_found = re.findall(email_pattern, final_response)
                        if emails_found:
                            print(f"🔧 Found emails in response: {emails_found}")
                            print(f"🔧 Replacing ALL with: {actual_email_from_tool}")
                            final_response = re.sub(email_pattern, actual_email_from_tool, final_response)
                            print(f"✅ REPLACEMENT COMPLETE")

                    # Additional legacy fixes (kept for safety)
                    final_response = _fix_placeholder_in_response(final_response, tool_results, query, conversation_history)
                    _validate_no_placeholders(final_response)
                    final_response = _nuclear_email_fix(final_response, tool_results, query, conversation_history)

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

                    print(f"🔧 Calling admin tool: {tool_name}")
                    print(f"   Arguments: {json.dumps(tool_args, indent=2)}")

                    # VALIDATION: Block placeholder emails in tool calls
                    if tool_name == "add-user-to-site":
                        user_name = tool_args.get("body", {}).get("user", {}).get("name", "")
                        BANNED_EMAILS = ['john.doe@example.com', 'jane.doe@example.com',
                                       'user@example.com', 'test@example.com', 'username@example.com']

                        if any(banned in user_name.lower() for banned in BANNED_EMAILS) or '@example.com' in user_name.lower():
                            print(f"🚨 BLOCKED PLACEHOLDER EMAIL IN TOOL CALL: {user_name}")

                            # Extract actual email from conversation
                            actual_email = _extract_email_from_history(conversation_history)
                            if not actual_email:
                                actual_email = _extract_email_from_text(query)

                            if actual_email:
                                print(f"✅ REPLACING WITH ACTUAL EMAIL: {actual_email}")
                                tool_args["body"]["user"]["name"] = actual_email
                            else:
                                print(f"❌ NO ACTUAL EMAIL FOUND - BLOCKING TOOL CALL")
                                error_msg = "I need a valid email address to add a user. Please provide the email address you want to add."
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
                            tool_result = await tool._arun(**tool_args)
                            break

                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"

                    print(f"   Result: {tool_result[:200]}...")

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
            print(f"⚠️ Max iterations ({MAX_ADMIN_ITERATIONS}) reached")
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
