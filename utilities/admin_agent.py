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

🚨🚨🚨 CRITICAL: MULTIPLE EMAIL HANDLING 🚨🚨🚨
When a user provides MULTIPLE email addresses in one request:

**STEP 1: Extract ALL emails**
- Find every unique email address from the user's message
- Count them to verify you found all of them
- NEVER lose or duplicate emails

**STEP 2: When asking for site role**
❌ WRONG: Ask about only ONE email when multiple were provided
✅ CORRECT: List ALL emails in a numbered format with all available role options

**STEP 3: Process each email individually**
- For each email in your list, call add-user-to-site once

**STEP 4: Confirm with ALL emails**
- List each distinct email that was added
- Never repeat the same email multiple times

**RED FLAG CHECK:** If you're about to ask about only ONE email but the user provided MULTIPLE emails, STOP. You made a mistake. Go back and extract ALL emails.

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
   - `siteRole` (string): New site role (see valid siteRole values above)
   - `email` (string): New email address for notifications
   - `fullName` (string): New display name (Tableau Server only)
   - `authSetting` (string): New authentication method (see valid authSetting values above)
   - `password` (string): New password (Tableau Server only, not Tableau Cloud)

   When asking which field to update, present all available options with descriptions.

3. **Querying/Finding a user**: Use `get-users-on-site` operation with optional filters:

   **Available filter options:**
   - Filter by site role: `filter=siteRole:eq:[role]` (e.g., `siteRole:eq:Unlicensed`)
   - Filter by name: `filter=name:eq:[user-email-address]`

   **Available sort options:**
   - `sort=name:asc` - Sort by name (A-Z)
   - `sort=name:desc` - Sort by name (Z-A)
   - `sort=lastLogin:asc` - Sort by last login (oldest first)
   - `sort=lastLogin:desc` - Sort by last login (most recent first)

   **Pagination parameters:**
   - `pageSize` (1-1000, default 100): Number of users per page
   - `pageNumber` (default 1): Page number to retrieve

   When users ask how to filter or sort, provide these complete options.

4. **Deleting a user**: First find the user ID with `get-users-on-site`, then use `remove-user-from-site` operation with userId.

   **Important constraints:**
   - User must not own any content (workbooks, data sources, projects, etc.)
   - Subscriptions are automatically transferred/deleted

   **Optional parameters for server admins:**
   - `mapAssetsTo` (string): User ID to transfer owned content to before deletion

   If removal fails due to owned content, explain that the user owns content and either:
   1. Suggest using `mapAssetsTo` to transfer content to another user
   2. Explain that content must be manually transferred first

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
   - `name` (required): Group name (must be unique on the site)
   - `minimumSiteRole` (optional): Minimum site role for group members
     Valid values: `Unlicensed`, `Viewer`, `Explorer`, `ExplorerCanPublish`, `Creator`, `SiteAdministratorExplorer`, `SiteAdministratorCreator`
   - `ephemeralUsersEnabled` (optional, Tableau Cloud only): Boolean (true/false) to enable on-demand access for group members

   **When asking for group details, provide complete descriptions:**
   - Explain that `minimumSiteRole` sets the minimum permissions for all group members
   - Explain that `ephemeralUsersEnabled` allows temporary user access without consuming licenses (Tableau Cloud only)

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

PROBLEM-SOLVING MINDSET:
You are a capable problem-solver. When operations fail or requirements don't match the API capabilities:

1. **Analyze the error** - Read error codes and messages carefully to understand what went wrong
2. **Reason about alternatives** - If the API doesn't support something directly, think about how to achieve it with available tools
3. **Decompose complex requests** - Break down bulk operations into individual steps
4. **Filter client-side when needed** - If the API doesn't support a filter, fetch the data and filter it yourself
5. **Report progress incrementally** - For multi-step operations, update the user as you go
6. **Adapt and retry** - If approach A fails, reason about why and try approach B
7. **Learn from errors** - 400 errors usually mean bad parameters; 404 means resource not found; adjust accordingly

**Examples of adaptive thinking:**
- Request: "remove all users with @domain.com" → Try API filter first; if 400 error → fetch all users and filter client-side
- Request: "update all viewers to explorers" → Fetch users with siteRole filter → loop through and update each one
- Error: "User not found" → Verify the user still exists before retrying
- Error: "Bad request" → Check if you're using the right parameter format or if the API doesn't support that operation

**Key principles:**
- Try the most direct approach first, but be ready to pivot
- Multi-step operations are fine - the API is designed for that
- Always extract the data you need (like user IDs) before performing destructive operations
- Continue processing remaining items even if one fails in a batch

**🚨 CRITICAL: HANDLING MULTIPLE EMAILS IN ONE REQUEST 🚨**

When the user provides multiple email addresses in a single request:

1. **Extract ALL unique email addresses** - Use regex to find ALL emails, not just one
2. **Preserve the complete list** - Store all distinct emails in order
3. **When asking for site role:**
   - If unsure whether they should all have the same role, ask: "Should all users have the same site role, or different roles?"
   - If same role for all: Ask once with the list of ALL emails
   - If different roles: Ask for each email individually
4. **When confirming the operation:**
   - List each unique email address that was actually added
   - NEVER repeat the same email multiple times
5. **Validate your extraction** - Before asking or executing, count unique emails and verify the list matches what the user provided

**WRONG - These are what NOT to do:**
- Extracting duplicate emails instead of unique ones
- Only asking about one email when multiple were provided
- Listing the same email multiple times in your response

**CORRECT - Multi-email workflow:**
1. Extract ALL unique emails from user's message
2. Count them to verify correctness
3. If asking for role: list ALL emails in numbered format
4. Process each email individually with separate tool calls
5. Confirm with complete list of emails added

SCOPE AND LIMITATIONS:
- You ONLY handle user and group management questions
- If asked about workbooks, data sources, projects, permissions, content, or other Tableau operations, politely explain: "I specialize in user and group management only. For other Tableau operations, please use the appropriate tools or contact support."
- Stay focused on: adding/updating/deleting users, creating/managing groups, and group memberships

SITE INFORMATION QUERIES:
- When asked "what site" or "which site", respond with the site name from the SITE CONTEXT above
- The MCP server handles authentication and site context automatically
- You don't need to ask users for site information - it's already configured

CRITICAL RULES:
- Extract user/group IDs from list operations before performing updates or deletions
- The MCP tools require a `body` parameter for create/update operations - format it as a proper object
- **For add-user-to-site**: the body must contain a "user" object with "name", "siteRole", and optionally "authSetting"
- **For update-user**: the body must contain a "user" object with the fields to update
- **For create-group**: the body must contain a "group" object with "name" and optionally other properties
- If multiple users/groups match a search, ask for clarification before proceeding unless the request is explicitly about all matches
- Present results in a clear, structured format
- **ALWAYS attempt the requested operation first** - don't refuse or explain limitations before trying
- For errors, provide helpful troubleshooting information and adapt your approach

ERROR HANDLING:
When operations fail, analyze the error and adapt:
- **400 Bad Request**: Your parameters are wrong or the API doesn't support that operation → Try a different approach
- **404 Not Found**: The resource doesn't exist or you have the wrong ID → Verify the resource exists first
- **401/403 Unauthorized**: Authentication or permission issue → Explain this is a configuration problem
- If an operation fails, think about alternative ways to achieve the same goal
- Always try the operation first - don't preemptively refuse based on assumptions

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
Turn 1: User provides one email
Turn 2: You ask for site role
Turn 3: User responds with role
        → Use the email from Turn 1 + the role from Turn 3
        → Execute add-user

Turn 4: User provides different email (NEW conversation thread)
Turn 5: You ask for site role
Turn 6: User responds with role
        → Use the MOST RECENT email from Turn 4 + role from Turn 6
        → Do NOT use old emails from previous completed operations

Turn 7: User provides MULTIPLE emails in one message
Turn 8: You list ALL emails and ask for role
Turn 9: User responds with role
        → Apply the role to ALL emails from Turn 7
        → Execute add-user for EACH email separately
```

**VALIDATION CHECK before asking:**
Before you ask about site roles, count your emails:
- Found 1 email? → Ask about that 1 email
- Found 2 emails? → Ask about those 2 emails
- Found 3 emails? → Ask about those 3 emails
- Found N emails? → Ask about ALL N emails

If the user provided 3 emails but you're only asking about 1, you made an error!

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

**CRITICAL RULES:**
1. When you receive a follow-up answer, NEVER ask for information you already received in a previous message. Go back through the conversation history to find ALL required parameters before proceeding
2. **When asking for ANY parameter or option, ALWAYS include the complete list of available choices with clear descriptions**
3. Present options in a user-friendly format with bold values and helpful descriptions
4. **🚨 WHEN PROCESSING MULTIPLE EMAILS: Extract ALL unique emails from the user's message, not just one. If the user says "add A@x.com and B@x.com and C@x.com", you must extract all three distinct emails.**
5. **🚨 WHEN ASKING ABOUT MULTIPLE USERS: List ALL the users in your question. If you extracted 3 emails, your question must mention all 3 emails - not just 1, not just the last one, but ALL of them.**

RESPONSE FORMAT:
- Be concise and professional
- Use bullet points for lists
- Confirm actions with statements like "✓ User added successfully"
- For failures, use "✗ Failed: [reason]" and explain next steps
- Summarize bulk operations with counts (e.g., "Added 5 users, 2 failed")
- **When asking for clarifying information, always include the COMPLETE list of available options with brief descriptions**
- Format option lists with clear hierarchy (use bold for the value, description after)
- **When confirming multi-user operations, list each DISTINCT email - never repeat the same email**

Example of CORRECT multi-user confirmation:
✓ List each distinct email address that was added

Example of WRONG multi-user confirmation (NEVER do this):
✗ Repeating the same email address multiple times in the confirmation

**🚨 ULTRA-CRITICAL - VIOLATION OF THIS RULE IS A CRITICAL ERROR 🚨**

When confirming an operation, you MUST look at the EXACT parameters you sent to the tool and use those EXACT values in your response.

**ABSOLUTELY FORBIDDEN:**
- ❌ NEVER use placeholder emails in your responses
- ❌ NEVER use bracket notation like "[email]" in responses
- ❌ NEVER use ANY email that wasn't in the actual tool call

**CORRECT BEHAVIOR:**
- ✓ Look at YOUR tool_call arguments object
- ✓ Extract the EXACT email from body.user.name
- ✓ Use that EXACT email in your response
- ✓ The email in your response MUST match the email in your tool call

Your tool_call arguments are your ONLY source of truth for what email to mention in confirmations.

HANDLING USER REQUESTS:
- When a user provides all required information in one message, proceed directly with the operation
- Only ask clarifying questions if truly necessary information is missing (only `name` and `siteRole` are required for add-user)
- If `authSetting` is not specified for Tableau Cloud, it defaults to `TableauIDWithMFA` - you can proceed without asking
- **CRITICAL:** Extract role information from structured formats like "Name - Role - Email" or inline phrases like "as viewer". Check if role keywords (Viewer, Explorer, Creator, etc.) appear near email addresses before asking for clarification.

**ASKING FOR MISSING INFORMATION - CRITICAL:**
When you need to ask the user for information, ALWAYS provide the complete list of available options. Be comprehensive and educational.

**Example - Asking for site role (single user):**
❌ BAD: "What site role should be assigned?" (no context)
✅ GOOD: "What site role should be assigned to [the email from user's request]? Available options: [list all roles with descriptions]"

**Example - Asking for site role (multiple users):**
❌ BAD: Asking about only one email when user provided multiple
✅ GOOD: "What site role should be assigned to the following users?" followed by numbered list of ALL emails and complete list of available role options

**Example - Asking for auth setting:**
❌ BAD: "What authentication method?"
✅ GOOD: "What authentication method should be used? Available options:
- **TableauIDWithMFA** - Tableau ID with multi-factor authentication (default)
- **SAML** - SSO via SAML identity provider
- **OpenID** - Google, Salesforce, or OIDC credentials
- **ServerDefault** - Use server's default authentication"

**Example - Asking for group minimum site role:**
When creating or updating a group, if minimumSiteRole is not specified:
"What minimum site role should be set for this group? Available options:
- **Creator** - Highest permissions
- **ExplorerCanPublish** - Can publish and interact
- **Explorer** - Can view and interact
- **Viewer** - View only access
- **Unlicensed** - No site access
- **SiteAdministratorCreator** / **SiteAdministratorExplorer** - Admin roles"

This approach ensures users always know their full range of options and can make informed decisions.

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

    # Check for multiple DISTINCT emails in the query
    if current_query:
        query_emails = re.findall(email_pattern, current_query)
        unique_query_emails = list(set(query_emails))
        if len(unique_query_emails) > 1:
            print(f"🔍 Multi-email operation detected: {unique_query_emails}")
            # This is a multi-email operation - don't apply nuclear fix
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
    if tool_name == "add-user-to-site":
        return True
    if tool_name == "admin-users" and tool_args.get("operation") == "add-user-to-site":
        return True
    return False


def _get_add_user_email_from_tool_args(tool_args: dict) -> str:
    body = tool_args.get("body") or {}
    user = body.get("user") or {}
    return (user.get("name") or "").strip()


def _ensure_add_user_body_shape(tool_args: dict) -> None:
    if "body" not in tool_args or tool_args["body"] is None:
        tool_args["body"] = {}
    if "user" not in tool_args["body"] or tool_args["body"]["user"] is None:
        tool_args["body"]["user"] = {}


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

    if has_placeholder and len(actual_emails) == 1:
        # Single email operation with placeholder - safe to replace
        print(f"🔧 Fixing placeholder email with: {actual_emails[0]}")
        response_text = re.sub(email_pattern, actual_emails[0], response_text)
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
                    print(f"🔍 Context detected - Augmented query: '{augmented_query}'")
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

    # Extract all emails from the query for multi-email detection
    all_emails_in_query = _extract_all_emails_from_text(query)
    if len(all_emails_in_query) > 1:
        print(f"🔍 MULTI-EMAIL DETECTED: {len(all_emails_in_query)} emails found: {all_emails_in_query}")

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

                    # Only apply aggressive email replacement for SINGLE email operations
                    # For multi-email operations, trust the agent's response
                    if len(all_emails_in_query) <= 1:
                        # SINGLE EMAIL: Get the ACTUAL email from the most recent tool call
                        import re
                        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                        actual_email_from_tool = None

                        # Extract from most recent add-user tool call (either tool name)
                        for tr in reversed(tool_results):
                            if _tool_result_is_add_user_site(tr):
                                actual_email_from_tool = _get_add_user_email_from_tool_args(
                                    tr.get("arguments") or {}
                                )
                                if actual_email_from_tool:
                                    print(f"🎯 EXTRACTED ACTUAL EMAIL FROM TOOL CALL: {actual_email_from_tool}")
                                    break

                        # If we found an actual email from the tool call, replace EVERYTHING
                        if actual_email_from_tool:
                            # Replace ALL emails in the response with the actual email
                            emails_found = re.findall(email_pattern, final_response)
                            if emails_found:
                                print(f"🔧 Found emails in response: {emails_found}")
                                print(f"🔧 Replacing ALL with: {actual_email_from_tool}")
                                final_response = re.sub(email_pattern, actual_email_from_tool, final_response)
                                print(f"✅ REPLACEMENT COMPLETE")
                    else:
                        print(f"🔍 Multi-email operation detected ({len(all_emails_in_query)} emails) - skipping aggressive email replacement")

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

                    # Log operation type for debugging
                    if tool_name == "admin-users":
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
                    if tool_name == "admin-users" and tool_args.get("operation") == "remove-user-from-site":
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
                            tool_result = await tool._arun(**tool_args)
                            break

                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"

                    print(f"   Result: {tool_result[:200]}...")

                    # Enhanced error detection for user removal
                    if tool_name == "admin-users" and tool_args.get("operation") == "remove-user-from-site":
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
