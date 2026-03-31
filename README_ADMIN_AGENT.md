# Tableau Admin Agent

An intelligent admin agent for managing Tableau Cloud users and groups using the Model Context Protocol (MCP).

## Overview

The Admin Agent provides a natural language interface for Tableau Cloud administrative tasks, including:

- **User Management**: Add, update, delete, and list users
- **Group Management**: Create, update, delete groups, and manage group memberships
- **Intelligent Operations**: The agent understands admin requests and executes the appropriate MCP tools

## Architecture

```
┌─────────────────┐
│   User Query    │
│ (Natural Lang.) │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│   Admin Agent       │
│   (LangChain +      │
│    GPT-4o)          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Admin MCP Server  │
│   (User & Group     │
│    Management)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Tableau Cloud      │
│  REST API           │
└─────────────────────┘
```

## Setup

### 1. Environment Variables

Add the following to your `.env` file:

```bash
# Admin MCP Server
ADMIN_MCP_SERVER='https://your-admin-mcp-server.example.com/admin-mcp'

# OpenAI (for the agent)
OPENAI_API_KEY='your-openai-api-key'
OPENAI_MODEL='gpt-4o'  # or gpt-4o-mini

# Optional: LangSmith for debugging
LANGCHAIN_TRACING='true'
LANGCHAIN_API_KEY='your-langsmith-key'
LANGCHAIN_PROJECT='tableau-admin-agent'
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Admin Agent

```bash
# Run as a web service on port 8001
python admin_web_app.py

# Or set custom port
ADMIN_PORT=9000 python admin_web_app.py
```

## Usage Examples

### User Management

```
"Add a new user john.doe@example.com with Creator role"

"List all users with Explorer role"

"Update user jane.smith@example.com to Viewer role"

"Delete user old.user@example.com"

"Show me details for user admin@example.com"
```

### Group Management

```
"Create a new group called 'Data Analysts'"

"Add user john.doe@example.com to the 'Marketing' group"

"List all groups"

"Remove user jane.smith@example.com from 'Sales' group"

"Delete the 'Old Team' group"

"Show me all users in the 'Executives' group"
```

### Bulk Operations

```
"Add the following users: john@example.com (Creator), jane@example.com (Explorer)"

"Remove all users from the 'Temp' group"

"List all users and their assigned groups"
```

## Admin MCP Server Requirements

The admin agent expects the MCP server to provide the following tools:

### User Management Tools

- `add-user`: Add a new user to the site
  - Parameters: `username`, `siteRole`, `authSetting`, optional `email`
- `update-user`: Update user information
  - Parameters: `userId`, optional `username`, `siteRole`, `email`, `authSetting`
- `delete-user`: Remove a user from the site
  - Parameters: `userId`
- `get-user`: Get detailed user information
  - Parameters: `userId` or `username`
- `list-users`: List users with optional filters
  - Parameters: optional `siteRole`, `authSetting`, `filter`

### Group Management Tools

- `create-group`: Create a new group
  - Parameters: `name`, optional `description`
- `update-group`: Update group information
  - Parameters: `groupId`, optional `name`, `description`
- `delete-group`: Delete a group
  - Parameters: `groupId`
- `get-group`: Get group details and members
  - Parameters: `groupId` or `groupName`
- `list-groups`: List all groups
  - Parameters: optional `filter`
- `add-user-to-group`: Add a user to a group
  - Parameters: `userId`, `groupId`
- `remove-user-from-group`: Remove a user from a group
  - Parameters: `userId`, `groupId`

## API Endpoints

### Web Interface

- `GET /` - Admin chat interface (HTML)
- `GET /health` - Health check and capabilities
- `GET /system-prompt` - Get the admin system prompt
- `GET /admin/capabilities` - List available admin operations

### Chat Endpoints

- `POST /admin-chat` - Send admin request (non-streaming)
  ```json
  {
    "message": "Add user john.doe@example.com",
    "history": []
  }
  ```

- `POST /admin-chat-stream` - Send admin request (streaming with progress)
  - Returns Server-Sent Events (SSE) stream

### Response Format

```json
{
  "response": "✓ User added successfully: john.doe@example.com with Creator role",
  "tool_results": [
    {
      "tool": "add-user",
      "arguments": {
        "username": "john.doe@example.com",
        "siteRole": "Creator",
        "authSetting": "ServerDefault"
      },
      "result": "{\"user\": {\"id\": \"123\", \"name\": \"john.doe@example.com\"}}"
    }
  ],
  "iterations": 2
}
```

## Security Best Practices

1. **Authentication**: Ensure the admin MCP server requires proper authentication
2. **Authorization**: Limit admin agent access to authorized users only
3. **Audit Logging**: Log all admin operations for compliance
4. **Validation**: The agent validates IDs before destructive operations
5. **Confirmation**: For critical operations, the agent confirms before executing

## Troubleshooting

### Connection Issues

```
Error: ADMIN_MCP_SERVER environment variable is required
```
**Solution**: Set `ADMIN_MCP_SERVER` in your `.env` file

### Tool Not Found

```
Error: No admin tools available from the MCP server
```
**Solution**: Check that the admin MCP server is running and accessible

### Invalid Parameters

```
Error: Invalid arguments for tool add-user
```
**Solution**: The agent will learn from errors and retry with correct parameters

## Development

### File Structure

```
utilities/
  admin_agent.py          # Main admin agent implementation
admin_web_app.py          # FastAPI web application
README_ADMIN_AGENT.md     # This file
.env                      # Environment variables (create from .env_template)
```

### Testing the Agent

```python
import asyncio
from utilities.admin_agent import admin_mcp_chat

async def test_admin_agent():
    result = await admin_mcp_chat("List all users")
    print(result["response"])

asyncio.run(test_admin_agent())
```

### Extending the Agent

To add new capabilities:

1. Add new tools to your admin MCP server
2. The agent will automatically discover and use them
3. Update the `ADMIN_SYSTEM_PROMPT` in `admin_agent.py` if needed

## License

Same as the parent project.

## Support

For issues or questions:
1. Check the admin MCP server logs
2. Enable LangSmith tracing for debugging
3. Review the FastAPI logs (`admin_web_app.py`)
