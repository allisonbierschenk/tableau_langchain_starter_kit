# Admin MCP Server - 404 Error on Write Operations

## Issue Description

The Admin MCP server at `https://admin-mcp.up.railway.app/tableau-mcp` has a bug where **POST operations fail with 404 errors**, while GET operations work correctly.

## What Works ✅

- **Query Users** (`get-users-on-site`) - Returns all users successfully
- **Query Groups** (`query-groups`) - Returns all groups successfully  
- **Authentication** - Server successfully authenticates and gets site LUID
- **Get Users in Group** (`get-users-in-group`) - Works correctly

## What Fails ❌

- **Add User to Site** (`add-user-to-site`) - Returns 404
- **Create Group** (`create-group`) - Returns 404
- **Update User** (`update-user`) - Returns 404
- **All other POST/PUT/DELETE operations** - Return 404

## Root Cause

The MCP server:
1. ✅ Successfully authenticates with Tableau and receives the site LUID in the auth response
2. ✅ Correctly uses the site LUID in GET request URLs (which is why reads work)
3. ❌ **Fails to use the site LUID in POST/PUT/DELETE request URLs**

The Tableau REST API requires the site LUID in the URL path:
```
GET  /api/{version}/sites/{site-luid}/users      ← Works
POST /api/{version}/sites/{site-luid}/users      ← Fails with 404
```

## Evidence

Testing shows:
- `get-users-on-site` returns 58 users (authentication and GET requests work)
- `add-user-to-site` with all correct parameters returns 404
- `add-user-to-site` with explicit `siteId` parameter also returns 404
- `create-group` returns 404

The error message from the server is: `"requestId: 1, error: Request failed with status code 404"`

## Solution

The MCP server code needs to be updated to properly use the site LUID from the authentication token when constructing POST/PUT/DELETE request URLs.

### Where to Look in the Server Code:

1. **RestApi class** - Check how it constructs URLs for different HTTP methods
2. **URL construction logic** - Ensure `siteId` or `restApi.siteId` is included in the path
3. **Operation handlers** - Verify that POST operations like `add-user-to-site` use the site LUID

Example of what might be wrong:
```typescript
// ❌ Wrong - missing site LUID
const url = `/api/${version}/users`

// ✅ Correct - includes site LUID
const url = `/api/${version}/sites/${this.siteId}/users`
```

### Testing the Fix:

After fixing the server, test with:
```bash
python3 test_admin_agent.py
```

The test should successfully add a user without returning a 404 error.

## Temporary Workarounds

### Option 1: Use Tableau REST API Directly
Until the MCP server is fixed, you can use the Tableau REST API directly via Python or curl.

### Option 2: Read-Only Operations
The admin agent can still be used for:
- Listing all users
- Querying groups  
- Finding user information
- Viewing group memberships

## Current Status

The admin agent has been updated to:
- ✅ Use correct Tableau REST API parameters (from official documentation)
- ✅ Handle conversation context properly
- ✅ Map user-friendly inputs to API values (e.g., "mfa" → "TableauIDWithMFA")
- ✅ Provide clear error messages about the write operation limitation
- ⏳ Waiting for MCP server fix to enable write operations

## Next Steps

1. **Access the MCP server source code** on Railway
2. **Locate the URL construction logic** for POST/PUT/DELETE operations
3. **Update to include site LUID** in the URL path
4. **Deploy the fix** to Railway
5. **Test** using the admin agent

## Contact

If you need help fixing the MCP server code, please share:
- The MCP server repository/source code
- How it's currently constructing Tableau API URLs
- The RestApi class implementation
