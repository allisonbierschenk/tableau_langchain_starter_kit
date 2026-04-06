# MCP Server Fix Instructions - 404 Error on Write Operations

## Problem Summary

The Admin MCP server at `https://admin-mcp.up.railway.app/tableau-mcp` returns **404 errors** for all POST/PUT/DELETE operations (add-user-to-site, create-group, update-user, etc.), while GET operations (get-users-on-site, query-groups) work correctly.

**Root Cause:** The server is not properly using the site LUID in the URL path for write operations.

---

## Diagnosis Confirmation

### Test Results:
- ✅ Authentication works - server receives site LUID from Tableau auth response
- ✅ `get-users-on-site` returns data successfully (58 users)
- ✅ `query-groups` returns data successfully
- ❌ `add-user-to-site` returns 404 (with or without explicit siteId parameter)
- ❌ `create-group` returns 404
- ❌ All POST/PUT/DELETE operations return 404

### Error Message:
```
requestId: 1, error: Request failed with status code 404
```

---

## The Fix

### Step 1: Locate the URL Construction Logic

Find where your MCP server constructs Tableau REST API URLs. This is typically in:
- A `RestApi` class or similar
- An HTTP client wrapper
- Tool handler functions for each operation

Look for code that builds URLs like:
```typescript
const url = `${this.baseUrl}/api/${this.apiVersion}/sites/${siteId}/users`
```

### Step 2: Identify the Problem

Compare how GET operations vs POST operations construct URLs:

**GET Operations (WORKING):**
```typescript
// Example: get-users-on-site
const url = `/api/${apiVersion}/sites/${this.siteId}/users`
// Returns: /api/3.28/sites/abc123-def456-789/users
```

**POST Operations (BROKEN):**
```typescript
// Example: add-user-to-site
const url = `/api/${apiVersion}/sites/${siteId}/users`  // ← siteId might be undefined or wrong
// OR
const url = `/api/${apiVersion}/users`  // ← Missing /sites/{site-luid}/ entirely
```

### Step 3: Ensure Site LUID is Available

After authentication, the server should store the site LUID. Verify this exists:

```typescript
// After auth/signin response:
const authResponse = await tableau.auth.signin(...)

// The response includes:
authResponse.credentials.site.id  // ← This is the site LUID (e.g., "abc123-def456-789")

// Store it:
this.siteId = authResponse.credentials.site.id
```

Also check if you have a helper function like:
```typescript
getSiteLuidFromAccessToken(token: string): string {
  // Extracts site LUID from JWT token
}
```

### Step 4: Fix the POST/PUT/DELETE URL Construction

Update your tool handlers to use the site LUID:

#### Example Fix for `add-user-to-site`:

**Before (Broken):**
```typescript
async handleAddUserToSite(args: any) {
  const { siteId, body } = args
  
  // ❌ Problem: Using siteId parameter which might not be the LUID
  const url = `/api/${this.apiVersion}/sites/${siteId || 'default'}/users`
  
  return this.httpClient.post(url, body)
}
```

**After (Fixed):**
```typescript
async handleAddUserToSite(args: any) {
  const { body } = args
  
  // ✅ Use the site LUID from auth token
  const siteLuid = this.restApi.siteId || this.getSiteLuidFromAccessToken(this.accessToken)
  const url = `/api/${this.apiVersion}/sites/${siteLuid}/users`
  
  return this.httpClient.post(url, body)
}
```

#### Example Fix for `create-group`:

**Before (Broken):**
```typescript
async handleCreateGroup(args: any) {
  const { siteId, body } = args
  const url = `/api/${this.apiVersion}/groups`  // ❌ Missing /sites/{site-luid}/
  
  return this.httpClient.post(url, body)
}
```

**After (Fixed):**
```typescript
async handleCreateGroup(args: any) {
  const { body } = args
  
  // ✅ Include site LUID in path
  const siteLuid = this.restApi.siteId || this.getSiteLuidFromAccessToken(this.accessToken)
  const url = `/api/${this.apiVersion}/sites/${siteLuid}/groups`
  
  return this.httpClient.post(url, body)
}
```

### Step 5: Apply the Pattern to All Write Operations

Fix these operations with the same pattern:
- `add-user-to-site` → POST `/api/{version}/sites/{site-luid}/users`
- `update-user` → PUT `/api/{version}/sites/{site-luid}/users/{user-id}`
- `remove-user-from-site` → DELETE `/api/{version}/sites/{site-luid}/users/{user-id}`
- `create-group` → POST `/api/{version}/sites/{site-luid}/groups`
- `update-group` → PUT `/api/{version}/sites/{site-luid}/groups/{group-id}`
- `delete-group` → DELETE `/api/{version}/sites/{site-luid}/groups/{group-id}`
- `add-user-to-group` → POST `/api/{version}/sites/{site-luid}/groups/{group-id}/users`
- `remove-user-from-group` → DELETE `/api/{version}/sites/{site-luid}/groups/{group-id}/users/{user-id}`

### Step 6: Optional - Create a Helper Function

Consider creating a URL builder helper:

```typescript
private buildApiUrl(endpoint: string, pathParams?: Record<string, string>): string {
  const siteLuid = this.restApi.siteId || this.getSiteLuidFromAccessToken(this.accessToken)
  
  if (!siteLuid) {
    throw new Error('Site LUID not available. Authentication may have failed.')
  }
  
  // Build base URL with site context
  let url = `/api/${this.apiVersion}/sites/${siteLuid}${endpoint}`
  
  // Replace path parameters if provided
  if (pathParams) {
    for (const [key, value] of Object.entries(pathParams)) {
      url = url.replace(`{${key}}`, value)
    }
  }
  
  return url
}

// Usage:
const url = this.buildApiUrl('/users')  // → /api/3.28/sites/{site-luid}/users
const url = this.buildApiUrl('/groups/{group-id}/users', { 'group-id': groupId })
```

---

## Testing the Fix

### Test 1: Basic Connection Test
```bash
# Should work (already working):
curl -X POST https://admin-mcp.up.railway.app/tableau-mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "admin-users",
      "arguments": {
        "operation": "get-users-on-site",
        "pageSize": 1
      }
    },
    "id": 1
  }'
```

### Test 2: Add User (After Fix)
```bash
# Should work after fix:
curl -X POST https://admin-mcp.up.railway.app/tableau-mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "admin-users",
      "arguments": {
        "operation": "add-user-to-site",
        "body": {
          "user": {
            "name": "testuser@example.com",
            "siteRole": "Viewer",
            "authSetting": "TableauIDWithMFA"
          }
        }
      }
    },
    "id": 1
  }'
```

**Expected Response (Success):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"user\":{\"id\":\"...\",\"name\":\"testuser@example.com\",\"siteRole\":\"Viewer\",\"authSetting\":\"TableauIDWithMFA\"}}"
      }
    ]
  },
  "id": 1
}
```

**Current Response (Error - should be fixed):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "requestId: 1, error: Request failed with status code 404"
      }
    ],
    "isError": true
  },
  "id": 1
}
```

### Test 3: Create Group (After Fix)
```bash
curl -X POST https://admin-mcp.up.railway.app/tableau-mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "admin-groups",
      "arguments": {
        "operation": "create-group",
        "body": {
          "group": {
            "name": "Test Group"
          }
        }
      }
    },
    "id": 1
  }'
```

### Test 4: Use the Admin Agent Test Suite
```bash
cd /Users/abierschenk/Desktop/TableauRepos/tableau_langchain_starter_kit
python3 test_admin_agent.py
```

This will test:
- Listing users (should work before and after)
- Adding a user (should work after fix)
- Listing groups (should work before and after)

---

## Validation Checklist

After deploying the fix, verify:

- [ ] `get-users-on-site` still works (didn't break reads)
- [ ] `add-user-to-site` returns 201 Created (not 404)
- [ ] `create-group` returns 201 Created (not 404)
- [ ] `update-user` returns 200 OK (not 404)
- [ ] `remove-user-from-site` returns 204 No Content (not 404)
- [ ] Error messages include proper details (not just "404")

---

## Common Pitfalls to Avoid

### 1. Using Site Name Instead of Site LUID
❌ **Wrong:** `siteId = "eacanada"` (site name)  
✅ **Correct:** `siteId = "abc123-def456-789ghi"` (site LUID from auth response)

### 2. Using the `siteId` Parameter from Tool Arguments
The `siteId` parameter in tool arguments is optional and may not be the LUID. Always use the site LUID from the auth token:

```typescript
// ❌ Don't do this:
const siteLuid = args.siteId  // Might be site name or undefined

// ✅ Do this:
const siteLuid = this.restApi.siteId  // From auth response
```

### 3. Inconsistent URL Construction
Ensure ALL write operations use the same pattern. Don't have some operations using site LUID and others not.

### 4. Missing Site Context in URL
❌ **Wrong:** `/api/3.28/users`  
✅ **Correct:** `/api/3.28/sites/{site-luid}/users`

---

## Expected URL Patterns (Reference)

All Tableau Cloud/Server REST API endpoints follow this pattern:

```
GET    /api/{version}/sites/{site-luid}/users
POST   /api/{version}/sites/{site-luid}/users
PUT    /api/{version}/sites/{site-luid}/users/{user-id}
DELETE /api/{version}/sites/{site-luid}/users/{user-id}

GET    /api/{version}/sites/{site-luid}/groups
POST   /api/{version}/sites/{site-luid}/groups
PUT    /api/{version}/sites/{site-luid}/groups/{group-id}
DELETE /api/{version}/sites/{site-luid}/groups/{group-id}

POST   /api/{version}/sites/{site-luid}/groups/{group-id}/users
DELETE /api/{version}/sites/{site-luid}/groups/{group-id}/users/{user-id}
```

**Note:** The `/sites/{site-luid}/` part is REQUIRED for all operations.

---

## Deployment

After making the fix:

1. **Commit the changes:**
   ```bash
   git add .
   git commit -m "Fix: Include site LUID in POST/PUT/DELETE request URLs"
   ```

2. **Deploy to Railway:**
   ```bash
   git push
   # Railway should auto-deploy
   ```

3. **Verify deployment:**
   - Check Railway logs for successful deployment
   - Run the test suite
   - Verify no errors in Railway logs

---

## Additional Resources

- **Tableau REST API Documentation:** https://help.tableau.com/current/api/rest_api/en-us/REST/rest_api_concepts_auth.htm
- **Admin Agent Test Suite:** `/test_admin_agent.py`
- **Bug Documentation:** `/ADMIN_MCP_SERVER_BUG.md`
- **Tableau API Reference:** `/TABLEAU_API_REFERENCE.md`

---

## Support

If you encounter issues after applying this fix:

1. Check Railway logs for detailed error messages
2. Verify the site LUID is being extracted correctly from auth response
3. Use curl to test endpoints directly (see Test section above)
4. Ensure Tableau credentials in Railway environment variables are correct

Questions? Check the conversation history or create a new issue.
