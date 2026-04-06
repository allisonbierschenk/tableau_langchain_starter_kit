# MCP Server Issue - Quick Summary

**Server:** `https://admin-mcp.up.railway.app/tableau-mcp`

## The Problem
All write operations (POST/PUT/DELETE) return **404 errors**.  
Read operations (GET) work perfectly.

## Why It's Happening
The server isn't including the **site LUID** in the URL path for POST/PUT/DELETE requests.

## Examples

### What Works ✅
```
GET /api/3.28/sites/{site-luid}/users  → Returns 58 users
```

### What Fails ❌
```
POST /api/3.28/sites/{site-luid}/users → 404 error
```

The server has the site LUID (from auth), but isn't using it in write operation URLs.

## The Fix
Ensure all POST/PUT/DELETE operations construct URLs like:
```typescript
const siteLuid = this.restApi.siteId  // From auth response
const url = `/api/${apiVersion}/sites/${siteLuid}/users`
```

Not:
```typescript
const url = `/api/${apiVersion}/users`  // ❌ Missing site context
```

## Operations That Need Fixing
- `add-user-to-site`
- `update-user`
- `remove-user-from-site`
- `create-group`
- `update-group`
- `delete-group`
- `add-user-to-group`
- `remove-user-from-group`

## Full Instructions
See `MCP_SERVER_FIX_INSTRUCTIONS.md` for detailed step-by-step fix instructions.

## Testing
After fixing, test with:
```bash
python3 test_admin_agent.py
```

## Questions?
Contact: Allison Bierschenk
