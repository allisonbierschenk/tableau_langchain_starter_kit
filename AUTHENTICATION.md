# Tableau Authentication for Admin Portal

## Overview

The Admin Portal uses **Tableau REST API authentication** to secure access. Only users with Tableau Site Administrator roles can access the portal.

## How It Works

### 1. Authentication Flow

```
User enters credentials → Tableau REST API validates → Check site role → Create session
```

1. User provides Tableau username/password or Personal Access Token (PAT)
2. Backend calls Tableau REST API `/signin` endpoint
3. Tableau returns authentication token and user information including site role
4. System verifies user has admin role (`SiteAdministratorCreator` or `SiteAdministratorExplorer`)
5. If authorized, session is created and user can access the admin portal

### 2. Allowed Roles

Only these Tableau site roles can access the admin portal:
- **SiteAdministratorCreator** - Site admin with Creator license
- **SiteAdministratorExplorer** - Site admin with Explorer license

All other roles (Creator, Explorer, Viewer, etc.) will be denied access.

### 3. Session Management

- Sessions expire after **4 hours** (Tableau token lifetime)
- Users are automatically logged out when session expires
- Manual logout invalidates the Tableau token immediately

## Configuration

### Environment Variables

Required in `.env`:

```bash
# Tableau Server Configuration
TABLEAU_DOMAIN_FULL=https://prod-ca-a.online.tableau.com
TABLEAU_SITE=your-site-name
TABLEAU_API_VERSION=3.23

# Session Secret (generate a secure random key for production)
SESSION_SECRET=change-this-to-a-secure-random-key-in-production

# MCP Admin Server
ADMIN_MCP_SERVER=http://localhost:3001/mcp
```

### Generate Session Secret

For production, generate a secure session secret:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Add the result to your `.env` file as `SESSION_SECRET`.

## Authentication Methods

### Option 1: Username & Password

```
Username: your.email@company.com
Password: your-tableau-password
```

### Option 2: Personal Access Token (PAT)

Check the "Use Personal Access Token" box and enter:
```
Username: your-token-name
Password: your-token-secret
```

**Creating a PAT:**
1. Sign in to Tableau Cloud
2. Go to Settings → Personal Access Tokens
3. Click "Create new token"
4. Copy the token name and secret

## Security Benefits

### Why Tableau Auth is the Best Choice

1. **No separate credential management** - Uses existing Tableau accounts
2. **Role-based access control** - Only admins can access
3. **Audit trail** - All actions logged under user's Tableau identity
4. **Token-based** - No passwords stored in the application
5. **Automatic expiration** - Tokens expire after 4 hours
6. **SSO integration** - If Tableau uses SSO, auth follows org policies

### Additional Security Layers

Consider adding these for production:

1. **IP Allowlisting** - Restrict to corporate network IPs
2. **Rate Limiting** - Prevent brute force attacks
3. **HTTPS Only** - Enforce TLS encryption (automatic on Railway)
4. **Audit Logging** - Log all admin operations with timestamps

## API Endpoints

### `/login` (POST)
Authenticate with Tableau credentials
```json
{
  "username": "user@company.com",
  "password": "password",
  "use_pat": false
}
```

### `/logout` (POST)
Invalidate session and Tableau token

### `/auth/status` (GET)
Check current authentication status
```json
{
  "authenticated": true,
  "user_name": "user@company.com",
  "site_role": "SiteAdministratorCreator",
  "is_admin": true
}
```

### `/mcp-chat` and `/mcp-chat-stream` (POST)
Admin operations (requires authentication)

## Troubleshooting

### "Authentication failed"
- Verify credentials are correct
- Check that `TABLEAU_DOMAIN_FULL` and `TABLEAU_SITE` are set correctly
- Ensure your Tableau account is active

### "Access denied. Admin role required"
- Your Tableau role is not a Site Administrator
- Contact your Tableau admin to upgrade your role

### "Session expired"
- Sessions expire after 4 hours
- Simply log in again to continue

### Connection errors
- Check that `TABLEAU_DOMAIN_FULL` URL is correct
- Verify network connectivity to Tableau Cloud
- Check firewall settings

## Development vs Production

### Development (Localhost)
```bash
SESSION_SECRET=dev-secret-key-not-for-production
```

### Production (Railway/Cloud)
```bash
# Generate secure random key
SESSION_SECRET=<64-character-hex-string>
```

## Implementation Details

### Code Structure

- **`utilities/tableau_auth.py`** - Authentication logic
- **`web_app.py`** - FastAPI endpoints with auth middleware
- **`static/index.html`** - Login UI
- **`static/script.js`** - Client-side auth handling

### Session Storage

Sessions are stored server-side using FastAPI's `SessionMiddleware` with encrypted cookies.

### Token Refresh

Tokens are not automatically refreshed. Users must re-authenticate after 4 hours.

## Best Practices

1. **Never commit `.env` to git** - Use `.env.example` for templates
2. **Rotate session secrets regularly** - Every 90 days minimum
3. **Monitor failed login attempts** - Alert on unusual activity
4. **Use PATs for automation** - Username/password for interactive use
5. **Document admin access** - Keep records of who has access

## Questions?

- Review Tableau REST API docs: https://help.tableau.com/current/api/rest_api/en-us/REST/rest_api_ref_authentication.htm
- Check application logs for detailed error messages
- Verify environment variables are set correctly
