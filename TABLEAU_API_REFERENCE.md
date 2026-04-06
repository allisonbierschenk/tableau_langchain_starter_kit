# Tableau REST API - Users and Groups Reference

**Documentation Source:** https://help.tableau.com/current/api/rest_api/en-us/REST/rest_api_ref_users_and_groups.htm

**API Version:** 3.28 (current as of extraction)

---

## 1. Add User to Site

**Endpoint:** `POST /api/{api-version}/sites/{site-id}/users`

### Request Body (XML)
```xml
<tsRequest>
  <user name="user-name"
        siteRole="site-role"
        authSetting="auth-setting"
        identityPoolName="identity-pool-name"
        idpConfigurationId="idp-configuration-id"
        email="email-address" />
</tsRequest>
```

### Request Body (JSON)
```json
{
  "user": {
    "name": "user-name",
    "siteRole": "site-role",
    "authSetting": "auth-setting",
    "identityPoolName": "identity-pool-name",
    "idpConfigurationId": "idp-configuration-id",
    "email": "email-address"
  }
}
```

### Required Parameters
- **name** (string): The user name
  - **Tableau Server with local auth**: Can be any name
  - **Tableau Server with Active Directory**: Must be existing AD user (e.g., `example\Adam` or `adam@example.com`)
  - **Tableau Cloud**: Must be the email address for sign-in
  
- **siteRole** (string): The site role to assign

### Optional Parameters
- **authSetting** (string): Authentication method (see valid values below)
- **identityPoolName** (string): For Tableau Server with identity pools (API 3.19+)
- **idpConfigurationId** (string): For Tableau Cloud authentication config (API 3.25+)
- **email** (string): Email address for Tableau Cloud (API 3.26+)

### Valid Values for siteRole
- `Creator`
- `Explorer`
- `ExplorerCanPublish`
- `SiteAdministratorExplorer`
- `SiteAdministratorCreator`
- `Unlicensed`
- `Viewer`

**Note:** Cannot set `ServerAdministrator` via REST API

### Valid Values for authSetting
- **SAML**: User signs in via IdP configured for the site
- **OpenID**: (Tableau Cloud only) Google, Salesforce, or OIDC credentials
- **TableauIDWithMFA**: (Tableau Cloud only) TableauID with multi-factor authentication
- **ServerDefault**: Default authentication method for the server

**Important Notes:**
- For Tableau Cloud (API 3.25+): Use `idpConfigurationId` OR `authSetting`, not both
- If neither specified, defaults to `TableauIDWithMFA` (or `ServerDefault` if available)
- For new SAML/OIDC configs created after January 2025, must use `idpConfigurationId`

### Response
**Status Code:** 201

```xml
<tsResponse>
  <user id="user-id"
        name="user-name"
        siteRole="site-role"
        authSetting="auth-setting"
        identityPoolName="identity-pool-name"
        idpConfigurationId="idp-configuration-id"
        email="email-address" />
</tsResponse>
```

### Permissions
- Server administrators
- Site administrators

### Scope for JWT
`tableau:users:create`

---

## 2. Update User

**Endpoint:** `PUT /api/{api-version}/sites/{site-id}/users/{user-id}`

### Request Body (XML)
```xml
<tsRequest>
  <user fullName="new-full-name"
        email="new-email"
        password="new-password"
        siteRole="new-site-role"
        authSetting="new-auth-setting"
        identityPoolName="identity-pool-name"
        idpConfigurationId="idp-configuration-id" />
</tsRequest>
```

### Request Body (JSON)
```json
{
  "user": {
    "fullName": "new-full-name",
    "email": "new-email",
    "password": "new-password",
    "siteRole": "new-site-role",
    "authSetting": "new-auth-setting",
    "identityPoolName": "identity-pool-name",
    "idpConfigurationId": "idp-configuration-id"
  }
}
```

### All Parameters are Optional
Only include the fields you want to update.

- **fullName** (string): New full name (Tableau Server only, not Cloud)
- **email** (string): New email address
  - Tableau Server: User's email
  - Tableau Cloud (API 3.26+): For notifications only
- **password** (string): New password (Tableau Server only, not Cloud)
- **siteRole** (string): New site role (same values as Add User)
- **authSetting** (string): New authentication method (same values as Add User)
- **identityPoolName** (string): Identity pool name (Tableau Server only)
- **idpConfigurationId** (string): Authentication config ID (Tableau Cloud only, API 3.25+)

### Valid Values for siteRole
- `Creator`
- `Explorer`
- `ExplorerCanPublish`
- `ServerAdministrator` (API 2.8+, only by ServerAdministrator users)
- `SiteAdministratorExplorer`
- `SiteAdministratorCreator`
- `Unlicensed`
- `Viewer`

### Valid Values for authSetting
Same as Add User: `SAML`, `OpenID`, `TableauIDWithMFA`, `ServerDefault`

### Response
**Status Code:** 200

```xml
<tsResponse>
  <user name="user-name"
        fullName="new-full-name"
        email="new-email"
        siteRole="new-site-role"
        authSetting="new-auth-setting"
        identityPoolName="identity-pool-name"
        idpConfigurationId="idp-configuration-id" />
</tsResponse>
```

### Permissions
- Server administrators
- Site administrators (with limitations)

### Scope for JWT
`tableau:users:update` (API 3.27+)

---

## 3. Remove User from Site

**Endpoint:** `DELETE /api/{api-version}/sites/{site-id}/users/{user-id}`

**With asset mapping:** `DELETE /api/{api-version}/sites/{site-id}/users/{user-id}?mapAssetsTo={new-user-id}`

### Parameters
- **user-id** (required): ID of user to remove
- **mapAssetsTo** (optional): User ID to receive ownership of content
  - Only server administrators can use this parameter
  - If user owns only subscriptions, can delete without this parameter

### Request Body
None

### Response
**Status Code:** 204 (No Content)

### Important Notes
- User is deleted if they don't own assets other than subscriptions
- User cannot be deleted if they own content on any site
- If removed from all sites, user is deleted from server
- Only server administrators can reassign content ownership

### Permissions
- Server administrators
- Site administrators (cannot use mapAssetsTo)

### Scope for JWT
`tableau:users:delete`

---

## 4. Query Users on Site (Get Users on Site)

**Endpoint:** `GET /api/{api-version}/sites/{site-id}/users`

### Query Parameters
- **filter**: Filter expression (e.g., `siteRole:eq:Unlicensed`)
- **sort**: Sort expression (e.g., `name:asc`)
- **pageSize**: Number of items per page (1-1000, default 100)
- **pageNumber**: Page number (default 1)
- **fields**: Field expression (e.g., `_all_`, `_default_`)

### Example with Filters
```
GET /api/3.28/sites/{site-id}/users?filter=siteRole:eq:Unlicensed&sort=name:asc
```

### Request Body
None

### Response
**Status Code:** 200

```xml
<tsResponse>
  <pagination pageNumber="page-number" pageSize="page-size" totalAvailable="total-available" />
  <users>
    <user email="email"
          id="user-id"
          fullName="full-name"
          name="user-name"
          siteRole="site-role"
          lastLogin="date-time"
          externalAuthUserId="authentication-id-from-external-provider"
          authSetting="auth-setting"
          idpConfigurationId="idp-configuration-id"
          language="language-code"
          locale="locale-code">
      <domain name="domain-name" />
    </user>
    <!-- More users... -->
  </users>
</tsResponse>
```

### Response siteRole Values
- `Creator`
- `Explorer`
- `ExplorerCanPublish`
- `ServerAdministrator`
- `SiteAdministratorExplorer`
- `SiteAdministratorCreator`
- `Unlicensed`
- `ReadOnly`
- `Viewer`

### Permissions
- Server administrators
- Site administrators

### Scope for JWT
`tableau:users:read`

---

## 5. Create Group

**Endpoint:** `POST /api/{api-version}/sites/{site-id}/groups`

**With background job:** `POST /api/{api-version}/sites/{site-id}/groups?asJob={true|false}`

### Request Body - Local Group (XML)
```xml
<tsRequest>
  <group name="new-tableau-server-group-name"
         minimumSiteRole="minimum-site-role"
         ephemeralUsersEnabled="on-demand-access" />
</tsRequest>
```

### Request Body - Active Directory Group (XML, Tableau Server only)
```xml
<tsRequest>
  <group name="active-directory-group-name">
    <import source="ActiveDirectory"
            domainName="active-directory-domain-name"
            grantLicenseMode="license-mode"
            siteRole="minimum-site-role" />
  </group>
</tsRequest>
```

### Parameters
- **name** (required): Group name
- **minimumSiteRole** (optional for local, required with import): Minimum site role for group members
- **ephemeralUsersEnabled** (optional): Boolean for on-demand access (Tableau Cloud only, API 3.21+)
- **source** (for import): Must be `ActiveDirectory`
- **domainName** (for import): Active Directory domain name
- **grantLicenseMode** (for import): `onSync` or `onLogin`
- **siteRole** (for import): Minimum site role

### Valid Values for minimumSiteRole/siteRole
- `Unlicensed`
- `Viewer`
- `Explorer`
- `ExplorerCanPublish`
- `Creator`
- `SiteAdministratorExplorer`
- `SiteAdministratorCreator`

### Valid Values for grantLicenseMode
- **Local groups**: `onLogin` or unset (default: unset)
- **AD groups**: `onSync` or `onLogin`

### Response
**Status Code:** 201 (immediate), 202 (background job)

```xml
<tsResponse>
  <group id="new-group-id"
         name="group-name"
         minimumSiteRole="minimum-site-role"
         ephemeralUsersEnabled="true">
    <import domainName="domain-name"
            siteRole="default-site-role"
            grantLicenseMode="license-mode" />
  </group>
</tsResponse>
```

### Permissions
- Server administrators
- Site administrators

### Scope for JWT
`tableau:groups:create`

---

## 6. Add User to Group

**Endpoint:** `POST /api/{api-version}/sites/{site-id}/groups/{group-id}/users`

### Request Body - Single User (XML)
```xml
<tsRequest>
  <user id="user-id" />
</tsRequest>
```

### Request Body - Bulk Add (XML, API 3.21+)
```xml
<tsRequest>
  <users>
    <user id="user-id1" />
    <user id="user-id2" />
    <user id="user-id3" />
  </users>
</tsRequest>
```

### Parameters
- **user-id** (required): User ID to add (get from Query Users on Site)

### Response
**Status Code:** 200

```xml
<tsResponse>
  <user id="user-id"
        name="user-name"
        siteRole="site-role" />
</tsResponse>
```

Or for bulk:
```xml
<tsResponse>
  <users>
    <user id="user-id" name="user-name" siteRole="site-role" />
    <user id="user-id" name="user-name" siteRole="site-role" />
  </users>
</tsResponse>
```

### Permissions
- Server administrators
- Site administrators

### Scope for JWT
`tableau:groups:update`

---

## 7. Remove User from Group

**Endpoint (Single):** `DELETE /api/{api-version}/sites/{site-id}/groups/{group-id}/users/{user-id}`

**Endpoint (Bulk, API 3.21+):** `PUT /api/{api-version}/sites/{site-id}/groups/{group-id}/users/remove`

### Request Body - Single User
None

### Request Body - Bulk Remove (XML)
```xml
<tsRequest>
  <users>
    <user id="user-id1" />
    <user id="user-id2" />
    <user id="user-id3" />
  </users>
</tsRequest>
```

### Parameters
- **user-id** (required for single remove): User ID to remove

### Response
**Status Code:** 204 (No Content)

### Permissions
- Server administrators
- Site administrators

### Scope for JWT
`tableau:groups:update`

---

## Key Notes for Admin Agent Configuration

1. **Authentication Settings Priority (Tableau Cloud):**
   - For new configs (Jan 2025+): Use `idpConfigurationId`
   - For "Initial" prefixed configs: Can use `authSetting` or `idpConfigurationId`
   - Never use both attributes together

2. **Site Role Hierarchy (least to most permissions):**
   - Unlicensed → Viewer → Explorer → ExplorerCanPublish → Creator → SiteAdministratorExplorer → SiteAdministratorCreator

3. **User Deletion:**
   - Must not own content (except subscriptions)
   - Can map assets to another user (server admins only)
   - Removed from all sites = deleted from server

4. **Bulk Operations (API 3.21+):**
   - Add multiple users to group
   - Remove multiple users from group

5. **Filtering & Pagination:**
   - Use `filter` parameter for queries (e.g., `siteRole:eq:Unlicensed`)
   - Page size: 1-1000 (default 100)
   - Can combine filter, sort, and pagination

6. **Permissions Required:**
   - Most operations: Server or Site administrators
   - Content reassignment: Server administrators only
   - ServerAdministrator role changes: ServerAdministrator users only
