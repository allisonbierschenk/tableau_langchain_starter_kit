# Slack Bot Demo Setup

This guide sets up a custom Slack app that sends requests to this repo's admin-agent API endpoint.

Demo architecture:

`Slack -> slack_bot.py -> ADMIN_API_BASE_URL + ADMIN_API_PATH -> admin agent -> MCP -> Tableau`

## 1) Add files in this repo

These files are part of this repo implementation:

- `slack_bot.py`
- `utilities/slack_admin_api_client.py`
- `.env_template` updates
- `requirements.txt` updates

## 2) Create a Slack app

1. Open [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** -> **From scratch**
3. Choose app name and workspace

## 3) Configure bot permissions

Open **OAuth & Permissions** and add Bot Token Scopes:

- `app_mentions:read`
- `chat:write`
- `im:history`
- `im:read`
- `im:write`
- `channels:history` (for mention flows in channels)

Click **Install to Workspace** (or **Reinstall**) and copy:

- **Bot User OAuth Token** -> `SLACK_BOT_TOKEN` (`xoxb-...`)

## 4) Configure Socket Mode

1. Open **Socket Mode**
2. Enable Socket Mode
3. Click **Generate Token and Scopes**
4. Add scope `connections:write`
5. Copy generated token -> `SLACK_APP_TOKEN` (`xapp-...`)

## 5) Copy Signing Secret

1. Open **Basic Information**
2. Under **App Credentials**, copy **Signing Secret**
3. Set it as `SLACK_SIGNING_SECRET`

## 6) Set event subscriptions

1. Open **Event Subscriptions**
2. Enable events
3. Subscribe to bot events:
   - `app_mention`
   - `message.im`

With Socket Mode, you do not need a public Request URL.

## 7) Get allowed Slack user IDs

For each allowed demo user:

1. Open profile in Slack client
2. Click **More**
3. Click **Copy member ID**

Add IDs as comma-separated values in:

`SLACK_ALLOWED_USER_IDS=U012ABCDEF,U045GHIJKL`

For demo open access to everyone in the workspace, set:

`SLACK_ALLOWED_USER_IDS=*`

## 8) Configure environment variables

Create `.env` from `.env_template` and set:

```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_APP_TOKEN=xapp-...
SLACK_ALLOWED_USER_IDS=*
SLACK_MAX_HISTORY_ITEMS=20

ADMIN_API_BASE_URL=https://eacloudadminagent.up.railway.app
ADMIN_API_PATH=/slack-mcp-chat
ADMIN_API_BEARER_TOKEN=your-shared-secret
ADMIN_API_TIMEOUT_SECONDS=45
```

Notes:
- `ADMIN_API_PATH` defaults to `/slack-mcp-chat`.
- `ADMIN_API_BEARER_TOKEN` is required and must match the API service value.
- Set `SLACK_ALLOWED_USER_IDS=*` to allow all users in your workspace for demo simplicity.

## 9) Install dependencies

```bash
pip install -r requirements.txt
```

## 10) Run the bot

```bash
python slack_bot.py
```

## 11) Demo smoke tests

In Slack DM with the bot or mention in a channel:

- `list all users`
- `show all groups`
- `add user@example.com as Viewer`

Destructive operations require confirmation:

- Bot asks for `yes` / `no` before remove/delete requests.

## Troubleshooting

- `invalid_auth`: verify `SLACK_BOT_TOKEN`
- Socket mode connection errors: verify `SLACK_APP_TOKEN`
- No replies in channel: invite bot to channel and mention it
- "not authorized" responses: add your user ID to `SLACK_ALLOWED_USER_IDS`
- Admin API failures: verify `ADMIN_API_BASE_URL` and `ADMIN_API_PATH`
