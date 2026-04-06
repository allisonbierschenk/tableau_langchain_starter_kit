# Slack bot (Socket Mode)

Flow: **Slack → `slack_bot.py` → HTTP API (`web_app.py`) → MCP → Tableau.**

Responses from `/slack-mcp-chat` are converted from LLM-style `**bold**` to Slack mrkdwn `*bold*` so names render correctly in Slack.

## Railway (recommended)

Use **one Git branch**, **two services** from the same repo:

| Service | Start command |
|--------|----------------|
| API / web | `python web_app.py` |
| Slack bot | `python slack_bot.py` |

Build: leave default (Nixpacks). No custom build command unless you need one.

## Environment variables

**Slack bot service**

- `SLACK_BOT_TOKEN` — OAuth & Permissions → Bot User OAuth Token (`xoxb-…`)
- `SLACK_SIGNING_SECRET` — Basic Information → App Credentials
- `SLACK_APP_TOKEN` — Basic Information → App-Level Tokens, scope `connections:write` (`xapp-…`)
- `SLACK_ALLOWED_USER_IDS` — `*` for everyone, or comma-separated Slack user IDs
- `ADMIN_API_BASE_URL` — public URL of the **API** service (no trailing slash)
- `ADMIN_API_PATH` — `/slack-mcp-chat` (or `/mcp-chat` if your API exposes it without bearer auth)
- `ADMIN_API_BEARER_TOKEN` — shared secret; must match the API service if using `/slack-mcp-chat`
- `ADMIN_API_TIMEOUT_SECONDS` — optional (default 45)

**API service**

- Same vars as admin portal: `OPENAI_API_KEY`, `ADMIN_MCP_SERVER`, Tableau settings, etc.
- `ADMIN_API_BEARER_TOKEN` — same value as bot service (for `/slack-mcp-chat`)

## Slack app configuration

- **Bot token scopes:** `app_mentions:read`, `chat:write`, `im:history`, `im:read`, `im:write`, `channels:history`
- **Event Subscriptions:** `app_mention`, `message.im`
- **Socket Mode:** enabled
- Install (or reinstall) the app to the workspace after scope changes

## Run locally

```bash
pip install -r requirements.txt
python slack_bot.py
```

Test in a DM or with `@YourBot list all users`.
