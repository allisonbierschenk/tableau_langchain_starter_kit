"""Cross-MCP governance text (one place; not workflow-specific)."""

CROSS_MCP_GOVERNANCE = """
**Dual MCP (Tableau + Slack):**
- **Tableau MCP** tools use their **native names** from the server (e.g. `admin-users`, `list-workbooks`)—same as single-MCP mode.
- **Slack MCP** tools use the `slack_mcp__` prefix so they route to the Slack server.
- Prefer Tableau tools for site data, admin, permissions, jobs, and operations; use Slack tools for workspace context and messaging after you have facts from Tableau.
- Never invent metrics, owner emails, or Slack channel/user ids—only values from tool results or explicit user text.
- For **Tableau Pulse** subscription answers, never leave a user-facing bullet as UUID-only when the metric display name exists in tool JSON; resolve or call follow-up tools for titles. If subscription JSON is id-only, use site-wide Pulse definition/metric list or batch-get tools (per [Pulse REST](https://help.tableau.com/current/api/rest_api/en-us/REST/rest_api_ref_pulse.htm)) before answering.
- If a Slack step fails (auth, missing user), report the error and continue transparently.
"""

SLACK_MCP_AUTH_README_SNIPPET = """
Server-side Slack MCP (`https://mcp.slack.com/mcp` or your proxy) requires credentials the HTTP client can send on each request.
Cursor obtains OAuth interactively; for this API use one of:
- `SLACK_MCP_BEARER_TOKEN` — `Authorization: Bearer <token>` (token from your approved OAuth / proxy flow), or
- `SLACK_MCP_AUTH_HEADER` + `SLACK_MCP_AUTH_VALUE` — custom header for a gateway.

Without a valid token, `tools/list` / `tools/call` to Slack MCP will fail at runtime; Tableau MCP still works if only `ADMIN_MCP_SERVER` is configured.
"""
