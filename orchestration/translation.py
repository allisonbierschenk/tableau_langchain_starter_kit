"""TranslationLayer: cross-system resolution (MCP-backed, registered keys only)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from utilities.identity_mapping import SlackMCPIdentityMapper


class TranslationLayer(Protocol):
    async def resolve(self, key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class McpTranslationLayer:
    """
    Registered translations only. Currently: email_to_slack_user_id via Slack MCP.
    """

    def __init__(self, slack_mapper: SlackMCPIdentityMapper):
        self._slack_mapper = slack_mapper

    async def resolve(self, key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if key == "email_to_slack_user_id":
            email = str(payload.get("email") or "").strip()
            if not email:
                return {"ok": False, "error": "missing_email"}
            r = await self._slack_mapper.resolve_slack_user_id(email)
            return {
                "ok": r.ok,
                "email": r.email,
                "slack_user_id": r.slack_user_id,
                "error": r.error,
            }
        raise ValueError(f"Unknown translation key: {key!r}")
