"""
Email -> Slack user id via Slack MCP lookup tools only (decoupled for any admin workflow).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from pydantic import BaseModel, Field


class MCPCaller(Protocol):
    async def list_tools(self) -> List[Dict[str, Any]]: ...

    async def call_tool(self, tool_name: str, arguments: dict) -> Any: ...


class IdentityMappingResult(BaseModel):
    email: str
    slack_user_id: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None
    lookup_tool_name: Optional[str] = None
    raw: Optional[Any] = Field(default=None)


_EMAIL_PARAM_CANDIDATES: Tuple[str, ...] = (
    "email",
    "user_email",
    "email_address",
    "address",
    "userEmail",
)


def _normalize_mcp_tool_result(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict) and "content" in raw:
        texts: List[str] = []
        for block in raw.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if t:
                    texts.append(str(t))
        joined = "\n".join(texts).strip()
        if not joined:
            return raw
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined
    return raw


def slack_user_id_from_slack_api_shape(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    if obj.get("ok") is False:
        return None
    user = obj.get("user")
    if isinstance(user, dict):
        uid = user.get("id")
        if uid:
            return str(uid)
    if isinstance(obj.get("id"), str) and str(obj["id"]).startswith(("U", "W")):
        return str(obj["id"])
    return None


def _schema_properties(tool: Dict[str, Any]) -> Dict[str, Any]:
    schema = tool.get("inputSchema") or {}
    return schema.get("properties") or {}


def _pick_email_argument_key(tool: Dict[str, Any], forced: Optional[str] = None) -> str:
    if forced:
        return forced
    props = _schema_properties(tool)
    if not props:
        return "email"
    lower_map = {k.lower(): k for k in props}
    for cand in _EMAIL_PARAM_CANDIDATES:
        if cand in props:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for k in props:
        if "email" in k.lower():
            return k
    required = (tool.get("inputSchema") or {}).get("required") or []
    if len(required) == 1:
        return str(required[0])
    return "email"


def _score_lookup_tool(tool: Dict[str, Any]) -> float:
    name = (tool.get("name") or "").lower()
    desc = (tool.get("description") or "").lower()
    blob = f"{name} {desc}"
    score = 0.0
    if re.search(r"lookup", blob):
        score += 2.0
    if "email" in blob:
        score += 3.0
    if "user" in blob and ("slack" in blob or "member" in blob or "workspace" in blob):
        score += 1.0
    props = _schema_properties(tool)
    if any("email" in k.lower() for k in props):
        score += 2.5
    return score


@dataclass
class _ResolvedLookupTool:
    tool_name: str
    email_argument: str


class SlackMCPIdentityMapper:
    """Resolve email to Slack user id using Slack MCP tools only."""

    def __init__(
        self,
        mcp: MCPCaller,
        *,
        lookup_tool_name: Optional[str] = None,
        email_argument: Optional[str] = None,
    ):
        self._mcp = mcp
        self._lookup_tool_name = lookup_tool_name
        self._email_argument = email_argument
        self._resolved: Optional[_ResolvedLookupTool] = None

    @classmethod
    def from_env_defaults(cls, mcp: MCPCaller) -> "SlackMCPIdentityMapper":
        tool = (os.getenv("SLACK_MCP_EMAIL_LOOKUP_TOOL") or "").strip() or None
        arg = (os.getenv("SLACK_MCP_EMAIL_ARGUMENT") or "").strip() or None
        return cls(mcp, lookup_tool_name=tool, email_argument=arg)

    async def _resolve_lookup_tool(self, tools: Sequence[Dict[str, Any]]) -> _ResolvedLookupTool:
        if self._resolved:
            return self._resolved

        if self._lookup_tool_name:
            match = next((t for t in tools if t.get("name") == self._lookup_tool_name), None)
            if not match:
                raise ValueError(
                    f"Configured SLACK_MCP_EMAIL_LOOKUP_TOOL={self._lookup_tool_name!r} not in tools/list"
                )
            email_arg = _pick_email_argument_key(match, self._email_argument)
            self._resolved = _ResolvedLookupTool(self._lookup_tool_name, email_arg)
            return self._resolved

        if not tools:
            raise ValueError("Slack MCP returned no tools from tools/list")

        ranked = sorted(tools, key=_score_lookup_tool, reverse=True)
        best = ranked[0]
        if _score_lookup_tool(best) < 2.0:
            raise ValueError(
                "Could not pick a Slack email lookup tool; set SLACK_MCP_EMAIL_LOOKUP_TOOL "
                "and optionally SLACK_MCP_EMAIL_ARGUMENT."
            )
        name = str(best.get("name") or "")
        email_arg = _pick_email_argument_key(best, self._email_argument)
        self._resolved = _ResolvedLookupTool(name, email_arg)
        return self._resolved

    async def resolve_slack_user_id(self, email: str) -> IdentityMappingResult:
        cleaned = (email or "").strip()
        if not cleaned:
            return IdentityMappingResult(email=email, ok=False, error="empty_email")

        try:
            tools = await self._mcp.list_tools()
            resolved = await self._resolve_lookup_tool(tools)
            tool_name = resolved.tool_name
            arg_key = resolved.email_argument

            tool_def = next((t for t in tools if t.get("name") == tool_name), {})
            args = self._build_arguments(tool_def, arg_key, cleaned)

            raw = await self._mcp.call_tool(tool_name, args)
            normalized = _normalize_mcp_tool_result(raw)
            uid = slack_user_id_from_slack_api_shape(normalized)

            if uid:
                return IdentityMappingResult(
                    email=cleaned,
                    slack_user_id=uid,
                    ok=True,
                    lookup_tool_name=tool_name,
                    raw=normalized,
                )

            err = None
            if isinstance(normalized, dict) and normalized.get("ok") is False:
                err = str(normalized.get("error") or "slack_lookup_failed")
            return IdentityMappingResult(
                email=cleaned,
                ok=False,
                error=err or "slack_user_id_not_found_in_mcp_response",
                lookup_tool_name=tool_name,
                raw=normalized,
            )
        except Exception as e:
            return IdentityMappingResult(email=cleaned, ok=False, error=str(e))

    def _build_arguments(
        self,
        tool_def: Dict[str, Any],
        email_key: str,
        email_value: str,
    ) -> Dict[str, Any]:
        props = _schema_properties(tool_def)
        required = (tool_def.get("inputSchema") or {}).get("required") or []
        args: Dict[str, Any] = {email_key: email_value}
        for key in required:
            if key in args:
                continue
            spec = props.get(key) or {}
            if spec.get("type") == "string" and "default" in spec:
                args[key] = spec["default"]
            elif spec.get("type") == "boolean":
                args[key] = False
            elif spec.get("type") in ("integer", "number"):
                args[key] = 0
            else:
                raise ValueError(
                    f"Slack MCP tool requires {key!r} without a generic default; "
                    "set SLACK_MCP_EMAIL_LOOKUP_TOOL to a simpler tool or extend mapping."
                )
        return args


def slack_mcp_extra_headers_from_env() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    bearer = (os.getenv("SLACK_MCP_BEARER_TOKEN") or "").strip()
    if bearer:
        token = bearer if bearer.lower().startswith("bearer ") else f"Bearer {bearer}"
        headers["Authorization"] = token
        return headers
    h_name = (os.getenv("SLACK_MCP_AUTH_HEADER") or "").strip()
    h_val = (os.getenv("SLACK_MCP_AUTH_VALUE") or "").strip()
    if h_name and h_val:
        headers[h_name] = h_val
    return headers
