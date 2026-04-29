"""
Normalize Tableau MCP tool arguments toward REST API contracts before tools/call.

Generic (not workflow-specific): admin-users + add-user-to-site body shape, siteId LUID, siteRole casing.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

# Tableau REST allowed site roles (PascalCase) — Add User to Site
_CANONICAL_SITE_ROLES = frozenset(
    {
        "Creator",
        "Explorer",
        "ExplorerCanPublish",
        "SiteAdministratorExplorer",
        "SiteAdministratorCreator",
        "Unlicensed",
        "Viewer",
    }
)

_ROLE_ALIASES = {
    "siteadministrator": "SiteAdministratorCreator",
    "siteadministratorcreator": "SiteAdministratorCreator",
    "siteadministratorexplorer": "SiteAdministratorExplorer",
    "siteadministrator": "SiteAdministratorExplorer",
    "explorercanpublish": "ExplorerCanPublish",
}


def dump_mcp_argument_value(value: Any) -> Any:
    """Recursively turn Pydantic sub-models in dict/list tool kwargs into plain Python."""
    if hasattr(value, "model_dump"):
        try:
            return dump_mcp_argument_value(value.model_dump())
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: dump_mcp_argument_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [dump_mcp_argument_value(v) for v in value]
    return value


def coerce_value_to_plain_dict(value: Any) -> Dict[str, Any]:
    """
    Some models pass nested objects as JSON strings. LangChain may also leave
    sub-models as instances. Coerce to a dict so callers can use .get().
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped
        return {}
    if isinstance(value, str):
        s = value.strip()
        if len(s) >= 2 and s[0] == "{":
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_site_role_for_rest(value: Any) -> Optional[str]:
    """Map common LLM / user casing to REST PascalCase siteRole; return None if empty."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s in _CANONICAL_SITE_ROLES:
        return s
    key = re.sub(r"[^a-zA-Z]", "", s).lower()
    if key in _ROLE_ALIASES:
        return _ROLE_ALIASES[key]
    # Title-case heuristic for simple names
    tc = s[:1].upper() + s[1:] if s else s
    if tc in _CANONICAL_SITE_ROLES:
        return tc
    # e.g. viewer -> Viewer
    for canon in _CANONICAL_SITE_ROLES:
        if canon.lower() == s.lower():
            return canon
    return s


def _normalize_add_user_to_site_body(body: Dict[str, Any]) -> None:
    """Mutate body in place to REST JSON shape { user: { name, siteRole, ... } }."""
    if not isinstance(body, dict):
        return

    user: Dict[str, Any]
    raw_user = body.get("user")
    if isinstance(raw_user, dict):
        user = raw_user
    else:
        user = coerce_value_to_plain_dict(raw_user)
        body["user"] = user

    # Flat mistakes: email / siteRole at body root
    for flat_key in ("email", "userEmail"):
        if flat_key in body and body[flat_key] and not user.get("name"):
            user["name"] = str(body.pop(flat_key)).strip()
    if "siteRole" in body and body["siteRole"] is not None and not user.get("siteRole"):
        sr = normalize_site_role_for_rest(body.pop("siteRole"))
        if sr:
            user["siteRole"] = sr

    if user.get("siteRole") is not None:
        sr2 = normalize_site_role_for_rest(user.get("siteRole"))
        if sr2:
            user["siteRole"] = sr2

    # Optional REST fields on user — keep if already nested
    for opt in ("authSetting", "identityPoolName", "idpConfigurationId", "email"):
        if opt in body and opt not in user and body[opt] is not None:
            user[opt] = body.pop(opt)


def normalize_tableau_mcp_tool_call(native_tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a shallow-safe copy of arguments with Tableau REST-oriented fixes applied.
    """
    out: Dict[str, Any] = {
        k: dump_mcp_argument_value(v) for k, v in dict(arguments or {}).items()
    }

    if native_tool_name == "admin-users" and "body" in out:
        out["body"] = coerce_value_to_plain_dict(out.get("body"))

    if native_tool_name != "admin-users":
        return out

    op = out.get("operation")
    if op != "add-user-to-site":
        return out

    body = out.get("body")
    if not isinstance(body, dict):
        body = coerce_value_to_plain_dict(body)
        out["body"] = body
    _normalize_add_user_to_site_body(body)

    # site-id in URI must be LUID: if model passed content URL name, substitute when configured
    site_luid = (os.getenv("TABLEAU_SITE_LUID") or "").strip()
    site_name = (os.getenv("TABLEAU_SITE") or "").strip()
    sid = out.get("siteId")
    if site_luid and site_name and sid is not None:
        sid_s = str(sid).strip()
        if sid_s.lower() == site_name.lower():
            out["siteId"] = site_luid

    return out


def maybe_log_admin_users_schema(mcp_tools: list) -> None:
    """If MCP_DEBUG_LOG_ADMIN_USERS_SCHEMA=1, print admin-users inputSchema (for MCP server debugging)."""
    if (os.getenv("MCP_DEBUG_LOG_ADMIN_USERS_SCHEMA") or "").strip() not in ("1", "true", "yes"):
        return
    for t in mcp_tools or []:
        if t.get("name") == "admin-users":
            schema = t.get("inputSchema") or {}
            s = json.dumps(schema, indent=2)
            if len(s) > 8000:
                s = s[:7980] + "\n... (truncated)"
            print("MCP_DEBUG admin-users inputSchema:", s)
            return
    print("MCP_DEBUG: no admin-users tool in tools/list")
