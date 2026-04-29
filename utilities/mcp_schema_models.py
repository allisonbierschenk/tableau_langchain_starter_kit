"""
Build dynamic Pydantic models from MCP JSON Schema (top-level + one nested object level).

Deeper nesting (e.g. body.user.*) is expanded recursively up to MAX_DEPTH for object types.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, create_model


MAX_SCHEMA_DEPTH = 6


def _sanitize_model_name(name: str) -> str:
    return (re.sub(r"[^0-9a-zA-Z_]", "_", name)[:80] or "MCPArgs").strip("_")


def _scalar_python_type(prop_details: Dict[str, Any]) -> type:
    t = prop_details.get("type", "string")
    if t == "number":
        return float
    if t in ("integer", "int"):
        return int
    if t == "boolean":
        return bool
    if t == "array":
        return list
    return str


def _object_model_from_schema(model_name: str, schema: Dict[str, Any], depth: int) -> Optional[type[BaseModel]]:
    """Return a Pydantic model for type object with properties, or None to fall back to dict."""
    if depth > MAX_SCHEMA_DEPTH:
        return None
    if schema.get("type") != "object":
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return None

    required = set(schema.get("required") or [])
    field_defs: Dict[str, Tuple[Any, Any]] = {}

    for pname, pdetails in props.items():
        if not isinstance(pdetails, dict):
            continue
        is_req = pname in required
        py_t, field = _field_for_property(pname, pdetails, f"{model_name}_{pname}", is_req, depth + 1)
        field_defs[pname] = (py_t, field)

    if not field_defs:
        return None

    safe = _sanitize_model_name(model_name)
    return create_model(
        safe,
        __config__=ConfigDict(extra="allow"),  # Allow extra params: MCP schemas may be incomplete vs actual tool capabilities
        **{k: (v[0], v[1]) for k, v in field_defs.items()},
    )


def _field_for_property(
    pname: str,
    pdetails: Dict[str, Any],
    path: str,
    is_required: bool,
    depth: int,
) -> Tuple[Any, Any]:
    ptype = pdetails.get("type", "string")

    if ptype == "object" and isinstance(pdetails.get("properties"), dict) and pdetails["properties"]:
        nested = _object_model_from_schema(path, pdetails, depth)
        if nested is not None:
            if is_required:
                return (nested, Field(..., description=str(pdetails.get("description", ""))))
            return (Optional[nested], Field(None, description=str(pdetails.get("description", ""))))

    # Many MCP tools (including tableau-mcp `admin-users`) declare `body` as JSON Schema
    # `type: string` while callers and REST expect a JSON object. Pre-refactor admin tools
    # always bound `body` as dict — keep that so structured tool calls validate.
    if pname == "body":
        desc = str(pdetails.get("description", ""))
        if is_required:
            return (Dict[str, Any], Field(..., description=desc))
        return (Optional[Dict[str, Any]], Field(None, description=desc))

    py_t = _scalar_python_type(pdetails)
    desc = str(pdetails.get("description", ""))
    if py_t is dict:
        py_t = dict  # unreachable guard
    if is_required:
        return (py_t, Field(..., description=desc))
    return (Optional[py_t], Field(None, description=desc))


def build_tool_args_model_from_input_schema(tool_name: str, input_schema: Dict[str, Any]) -> type[BaseModel]:
    """
    Build a Pydantic model for MCP tools/call arguments from inputSchema.
    Uses nested models for object properties (including body.user.*) when schema lists properties.
    """
    props = input_schema.get("properties") or {}
    if not props:
        class EmptyMCPArgs(BaseModel):
            model_config = ConfigDict(extra="allow")  # Allow extra params: MCP schemas may be incomplete

        return EmptyMCPArgs

    required = set(input_schema.get("required") or [])
    prefix = _sanitize_model_name(f"{tool_name}_args")
    field_defs: Dict[str, Tuple[Any, Any]] = {}

    for pname, pdetails in props.items():
        if not isinstance(pdetails, dict):
            continue
        is_req = pname in required
        py_t, fld = _field_for_property(pname, pdetails, f"{prefix}_{pname}", is_req, 0)
        field_defs[pname] = (py_t, fld)

    return create_model(
        prefix,
        __config__=ConfigDict(extra="allow"),  # Allow extra params: MCP schemas may be incomplete vs actual tool capabilities
        **{k: (v[0], v[1]) for k, v in field_defs.items()},
    )
