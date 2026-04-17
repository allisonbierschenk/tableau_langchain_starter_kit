"""Build ToolCatalog from MCPRouter (tools/list per server)."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from orchestration.mcp_router import MCPRouter
from orchestration.types import MCPServerId, ToolCatalog, ToolDefinition


def _langchain_safe_name(server_id: MCPServerId, native_name: str) -> str:
    """OpenAI function names: ^[a-zA-Z0-9_-]{1,64}$"""
    base = f"{server_id.value}__{native_name}"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", base)
    if len(safe) > 64:
        safe = safe[:61] + "_x"
    return safe


def tool_definitions_for_server(
    server_id: MCPServerId,
    raw_tools: List[Dict[str, Any]],
) -> List[ToolDefinition]:
    """Normalize tools/list entries for one MCP server."""
    out: List[ToolDefinition] = []
    seen_lc: set[str] = set()
    for t in raw_tools:
        native = str(t.get("name") or "")
        if not native:
            continue
        qn = _langchain_safe_name(server_id, native)
        suf = 2
        while qn in seen_lc:
            qn = _langchain_safe_name(server_id, f"{native}__{suf}")
            suf += 1
        seen_lc.add(qn)
        out.append(
            ToolDefinition(
                server_id=server_id,
                name=native,
                qualified_name=qn,
                description=str(t.get("description") or ""),
                input_schema=t.get("inputSchema") or {},
            )
        )
    return out


class ToolCatalogBuilder:
    @staticmethod
    async def load(router: MCPRouter) -> ToolCatalog:
        out: List[ToolDefinition] = []
        preferred = [MCPServerId.TABLEAU_MCP, MCPServerId.SLACK_MCP]
        for server_id in preferred:
            if server_id not in router.clients:
                continue
            raw_tools: List[Dict[str, Any]] = await router.list_tools_for(server_id)
            out.extend(tool_definitions_for_server(server_id, raw_tools))
        return ToolCatalog(tools=out)
