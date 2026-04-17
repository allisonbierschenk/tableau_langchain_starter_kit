"""LangChain tools that dispatch to MCPRouter (dual MCP, schema-driven)."""

from __future__ import annotations

import json
import re
from typing import Any, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from orchestration.mcp_router import MCPRouter
from orchestration.types import MCPServerId, ToolCatalog, ToolDefinition
from utilities.mcp_schema_models import build_tool_args_model_from_input_schema
from utilities.tableau_mcp_normalize import normalize_tableau_mcp_tool_call


class RoutedMCPLangChainTool(BaseTool):
    name: str
    description: str
    args_schema: type[BaseModel]
    router: Any = Field(exclude=True)
    server_id: MCPServerId = Field(exclude=True)
    native_tool_name: str = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async")

    async def _arun(self, **kwargs) -> str:
        try:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            if self.server_id == MCPServerId.TABLEAU_MCP:
                clean = normalize_tableau_mcp_tool_call(self.native_tool_name, clean)
            result = await self.router.call_tool(self.server_id, self.native_tool_name, clean)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


def create_routed_langchain_tools(router: MCPRouter, catalog: ToolCatalog) -> List[BaseTool]:
    tools: List[BaseTool] = []
    for t in catalog.tools:
        prefix = f"[{t.server_id.value}] "
        desc = prefix + (t.description or "No description")
        args_model = build_tool_args_model_from_input_schema(t.name, t.input_schema)
        tools.append(
            RoutedMCPLangChainTool(
                name=t.qualified_name,
                description=desc,
                args_schema=args_model,
                router=router,
                server_id=t.server_id,
                native_tool_name=t.name,
            )
        )
    return tools
