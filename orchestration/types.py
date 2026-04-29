"""Shared types for dual-MCP orchestration (no workflow-specific logic)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field


class MCPServerId(str, Enum):
    TABLEAU_MCP = "tableau_mcp"
    SLACK_MCP = "slack_mcp"


class ToolDefinition(BaseModel):
    """One MCP tool from tools/list, scoped to a server."""

    server_id: MCPServerId
    name: str = Field(description="Native MCP tool name for tools/call")
    qualified_name: str = Field(description="Stable id for LangChain + planner (OpenAI-safe)")
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ToolCatalog(BaseModel):
    tools: List[ToolDefinition]

    def by_qualified_name(self) -> Dict[str, ToolDefinition]:
        return {t.qualified_name: t for t in self.tools}

    def as_prompt_fragment(self) -> str:
        lines: List[str] = []
        for t in self.tools:
            lines.append(
                f"- qualified_name={t.qualified_name!r} server={t.server_id.value!r} "
                f"native_tool_name={t.name!r}\n  {t.description[:800]}"
            )
        return "\n".join(lines) if lines else "(no tools)"


class PlannerContext(BaseModel):
    user_goal: str
    conversation_turns: Optional[Sequence[Mapping[str, Any]]] = None
    constraints: Optional[Mapping[str, Any]] = None
    prior_attempts: Optional[Sequence[Mapping[str, Any]]] = None


class PlanStep(BaseModel):
    step_id: str
    server_id: MCPServerId
    tool_name: str = Field(description="Native MCP tool name on that server")
    intent: str = ""
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="ArgumentSource dicts per parameter (kind: literal|from_step|translate)",
    )


class ExecutionPlan(BaseModel):
    version: Literal["v1"] = "v1"
    steps: List[PlanStep]
    rationale: str = ""
    assumptions: Optional[List[str]] = None
    stop_conditions: Optional[List[str]] = None
