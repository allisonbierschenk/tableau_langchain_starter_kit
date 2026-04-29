"""Plan-then-execute orchestration over MCPRouter + BindingResolver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from orchestration.binding import BindingResolver
from orchestration.mcp_router import MCPRouter
from orchestration.task_planner import LLMTaskPlanner, PlanSchemaValidator
from orchestration.translation import McpTranslationLayer
from orchestration.types import PlannerContext, ToolCatalog


class OrchestrationEngine:
    def __init__(
        self,
        router: MCPRouter,
        *,
        translation: Optional[McpTranslationLayer] = None,
        planner: Optional[LLMTaskPlanner] = None,
    ):
        self.router = router
        self.planner = planner or LLMTaskPlanner()
        self.validator = PlanSchemaValidator()
        self.binding = BindingResolver(translation)

    async def plan_then_execute(self, ctx: PlannerContext, catalog: ToolCatalog) -> Dict[str, Any]:
        plan = await self.planner.plan(catalog, ctx)
        self.validator.validate(plan, catalog)
        step_results: Dict[str, Any] = {}
        trace: List[Dict[str, Any]] = []
        for step in plan.steps:
            args = await self.binding.build_tool_arguments(step.arguments, step_results)
            raw = await self.router.call_tool(step.server_id, step.tool_name, args)
            step_results[step.step_id] = raw
            trace.append(
                {
                    "step_id": step.step_id,
                    "server_id": step.server_id.value,
                    "tool_name": step.tool_name,
                    "arguments": args,
                    "result": raw,
                }
            )
        return {
            "plan": plan.model_dump(),
            "step_results": step_results,
            "trace": trace,
        }
