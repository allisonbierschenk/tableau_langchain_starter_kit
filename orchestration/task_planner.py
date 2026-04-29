"""LLM-backed TaskPlanner and structural plan validation."""

from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from orchestration.types import ExecutionPlan, PlannerContext, ToolCatalog


PLANNER_SYSTEM = """You are a TaskPlanner for a dual MCP system (Tableau + Slack).

You must emit a structured ExecutionPlan (version v1) that:
- Uses ONLY tools that appear in the tool catalog (match server_id and native tool_name exactly).
- Orders steps so dependencies come first. step_id must be unique (e.g. s1, s2, ...).
- For each step, `arguments` maps parameter names to ArgumentSource objects:
  - literal: {"kind": "literal", "value": <json-serializable>}
  - from_step: {"kind": "from_step", "step_id": "s1", "pointer": "/result/0/email"}  (JSON Pointer RFC 6901)
  - translate: {"kind": "translate", "translation_key": "email_to_slack_user_id", "source": {"kind": "literal", "value": {"email": "a@b.com"}}}
- Prefer tableau_mcp for data/admin; slack_mcp for messaging after facts are established.
- Do not invent tool names or parameters not in the catalog schemas.
"""


class PlanSchemaValidator:
    """Structural validation only (no domain rules)."""

    def validate(self, plan: ExecutionPlan, catalog: ToolCatalog) -> None:
        allowed = {(t.server_id, t.name) for t in catalog.tools}
        seen_step_ids: List[str] = []
        for step in plan.steps:
            if (step.server_id, step.tool_name) not in allowed:
                raise ValueError(
                    f"Plan step {step.step_id!r} references unknown tool "
                    f"{step.server_id.value!r} / {step.tool_name!r}"
                )
            for arg_name, src in step.arguments.items():
                if not isinstance(src, dict):
                    raise TypeError(f"Step {step.step_id} arg {arg_name!r} must be a dict ArgumentSource")
                self._walk_sources(src, seen_step_ids, step.step_id)
            seen_step_ids.append(step.step_id)

    def _walk_sources(self, src: dict, prior_step_ids: List[str], current_step_id: str) -> None:
        self._validate_source(src, prior_step_ids, current_step_id)
        if src.get("kind") == "translate":
            nested = src.get("source")
            if isinstance(nested, dict):
                self._walk_sources(nested, prior_step_ids, current_step_id)

    def _validate_source(self, src: dict, prior_step_ids: List[str], current_step_id: str) -> None:
        kind = src.get("kind")
        if kind == "literal":
            return
        if kind == "from_step":
            ref = str(src.get("step_id") or "")
            if ref not in prior_step_ids:
                raise ValueError(
                    f"Step {current_step_id!r}: from_step references unknown or future step_id {ref!r}"
                )
            return
        if kind == "translate":
            return
        if kind == "executor_fill":
            raise ValueError("executor_fill is not allowed in validated plans")
        raise ValueError(f"Unknown ArgumentSource kind {kind!r}")


class LLMTaskPlanner:
    def __init__(self, llm: ChatOpenAI | None = None):
        import os

        self._llm = llm or ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )

    async def plan(self, catalog: ToolCatalog, ctx: PlannerContext) -> ExecutionPlan:
        structured = self._llm.with_structured_output(ExecutionPlan)
        human = (
            f"User goal:\n{ctx.user_goal}\n\n"
            f"Tool catalog:\n{catalog.as_prompt_fragment()}\n"
        )
        if ctx.constraints:
            human += f"\nConstraints:\n{ctx.constraints}\n"
        if ctx.prior_attempts:
            human += f"\nPrior failed attempts (replan):\n{ctx.prior_attempts}\n"
        msg = await structured.ainvoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=human),
            ]
        )
        assert isinstance(msg, ExecutionPlan)
        return msg
