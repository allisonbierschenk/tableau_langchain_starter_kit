"""Resolve ExecutionPlan ArgumentSource dicts into concrete Python values."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from orchestration.translation import TranslationLayer


def _json_pointer_get(doc: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must start with /")
    cur = doc
    for raw_seg in pointer.lstrip("/").split("/"):
        seg = raw_seg.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            cur = cur[seg]
        else:
            raise KeyError(pointer)
    return cur


class BindingResolver:
    def __init__(self, translation: Optional[TranslationLayer] = None):
        self._translation = translation

    def resolve_argument_source(
        self,
        source: Mapping[str, Any],
        step_results: Dict[str, Any],
    ) -> Any:
        kind = source.get("kind")
        if kind == "literal":
            return source.get("value")
        if kind == "from_step":
            step_id = str(source.get("step_id") or "")
            ptr = str(source.get("pointer") or "/")
            if step_id not in step_results:
                raise KeyError(f"from_step: unknown step_id {step_id!r}")
            return _json_pointer_get(step_results[step_id], ptr)
        if kind == "translate":
            raise RuntimeError("translate ArgumentSource requires resolve_argument_source_async")

        raise ValueError(f"Unknown ArgumentSource kind: {kind!r}")

    async def resolve_argument_source_async(
        self,
        source: Mapping[str, Any],
        step_results: Dict[str, Any],
    ) -> Any:
        kind = source.get("kind")
        if kind == "literal":
            return source.get("value")
        if kind == "from_step":
            step_id = str(source.get("step_id") or "")
            ptr = str(source.get("pointer") or "/")
            if step_id not in step_results:
                raise KeyError(f"from_step: unknown step_id {step_id!r}")
            return _json_pointer_get(step_results[step_id], ptr)
        if kind == "translate":
            if not self._translation:
                raise RuntimeError("translate ArgumentSource requires TranslationLayer")
            key = str(source.get("translation_key") or "")
            nested = source.get("source")
            if not isinstance(nested, dict):
                raise ValueError("translate.source must be an ArgumentSource dict")
            inner = await self.resolve_argument_source_async(nested, step_results)
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    inner = {"email": inner}
            if not isinstance(inner, Mapping):
                raise TypeError("translate inner must resolve to mapping or JSON object string")
            return await self._translation.resolve(key, inner)
        if kind == "executor_fill":
            raise NotImplementedError(
                "executor_fill is reserved for future schema-guided fill; not implemented"
            )
        raise ValueError(f"Unknown ArgumentSource kind: {kind!r}")

    async def build_tool_arguments(
        self,
        arguments: Dict[str, Mapping[str, Any]],
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for arg_name, src in arguments.items():
            out[arg_name] = await self.resolve_argument_source_async(dict(src), step_results)
        return out
