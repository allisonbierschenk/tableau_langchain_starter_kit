"""
Best-effort enrichment: when the assistant text lists Pulse "Metric ID" lines but
tool results already contained id→title pairs in JSON, append the display name.

When payloads are id-only, optional MCP backfill walks site-wide Pulse catalog
responses (e.g. list-all-pulse-metric-definitions) to resolve UUID→title.

Does not invent names—only uses strings returned from MCP JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


def _is_uuid(s: str) -> bool:
    return bool(s and _UUID.fullmatch(s.strip()))


def _walk_pulse_id_names(obj: Any, acc: Dict[str, str]) -> None:
    """Collect lowercase UUID -> display name from nested Pulse-like JSON."""
    if isinstance(obj, dict):
        for nested_key in ("metricDefinition", "metric", "pulseMetric", "pulseMetricDefinition"):
            md = obj.get(nested_key)
            if isinstance(md, dict):
                mid = md.get("id") or md.get("luid")
                nm = md.get("name") or md.get("title")
                if isinstance(mid, str) and _is_uuid(str(mid)) and isinstance(nm, str) and nm.strip():
                    acc[str(mid).lower()] = nm.strip()

        for key in (
            "metricDefinitionId",
            "metricId",
            "pulseMetricId",
            "pulseDefinitionId",
            "definitionId",
        ):
            if key not in obj:
                continue
            v = obj[key]
            if not isinstance(v, str) or not _is_uuid(v):
                continue
            nm = (
                obj.get("name")
                or obj.get("title")
                or obj.get("metricName")
                or obj.get("displayName")
                or obj.get("specifiedName")
                or obj.get("vizTitle")
            )
            if isinstance(nm, str) and nm.strip():
                acc[v.lower()] = nm.strip()

        for val in obj.values():
            _walk_pulse_id_names(val, acc)
    elif isinstance(obj, list):
        for item in obj:
            _walk_pulse_id_names(item, acc)


def _iter_json_roots_from_tool_result(tr: Dict[str, Any]) -> List[Any]:
    """Yield dict/list JSON roots embedded in a single tool result."""
    roots: List[Any] = []
    r = tr.get("result")

    if isinstance(r, str):
        s = r.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                roots.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    elif isinstance(r, dict):
        roots.append(r)
        for block in r.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip().startswith(("{", "[")):
                    try:
                        roots.append(json.loads(t))
                    except json.JSONDecodeError:
                        pass
    elif isinstance(r, list):
        roots.append(r)
    return roots


def _walk_broad_uuid_to_display_name(obj: Any, acc: Dict[str, str]) -> None:
    """
    Walk any Pulse / REST JSON tree: register id|luid (UUID) -> best available label.
    Used on large catalog payloads (list-all-pulse-metric-definitions).
    """
    if isinstance(obj, dict):
        oid = obj.get("id") or obj.get("luid") or obj.get("pulseMetricLuid")
        name = (
            obj.get("name")
            or obj.get("title")
            or obj.get("specifiedName")
            or obj.get("vizTitle")
            or obj.get("displayName")
            or obj.get("metricName")
        )
        if isinstance(oid, str) and _is_uuid(str(oid)) and isinstance(name, str) and name.strip():
            acc[str(oid).lower()] = name.strip()
        for val in obj.values():
            _walk_broad_uuid_to_display_name(val, acc)
    elif isinstance(obj, list):
        for item in obj:
            _walk_broad_uuid_to_display_name(item, acc)


def metric_uuids_candidates_in_response(text: str) -> List[str]:
    """UUIDs on lines that look like Pulse metric listings (avoid unrelated admin UUIDs)."""
    if not text:
        return []
    low = text.lower()
    has_pulse_context = (
        "metric id" in low
        or "pulse" in low
        or "subscription" in low
        or "did not return human-readable" in low
    )
    if not has_pulse_context:
        return []
    ordered: List[str] = []
    seen: Set[str] = set()
    for line in text.split("\n"):
        ll = line.lower()
        if "metric id" not in ll and "pulse" not in ll:
            continue
        for m in _UUID.finditer(line):
            u = m.group(0)
            key = u.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(u)
    if not ordered and "did not return human-readable" in low:
        for m in _UUID.finditer(text):
            u = m.group(0)
            key = u.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(u)
    return ordered


def _pulse_catalog_tool_names(tableau_mcp_tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Ordered MCP tool names likely to return id+name maps for Pulse metrics."""
    if not tableau_mcp_tools:
        return []
    names = [t.get("name") or "" for t in tableau_mcp_tools]
    name_set = set(names)
    out: List[str] = []
    preferred = "list-all-pulse-metric-definitions"
    if preferred in name_set:
        out.append(preferred)
    for n in names:
        if not n or n in out:
            continue
        nl = n.lower()
        if "pulse" not in nl:
            continue
        if "list" in nl and "definition" in nl and "metric" in nl:
            out.append(n)
    return out[:5]


async def backfill_pulse_id_names_via_mcp(
    admin_client: Any,
    tableau_mcp_tools: Optional[List[Dict[str, Any]]],
    response_text: str,
    existing: Dict[str, str],
) -> Dict[str, str]:
    """
    If the assistant answer still references Pulse metric UUIDs without resolved titles,
    call lightweight MCP catalog tools and extract id→name from JSON.
    """
    uuids = metric_uuids_candidates_in_response(response_text)
    missing = [u for u in uuids if u.lower() not in existing]
    low = response_text.lower()
    if not uuids:
        return {}
    if not missing and "did not return human-readable" not in low:
        return {}

    extra: Dict[str, str] = {}
    for tool_name in _pulse_catalog_tool_names(tableau_mcp_tools):
        try:
            raw = await admin_client.call_tool(tool_name, {})
        except Exception as ex:
            print(f"⚠️ Pulse name backfill: {tool_name} failed: {ex}")
            continue
        tr = {"result": raw}
        for root in _iter_json_roots_from_tool_result(tr):
            _walk_pulse_id_names(root, extra)
            _walk_broad_uuid_to_display_name(root, extra)
        if uuids and all(u.lower() in {**existing, **extra} for u in uuids):
            print(f"📊 Pulse name backfill: resolved via `{tool_name}`")
            break
    return dict(extra)


def scrub_stale_pulse_api_disclaimer(text: str, uuids_in_answer: List[str], id_to_name: Dict[str, str]) -> str:
    """Remove the model's 'API did not return names' blurb when we now have every UUID mapped."""
    if not text or not uuids_in_answer:
        return text
    if any(u.lower() not in id_to_name for u in uuids_in_answer):
        return text
    low = text.lower()
    if "did not return human-readable" not in low:
        return text
    text = re.sub(
        r"(?is)\n?\s*Unfortunately,\s*the\s*API\s*did\s*not\s*return\s*human-readable\s*names\s*for\s*these\s*metrics\.?\s*",
        "\n",
        text,
    )
    text = re.sub(
        r"(?is)\n?\s*If\s*you\s*need\s*more\s*detailed\s*information.*?additional\s*details\.?\s*",
        "\n",
        text,
    )
    return text.strip()


def extract_pulse_metric_id_to_name(tool_results: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """Build map metric-uuid (lowercase) -> human-readable name from all tool results."""
    out: Dict[str, str] = {}
    for tr in tool_results or []:
        for root in _iter_json_roots_from_tool_result(tr):
            _walk_pulse_id_names(root, out)
    return out


def enrich_assistant_text_pulse_metric_names(text: str, id_to_name: Dict[str, str]) -> str:
    """
    For lines that mention Metric ID but not Metric Name, append resolved titles
    when id_to_name has an entry for that UUID (from the same request's tools).
    """
    if not text or not id_to_name:
        return text
    lines = text.split("\n")
    new_lines: List[str] = []
    for line in lines:
        low = line.lower()
        if "metric id" not in low or "metric name" in low:
            new_lines.append(line)
            continue
        titles: List[str] = []
        for uid_l, title in id_to_name.items():
            if re.search(rf"\b{re.escape(uid_l)}\b", line, flags=re.IGNORECASE):
                if title.lower() not in line.lower():
                    titles.append(title)
        if titles:
            ordered = list(dict.fromkeys(titles))
            suffix = ", ".join(f"**{t}**" for t in ordered)
            line = line.rstrip() + f" — **Metric name:** {suffix}"
        new_lines.append(line)
    return "\n".join(new_lines)
