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


def _walk_pulse_id_names(obj: Any, acc: Dict[str, str], _depth: int = 0) -> None:
    """Collect lowercase UUID -> display name from nested Pulse-like JSON."""
    if isinstance(obj, dict):
        # Log potential subscription objects for debugging
        if _depth == 0 and ("metricDefinitionId" in obj or "subscriptionId" in obj):
            keys = list(obj.keys())
            print(f"🔍 Found subscription-like object with keys: {keys[:10]}")

        # PATTERN 1: Top-level objects with id + specification (batch-get-metrics response)
        # Example: {"id": "uuid", "specification": {"basic": {"name": "Metric Name"}, ...}}
        if "id" in obj and "specification" in obj:
            mid = obj.get("id")
            spec = obj.get("specification")

            # DEBUG: Show what we're looking at (ANY depth, not just 0)
            if isinstance(mid, str) and _is_uuid(str(mid)):
                print(f"🔍 Found metric object (depth={_depth}): id={mid[:8]}...")
                print(f"   ALL TOP-LEVEL KEYS: {list(obj.keys())}")
                print(f"   specification type: {type(spec).__name__}")
                if isinstance(spec, dict):
                    print(f"   specification keys: {list(spec.keys())[:10]}")
                    basic = spec.get("basic")
                    print(f"   specification.basic type: {type(basic).__name__ if basic else 'None'}")
                    if isinstance(basic, dict):
                        print(f"   specification.basic keys: {list(basic.keys())}")
                        print(f"   specification.basic.name: {basic.get('name')}")
                    else:
                        print(f"   ⚠️ specification.basic is not a dict!")

                # Check for name at top level or other common locations
                for key in ['name', 'title', 'display_name', 'metric_name', 'definition_name']:
                    val = obj.get(key)
                    if val:
                        print(f"   Found at top level: obj.{key} = {val}")

            if isinstance(mid, str) and _is_uuid(str(mid)) and isinstance(spec, dict):
                # Try specification.basic.name (Pulse API v1 pattern)
                basic = spec.get("basic") or {}
                nm = basic.get("name") if isinstance(basic, dict) else None
                # Fallback: specification.name or top-level name
                if not nm:
                    nm = spec.get("name") or spec.get("title") or obj.get("name") or obj.get("title")
                if isinstance(nm, str) and nm.strip():
                    acc[str(mid).lower()] = nm.strip()
                    print(f"📊 Extracted: {mid[:8]}... → {nm.strip()}")
                else:
                    print(f"⚠️ No name found for metric {mid[:8]}... at depth {_depth}")
                    print(f"   Checked: basic.name={basic.get('name') if isinstance(basic, dict) else 'N/A'}")
                    print(f"   Checked: spec.name={spec.get('name') if isinstance(spec, dict) else 'N/A'}")
                    print(f"   Checked: spec.title={spec.get('title') if isinstance(spec, dict) else 'N/A'}")

        # PATTERN 2: Nested metricDefinition/metric objects (legacy pattern)
        for nested_key in ("metricDefinition", "metric", "pulseMetric", "pulseMetricDefinition"):
            md = obj.get(nested_key)
            if isinstance(md, dict):
                mid = md.get("id") or md.get("luid")
                nm = md.get("name") or md.get("title")
                if isinstance(mid, str) and _is_uuid(str(mid)) and isinstance(nm, str) and nm.strip():
                    acc[str(mid).lower()] = nm.strip()

        # PATTERN 3: Objects with metricDefinitionId/metricId + name at same level
        for key in (
            "metricDefinitionId",
            "metricId",
            "metric_id",  # snake_case variant from list-subscriptions
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
            # Also check nested specification.basic.name for this ID
            if not nm:
                spec = obj.get("specification")
                if isinstance(spec, dict):
                    basic = spec.get("basic")
                    if isinstance(basic, dict):
                        nm = basic.get("name")
            if isinstance(nm, str) and nm.strip():
                acc[v.lower()] = nm.strip()
                print(f"📊 Extracted from ID field: {v[:8]}... → {nm.strip()}")

        for val in obj.values():
            _walk_pulse_id_names(val, acc, _depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _walk_pulse_id_names(item, acc, _depth + 1)


def _parse_content_blocks_for_json(obj: Any, roots: List[Any]) -> None:
    """Helper to parse content[].text blocks that might contain JSON strings."""
    if not isinstance(obj, dict):
        return
    content_blocks = obj.get("content") or []
    if not content_blocks:
        return
    print(f"   🔍 Checking {len(content_blocks)} content blocks for nested JSON")
    for i, block in enumerate(content_blocks):
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                t_trimmed = t.strip()
                if t_trimmed.startswith(("{", "[")):
                    try:
                        parsed = json.loads(t)
                        roots.append(parsed)
                        size = len(parsed) if isinstance(parsed, (list, dict)) else "?"
                        print(f"   ✅ Parsed content[{i}].text as JSON: {type(parsed).__name__} with {size} items")
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ JSON decode failed for content[{i}].text: {str(e)[:100]}")

def _iter_json_roots_from_tool_result(tr: Dict[str, Any]) -> List[Any]:
    """Yield dict/list JSON roots embedded in a single tool result."""
    roots: List[Any] = []
    r = tr.get("result")

    if isinstance(r, str):
        s = r.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
                roots.append(parsed)
                print(f"   🔍 Parsed string result as JSON: {type(parsed).__name__}")
                # CRITICAL: Also check if this parsed dict has content blocks with more JSON!
                _parse_content_blocks_for_json(parsed, roots)
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON decode failed for string result: {e}")
    elif isinstance(r, dict):
        roots.append(r)
        # Check for nested JSON in content blocks
        _parse_content_blocks_for_json(r, roots)
    elif isinstance(r, list):
        roots.append(r)
        print(f"   🔍 Result is already a list with {len(r)} items")
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
        # Also check specification.basic.name for Pulse API v1
        if not name and "specification" in obj:
            spec = obj.get("specification")
            if isinstance(spec, dict):
                basic = spec.get("basic")
                if isinstance(basic, dict):
                    name = basic.get("name")
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
        tool_name = tr.get("tool", "unknown")

        # Debug: Show what we're processing
        if "pulse" in tool_name.lower():
            print(f"🔍 Processing tool result from: {tool_name}")
            result_preview = str(tr.get("result", ""))[:200]
            print(f"   Result preview: {result_preview}")

        roots = list(_iter_json_roots_from_tool_result(tr))

        if "pulse" in tool_name.lower():
            print(f"   Found {len(roots)} JSON roots to walk")
            for i, root in enumerate(roots):
                root_type = type(root).__name__
                if isinstance(root, list):
                    print(f"     Root {i}: list with {len(root)} items")
                elif isinstance(root, dict):
                    print(f"     Root {i}: dict with keys: {list(root.keys())[:10]}")
                else:
                    print(f"     Root {i}: {root_type}")

        for root in roots:
            before = len(out)
            _walk_pulse_id_names(root, out)
            after = len(out)
            if after > before:
                print(f"   ✅ Extracted {after - before} Pulse metric names from {tool_name}")

    if out:
        print(f"📊 Total extracted metric names: {len(out)}")
        for mid, mname in list(out.items())[:3]:
            print(f"   - {mid[:8]}... → {mname}")
    else:
        print(f"⚠️ No metric names extracted from {len(tool_results or [])} tool results")
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
