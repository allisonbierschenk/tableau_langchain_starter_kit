"""
Tableau MCP + LangChain integration.

Uses the official langchain-mcp-adapters package with the MCP Python SDK's
streamable HTTP transport. No custom JSON-RPC parsing or schema workarounds.
"""

import os
import json
import logging
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

from utilities.viewer_email import is_likely_email

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
if not MCP_SERVER_URL:
    raise ValueError("MCP_SERVER_URL environment variable is required")

MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are an intelligent data analyst with access to Tableau data through MCP tools. Help users understand their data by using tools strategically and presenting insights.

PRIORITY: DASHBOARD / "THIS DATA" CONTEXT
When the user is in a dashboard and you are given a **Dashboard context** with a datasource LUID below:
- Treat these as requests to **use the dashboard datasource and return data-driven answers** (do NOT answer with generic advice without querying):
  - "top insights from this data", "insights from this data", "analyze this dashboard", "what does this data show"
  - "What should I focus on to be proactive?", "What should I do next?", "What are my priorities?", "Give me recommendations", "What matters most?"
- For proactive/recommendation questions: use `get-datasource-metadata` to see fields, then `query-datasource` to get key metrics (e.g. top/bottom performers, trends, totals, problem areas). Then give 3–5 **concrete, data-backed recommendations** with numbers (e.g. "Focus on Region X—sales are down 15% vs last period" or "Customer segment Y has the highest churn; prioritize retention there").
- Use `get-datasource-metadata` with that datasourceLuid to get fields, then `query-datasource` with that LUID to run queries and return insights with numbers. Never respond with only generic advice when a dashboard LUID is provided—always query the data first.
- Do NOT use Pulse tools for "this data" or "dashboard" questions when a dashboard datasource LUID is provided.

DATA SOURCE QUESTIONS (general or when no dashboard LUID is provided):
1. Use `list-datasources` to find data sources (or use the provided LUID when in dashboard context).
2. Use `get-datasource-metadata` to get field information.
3. Use `query-datasource` to run queries and get actual data. Prefer aggregation (SUM, COUNT, AVG) and TOP filters for "top N" questions.

PULSE METRICS (only when user explicitly asks about Pulse/metrics):
- Use when the user says "my pulse metrics", "list pulse metrics", "pulse insights", "Tableau Pulse", etc.
- Use `list-all-pulse-metric-definitions` or `list-pulse-metric-subscriptions` to discover metrics, then `list-pulse-metrics-from-metric-definition-id` to get metric IDs and details.
- For a short summary: present the list of metrics and their definitions; you do not need to call `generate-pulse-metric-value-insight-bundle`.
- **Do NOT call `generate-pulse-metric-value-insight-bundle`** with only `pulseMetricIds`. That tool requires a **full bundle_request object** (version, options, input.metadata, input.metric with definition, specification, etc.). Building that from list results is complex. Prefer summarizing list results or using `generate-pulse-insight-brief` for natural language questions about metrics if the request fits that tool's parameters.

WORKBOOK/VIEW QUESTIONS:
- Use `list-workbooks`, `get-workbook`, `list-views`, `get-view-data`, `get-view-image` as needed.

GENERAL:
- Use `search-content` to search across workbooks, views, datasources.
- Ground responses in actual tool results; include numbers and brief insights where possible.
- **When presenting insights or numbers from query-datasource, include at the start of your response a clear scope block** (so users see it immediately, not buried in the body). Format it like this at the top of your reply:
  **Measures:** [e.g. SUM(Sales), COUNT(Orders)]
  **Time frame:** [e.g. last complete quarter, current month, all time, or no time filter]
  **Filters:** [e.g. none, or Region = West, Segment = Enterprise]
  Then give the insights in the body. Do not only mention these in passing in the body—put them in this visible block at the start of the response.
- If a tool fails, use discovery tools first or try an alternative (e.g. query-datasource with the provided LUID for "this data" instead of Pulse)."""


def _build_system_message(
    tools: list,
    tableau_viewer_id: Optional[str],
    preferred_datasource_name: Optional[str],
    resolved_luid: Optional[str],
    dashboard_has_pulse_objects: bool,
) -> str:
    parts = [SYSTEM_PROMPT]

    tool_list = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    parts.append(f"\nAvailable tools:\n{tool_list}")

    if tableau_viewer_id:
        parts.append(
            f"\n**Viewer identity (from extension / workbook):** `{tableau_viewer_id}`. "
            "If it contains @, treat it as the viewer's email when appropriate. "
            "Use for personalization only when appropriate."
        )

    if resolved_luid and preferred_datasource_name:
        parts.append(f"""
**Dashboard context:** The user is viewing a dashboard connected to the datasource "{preferred_datasource_name}". For ANY of these you MUST use the datasource and return data-driven answers (never generic advice without querying):
- "top insights", "insights from this data", "analyze this dashboard"
- "What should I focus on to be proactive?", "What should I do next?", "Give me recommendations", "What are my priorities?"
Steps: (1) get-datasource-metadata with datasourceLuid below to see fields. (2) query-datasource with this LUID to get key metrics (top/bottom performers, trends, totals). (3) Answer with 3–5 concrete insights or recommendations with numbers from the query results.
Use only query-datasource and get-datasource-metadata with:
- datasourceLuid: `{resolved_luid}`

**Start every insight response with a visible scope block:** At the very start of your reply (before the insights body), include these three lines so they appear clearly in the response:
**Measures:** [list what was aggregated, e.g. SUM(Revenue), COUNT(Orders)]
**Time frame:** [e.g. last complete quarter, current month, all time, or no time filter]
**Filters:** [e.g. none, or list any filters applied]
Then write the insights. The scope block must be at the top of the response, not only in the body text.

**Pulse Metric cards on the dashboard:** The dashboard may contain Pulse Metric objects. Those cards show values from specific metric definitions (fixed measure, time period, filters). Your answers use query-datasource on the same datasource with flexible queries, so **the numbers you return will not match the Pulse cards.** You MUST add one short sentence to every insight response, e.g.: "These numbers are from the same datasource with flexible queries; if your dashboard has Pulse Metric cards, their values use specific metric definitions and may differ." If the user says the numbers don't match, explain that Pulse cards are predefined metrics and this chat uses ad-hoc queries; for Pulse-based summaries they can ask "List my Pulse metrics" or "Summarize my Pulse metrics.""")

    return "\n".join(parts)


async def _resolve_datasource_luid(
    tools: list, preferred_name: str
) -> Optional[str]:
    """Call list-datasources to resolve a datasource name to its published LUID."""
    list_ds = next((t for t in tools if t.name == "list-datasources"), None)
    if list_ds is None:
        return None
    try:
        result = await list_ds.ainvoke({"filter": f"name:eq:{preferred_name}"})
        # Tool returns a string; parse to find the id/luid field
        text = result if isinstance(result, str) else json.dumps(result)
        parsed = json.loads(text) if text.strip().startswith(("{", "[")) else None
        if not parsed:
            return None
        items = parsed if isinstance(parsed, list) else parsed.get("datasources", parsed.get("value", []))
        if not items or not isinstance(items, list):
            return None
        first = items[0]
        return first.get("id") or first.get("luid") if isinstance(first, dict) else None
    except Exception as exc:
        logger.warning("Could not resolve datasource LUID for '%s': %s", preferred_name, exc)
        return None


async def tableau_mcp_chat(
    query: str,
    conversation_history: list[dict] | None = None,
    preferred_datasource_name: Optional[str] = None,
    dashboard_has_pulse_objects: bool = False,
    tableau_viewer_id: Optional[str] = None,
    viewer_email: Optional[str] = None,
) -> dict[str, Any]:
    """
    Process a chat query using Tableau MCP tools via langchain-mcp-adapters.

    Args:
        query: The user's message.
        conversation_history: Prior turns as [{"role": "user"|"assistant", "content": str}].
        preferred_datasource_name: Datasource name from the dashboard extension context.
        dashboard_has_pulse_objects: Whether the dashboard contains Pulse metric cards.
        tableau_viewer_id: The viewer's identity string from the workbook/extension.
        viewer_email: Viewer email forwarded as X-Tableau-Jwt-Username to the MCP server.
    """
    if conversation_history is None:
        conversation_history = []

    headers: dict[str, str] = {}
    if viewer_email and is_likely_email(viewer_email):
        headers["X-Tableau-Jwt-Username"] = viewer_email

    mcp_client = MultiServerMCPClient(
        {
            "tableau": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
                "headers": headers,
            }
        }
    )
    tools = await mcp_client.get_tools()
    logger.info("Found %d MCP tools: %s", len(tools), [t.name for t in tools])
    print(f"Found {len(tools)} MCP tools: {[t.name for t in tools]}")

    resolved_luid: Optional[str] = None
    if preferred_datasource_name:
        resolved_luid = await _resolve_datasource_luid(tools, preferred_datasource_name)
        if resolved_luid:
            logger.info("Resolved '%s' -> LUID %s", preferred_datasource_name, resolved_luid)
            print(f"📌 Resolved dashboard datasource '{preferred_datasource_name}' -> LUID {resolved_luid}")
        else:
            print(f"⚠️ Could not resolve LUID for '{preferred_datasource_name}'; agent will discover datasources")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
        max_tokens=1000,
        timeout=120,
    )
    llm_with_tools = llm.bind_tools(tools)

    system_content = _build_system_message(
        tools=tools,
        tableau_viewer_id=tableau_viewer_id,
        preferred_datasource_name=preferred_datasource_name,
        resolved_luid=resolved_luid,
        dashboard_has_pulse_objects=dashboard_has_pulse_objects,
    )

    messages: list = [SystemMessage(content=system_content)]
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=query))

    tool_results: list[dict] = []
    response: AIMessage | None = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.debug("Agent iteration %d/%d", iteration, MAX_ITERATIONS)
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]
            logger.info("Calling tool %s with args: %s", name, args)
            print(f"🔧 Executing {name} with args: {args}")

            tool = next((t for t in tools if t.name == name), None)
            if tool is None:
                content = f"Tool '{name}' not found."
            else:
                try:
                    content = await tool.ainvoke(args)
                    if not isinstance(content, str):
                        content = json.dumps(content)
                    tool_results.append({"tool": name, "arguments": args, "result": content})
                    print(f"✅ {name} completed")
                except Exception as exc:
                    content = f"Error: {exc}"
                    tool_results.append({"tool": name, "arguments": args, "error": str(exc)})
                    logger.warning("Tool %s failed: %s", name, exc)
                    print(f"❌ {name} failed: {exc}")

            messages.append(ToolMessage(content=content, tool_call_id=call_id))

    final_text = (response.content or "").strip() if response else ""
    if not final_text:
        final_text = (
            f"I reached the iteration limit ({MAX_ITERATIONS} steps). "
            "Try rephrasing or asking a more specific question."
        )

    return {
        "response": final_text,
        "tool_results": tool_results,
        "iterations": iteration,
        "logged_in_as": tableau_viewer_id,
        "tableau_viewer_id": tableau_viewer_id,
    }


# Backward-compatibility alias
async def langchain_mcp_chat(query: str) -> dict[str, Any]:
    return await tableau_mcp_chat(query)
