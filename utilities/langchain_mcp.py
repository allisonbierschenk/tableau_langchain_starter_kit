"""
Tableau MCP + LangChain integration.

Uses the official langchain-mcp-adapters package with the MCP Python SDK's
streamable HTTP transport and langgraph's create_react_agent for the agent loop.
"""

import json
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

from utilities.viewer_email import is_likely_email

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
if not MCP_SERVER_URL:
    raise ValueError("MCP_SERVER_URL environment variable is required")

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

PULSE METRICS:
- When a question involves a named business metric (MRR, ARR, churn, revenue, NPS, etc.) or the user asks about "metrics", "pulse", or "insights", use `list-all-pulse-metric-definitions` or `list-pulse-metric-subscriptions` to discover what Pulse metrics exist before deciding how to answer.
- Use `list-pulse-metrics-from-metric-definition-id` to get metric IDs and details.
- For a summary: present the list of metrics and their definitions.
- **Do NOT call `generate-pulse-metric-value-insight-bundle`** with only `pulseMetricIds`. That tool requires a **full bundle_request object**. Prefer summarizing list results or using `generate-pulse-insight-brief` instead.

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
- If a tool fails, use discovery tools first or try an alternative approach."""


def _parse_luid_from_list_datasources(text: str) -> Optional[str]:
    """Parse a published datasource LUID from a list-datasources response.

    Handles all known Tableau REST API response shapes:
      {"datasources": {"datasource": [{"id": "...", ...}]}}  -- standard nested
      [{"id": "..."}]                                         -- direct list
      {"value": [{"id": "..."}]}                             -- OData-style
    """
    text = text.strip()
    if not text.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        ds = parsed.get("datasources")
        if isinstance(ds, dict):
            items = ds.get("datasource", [])
        elif isinstance(ds, list):
            items = ds
        else:
            items = parsed.get("value", [])
    else:
        return None

    if not items or not isinstance(items, list):
        return None
    first = items[0]
    if not isinstance(first, dict):
        return None
    return first.get("id") or first.get("luid") or None


def _text_from_tool_message(result: Any) -> str:
    """Extract plain text from a ToolMessage or list of content blocks."""
    from langchain_core.messages import ToolMessage as LCToolMessage
    if isinstance(result, LCToolMessage):
        content = result.content
    else:
        content = result
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


async def _resolve_datasource_luid(
    tools: list, preferred_name: str
) -> Optional[str]:
    """Resolve a datasource name to its published LUID via list-datasources."""
    list_ds = next((t for t in tools if t.name == "list-datasources"), None)
    if list_ds is None:
        return None
    try:
        raw = await list_ds.ainvoke({"filter": f"name:eq:{preferred_name}"})
        text = _text_from_tool_message(raw)
        luid = _parse_luid_from_list_datasources(text)
        if not luid:
            logger.warning(
                "list-datasources returned no parseable LUID for '%s'. Response: %s",
                preferred_name, text[:300],
            )
        return luid
    except Exception as exc:
        logger.warning("Could not resolve datasource LUID for '%s': %s", preferred_name, exc)
        return None


def _build_system_prompt(
    tools: list,
    tableau_viewer_id: Optional[str],
    preferred_datasource_name: Optional[str],
    resolved_luid: Optional[str],
    dashboard_has_pulse_objects: bool,
) -> str:
    parts = [SYSTEM_PROMPT]
    parts.append("\nAvailable tools:\n" + "\n".join(f"- {t.name}: {t.description}" for t in tools))

    if tableau_viewer_id:
        parts.append(
            f"\n**Viewer identity:** `{tableau_viewer_id}`. "
            "If it contains @, treat it as the viewer's email when appropriate."
        )

    if resolved_luid and preferred_datasource_name:
        parts.append(f"""
**Dashboard context:** The user is viewing a dashboard connected to the datasource "{preferred_datasource_name}". For ANY of these you MUST use the datasource and return data-driven answers (never generic advice without querying):
- "top insights", "insights from this data", "analyze this dashboard"
- "What should I focus on to be proactive?", "What should I do next?", "Give me recommendations", "What are my priorities?"
Steps: (1) get-datasource-metadata with datasourceLuid below to see fields. (2) query-datasource with this LUID to get key metrics (top/bottom performers, trends, totals). (3) Answer with 3–5 concrete insights or recommendations with numbers from the query results.
Use only query-datasource and get-datasource-metadata with:
- datasourceLuid: `{resolved_luid}`

**Start every insight response with a visible scope block:**
**Measures:** [list what was aggregated, e.g. SUM(Revenue), COUNT(Orders)]
**Time frame:** [e.g. last complete quarter, current month, all time, or no time filter]
**Filters:** [e.g. none, or list any filters applied]
Then write the insights.

**Pulse Metric cards on the dashboard:** Your answers use query-datasource with flexible queries, so the numbers you return will not match Pulse cards. Add one short sentence: "These numbers are from the same datasource with flexible queries; if your dashboard has Pulse Metric cards, their values use specific metric definitions and may differ." """)

    return "\n".join(parts)


async def tableau_mcp_chat(
    query: str,
    conversation_history: list[dict] | None = None,
    preferred_datasource_name: Optional[str] = None,
    dashboard_has_pulse_objects: bool = False,
    tableau_viewer_id: Optional[str] = None,
    viewer_email: Optional[str] = None,
) -> dict[str, Any]:
    """
    Process a chat query using Tableau MCP tools.

    Uses langchain-mcp-adapters for tool loading and langgraph's create_react_agent
    for the agent loop — the standard, maintained integration pattern.
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
            print(f"📌 Resolved dashboard datasource '{preferred_datasource_name}' -> LUID {resolved_luid}")
        else:
            print(f"⚠️ Could not resolve LUID for '{preferred_datasource_name}'; agent will discover datasources")

    system_prompt = _build_system_prompt(
        tools=tools,
        tableau_viewer_id=tableau_viewer_id,
        preferred_datasource_name=preferred_datasource_name,
        resolved_luid=resolved_luid,
        dashboard_has_pulse_objects=dashboard_has_pulse_objects,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
        max_tokens=1000,
        timeout=120,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    # Build the input messages from conversation history + current query
    messages = []
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=query))

    result = await agent.ainvoke({"messages": messages})

    # Extract the final AI response and any tool calls made
    all_messages = result.get("messages", [])
    final_response = ""
    tool_results = []

    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.content:
            final_response = msg.content if isinstance(msg.content, str) else str(msg.content)
        if hasattr(msg, "name") and msg.type == "tool":
            tool_results.append({
                "tool": msg.name,
                "result": msg.content if isinstance(msg.content, str) else str(msg.content),
            })

    if not final_response:
        final_response = "I was unable to complete the analysis. Please try rephrasing your question."

    return {
        "response": final_response,
        "tool_results": tool_results,
        "iterations": len([m for m in all_messages if isinstance(m, AIMessage)]),
        "logged_in_as": tableau_viewer_id,
        "tableau_viewer_id": tableau_viewer_id,
    }


# Backward-compatibility alias
async def langchain_mcp_chat(query: str) -> dict[str, Any]:
    return await tableau_mcp_chat(query)
