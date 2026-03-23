"""
Clean HTTP MCP + LangChain integration for Tableau
Following the exact pattern from the Tableau e-bikes demo
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# LangChain Libraries  
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# MCP HTTP client
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utilities.viewer_email import is_likely_email

# Configuration (required: set MCP_SERVER_URL in .env or environment)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
if not MCP_SERVER_URL:
    raise ValueError("MCP_SERVER_URL environment variable is required")
MAX_MCP_ITERATIONS = 10  # Reduced for efficiency
MCP_PROTOCOL_VERSION = "2025-06-18"

# Learning and Memory System
class LearningMemory:
    """Intelligent memory system for learning from successes and failures"""
    
    def __init__(self):
        self.successful_patterns = {}
        self.failed_patterns = {}
        self.field_mappings = {}
        self.datasource_preferences = {}
        self.error_recovery_strategies = {}
    
    def learn_success(self, question_type: str, approach: dict, result: dict):
        """Remember successful approaches for future use"""
        if question_type not in self.successful_patterns:
            self.successful_patterns[question_type] = []
        self.successful_patterns[question_type].append({
            "approach": approach,
            "result_summary": result.get("summary", "success"),
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def learn_failure(self, question_type: str, approach: dict, error: str):
        """Remember failed approaches to avoid repeating them"""
        if question_type not in self.failed_patterns:
            self.failed_patterns[question_type] = []
        self.failed_patterns[question_type].append({
            "approach": approach,
            "error": error,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def get_best_approach(self, question_type: str) -> dict:
        """Get the best approach based on learned patterns"""
        if question_type in self.successful_patterns:
            # Return the most recent successful approach
            return self.successful_patterns[question_type][-1]["approach"]
        return {"strategy": "explore", "datasource": "auto", "fields": "discover"}
    
    def should_avoid_approach(self, question_type: str, approach: dict) -> bool:
        """Check if we should avoid a particular approach based on past failures"""
        if question_type in self.failed_patterns:
            for failed in self.failed_patterns[question_type]:
                if self._approaches_similar(failed["approach"], approach):
                    return True
        return False
    
    def _approaches_similar(self, approach1: dict, approach2: dict) -> bool:
        """Check if two approaches are similar enough to avoid repetition"""
        return (approach1.get("datasource") == approach2.get("datasource") and
                approach1.get("strategy") == approach2.get("strategy"))
    
    def get_error_recovery_strategy(self, error: str) -> dict:
        """Get recovery strategy based on error type"""
        if "Unknown Field" in error:
            return {"strategy": "field_discovery", "action": "list_fields_first"}
        elif "Invalid arguments" in error:
            return {"strategy": "argument_fix", "action": "simplify_arguments"}
        elif "datasource" in error.lower():
            return {"strategy": "datasource_switch", "action": "try_alternative_datasource"}
        else:
            return {"strategy": "general_recovery", "action": "simplify_and_retry"}

# Global learning memory instance
learning_memory = LearningMemory()

# Helper functions for intelligent error recovery
def _identify_question_type(query: str) -> str:
    """Identify the type of question to help with learning and recovery"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["medication", "pharmacy", "drug", "expir", "inventory"]):
        return "pharmacy"
    elif any(word in query_lower for word in ["sales", "profit", "customer", "product", "revenue", "best", "top", "sellers"]):
        return "sales"
    elif any(word in query_lower for word in ["list", "show", "what", "available"]):
        return "discovery"
    else:
        return "general"

async def _try_alternative_approach(tool_name: str, tool_args: dict, error: str, 
                                  recovery_strategy: dict, mcp_client, langchain_tools, query: str):
    """Try alternative approaches based on error type and recovery strategy"""
    
    try:
        if recovery_strategy["strategy"] == "field_discovery":
            # Try listing fields first to understand structure
            if "datasourceLuid" in tool_args:
                print("Trying field discovery approach...")
                for tool in langchain_tools:
                    if tool.name == "list-fields":
                        result = await tool._arun(datasourceLuid=tool_args["datasourceLuid"])
                        return {
                            "tool": "list-fields",
                            "arguments": {"datasourceLuid": tool_args["datasourceLuid"]},
                            "result": json.loads(result) if result.startswith('[') or result.startswith('{') else result
                        }
        
        elif recovery_strategy["strategy"] == "argument_fix":
            # Try simplifying arguments
            print("Trying simplified arguments...")
            simplified_args = _simplify_arguments(tool_args)
            for tool in langchain_tools:
                if tool.name == tool_name:
                    result = await tool._arun(**simplified_args)
                    return {
                        "tool": tool_name,
                        "arguments": simplified_args,
                        "result": json.loads(result) if result.startswith('[') or result.startswith('{') else result
                    }
        
        elif recovery_strategy["strategy"] == "datasource_switch":
            # Try alternative datasource
            print("Trying alternative datasource...")
            alternative_datasource = _find_alternative_datasource(tool_args, query)
            if alternative_datasource:
                modified_args = tool_args.copy()
                modified_args["datasourceLuid"] = alternative_datasource
                for tool in langchain_tools:
                    if tool.name == tool_name:
                        result = await tool._arun(**modified_args)
                        return {
                            "tool": tool_name,
                            "arguments": modified_args,
                            "result": json.loads(result) if result.startswith('[') or result.startswith('{') else result
                        }
        
        elif recovery_strategy["strategy"] == "general_recovery":
            # Try a completely different approach
            print("Trying general recovery approach...")
            return await _try_general_recovery(query, mcp_client, langchain_tools)
    
    except Exception as recovery_error:
        print(f"Recovery attempt failed: {str(recovery_error)}")
    
    return None

def _simplify_arguments(tool_args: dict) -> dict:
    """Simplify tool arguments to avoid complex formatting issues"""
    simplified = tool_args.copy()
    
    if "query" in simplified and isinstance(simplified["query"], dict):
        # Simplify query structure
        query = simplified["query"]
        if "fields" in query and len(query["fields"]) > 2:
            # Reduce to just 2 most important fields
            query["fields"] = query["fields"][:2]
        if "filters" in query and len(query["filters"]) > 1:
            # Remove complex filters
            query["filters"] = []
    
    return simplified

def _find_alternative_datasource(tool_args: dict, query: str) -> str:
    """Find alternative datasource based on query context"""
    query_lower = query.lower()
    
    # For pharmacy questions, try different pharmacy datasources
    if any(word in query_lower for word in ["medication", "pharmacy", "drug"]):
        pharmacy_datasources = [
            "50814edc-e918-4d9d-9248-9e6fb9d45d76",  # Pharmacist Inventory (Enriched)
            "96601c25-eca8-4072-90e3-27f25f667638",  # Pharmacist Inventory
            "462cc170-c1c8-4f0f-9dcc-1eac6ebdc9ef"   # pharmacy_dashboard_data_john
        ]
        for ds in pharmacy_datasources:
            if ds != tool_args.get("datasourceLuid"):
                return ds
    
    # For sales questions, try different sales datasources
    elif any(word in query_lower for word in ["sales", "profit", "customer"]):
        sales_datasources = [
            "d8c8b547-19a9-4850-9b3e-83afdcc691c5",  # Superstore Datasource (Samples)
            "e7156c17-345f-4d92-a315-f6abba2aec14",  # Superstore Datasource (default)
            "9d3d16b9-3c50-40c3-bfe2-2b80b4db641e"   # Databricks-Superstore
        ]
        for ds in sales_datasources:
            if ds != tool_args.get("datasourceLuid"):
                return ds
    
    return None

async def _try_general_recovery(query: str, mcp_client, langchain_tools):
    """Try a general recovery approach by exploring available data"""
    try:
        # Start with listing datasources to understand what's available
        for tool in langchain_tools:
            if tool.name == "list-datasources":
                result = await tool._arun()
                return {
                    "tool": "list-datasources",
                    "arguments": {},
                    "result": json.loads(result) if result.startswith('[') or result.startswith('{') else result
                }
    except Exception as e:
        print(f"General recovery failed: {str(e)}")
    
    return None

def _get_contextual_guidance(query: str, question_type: str, best_approach: dict) -> str:
    """Provide minimal contextual guidance - let the agent be flexible"""
    # Keep it simple - don't over-prescribe behavior
    return ""

# Intelligent system prompt with structured tool usage patterns
TABLEAU_SYSTEM_PROMPT = """You are an intelligent data analyst with access to Tableau data through MCP tools. Help users understand their data by using tools strategically and presenting insights.

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

class MCPHttpClient:
    """HTTP-based MCP client similar to the e-bikes demo"""

    def __init__(self, server_url: str, tableau_jwt_username: Optional[str] = None):
        self.server_url = server_url
        self.session = None
        email = (tableau_jwt_username or "").strip() if tableau_jwt_username else ""
        self._tableau_jwt_username = email if is_likely_email(email) else None

    def _mcp_request_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if self._tableau_jwt_username:
            headers["X-Tableau-Jwt-Username"] = self._tableau_jwt_username
            # headers["X-Tableau-Jwt-Username"] = 'slopez@superstore.com'

        return headers

    async def __aenter__(self):
        self.session = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    def _parse_sse_response(self, response_text: str):
        """Parse Server-Sent Events response to extract JSON-RPC result"""
        lines = response_text.strip().split('\n')
        
        # Look for the final result message (not notifications)
        for i, line in enumerate(lines):
            if line.startswith('event: message'):
                # Get the next data line
                if i + 1 < len(lines) and lines[i + 1].startswith('data: '):
                    data_line = lines[i + 1][6:]  # Remove 'data: ' prefix
                    try:
                        message = json.loads(data_line)
                        # Check for errors first
                        if message.get("id") == 1 and "error" in message:
                            return message
                        # Look for the actual result (not notifications)
                        if message.get("id") == 1 and "result" in message:
                            return message
                    except json.JSONDecodeError:
                        continue
        
        # If no proper result found, try parsing as direct JSON
        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            # Try to extract error from SSE format
            for line in lines:
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if "error" in data:
                            return data
                    except:
                        pass
            raise Exception(f"Could not parse MCP response: {response_text[:200]}...")
    
    async def list_tools(self):
        """Get available tools from MCP server"""
        request_data = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1
        }
        
        response = await self.session.post(
            self.server_url,
            json=request_data,
            headers=self._mcp_request_headers(),
        )
        response.raise_for_status()
        
        # Parse Server-Sent Events response
        result = self._parse_sse_response(response.text)
        
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")
            
        return result["result"]["tools"]
    
    async def call_tool(self, name: str, arguments: dict):
        """Execute a tool on the MCP server"""
        request_data = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            },
            "id": 1
        }
        
        response = await self.session.post(
            self.server_url,
            json=request_data,
            headers=self._mcp_request_headers(),
        )
        response.raise_for_status()
        
        # Parse Server-Sent Events response
        result = self._parse_sse_response(response.text)
        
        if "error" in result:
            error_info = result["error"]
            error_message = error_info.get("message", str(error_info))
            error_code = error_info.get("code", "unknown")
            # Extract full error details
            full_error = f"MCP Tool Error ({error_code}): {error_message}"
            raise Exception(full_error)
            
        return result["result"]["content"]

class MCPTool(BaseTool):
    """LangChain tool wrapper for MCP tools"""
    
    mcp_client: MCPHttpClient = Field(exclude=True)
    tool_name: str
    tool_description: str
    tool_schema: dict
    
    def __init__(self, mcp_client: MCPHttpClient, tool_name: str, tool_description: str, tool_schema: dict):
        super().__init__(
            name=tool_name,
            description=tool_description,
            mcp_client=mcp_client,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_schema=tool_schema
        )
    
    def _run(self, **kwargs) -> str:
        """Sync wrapper - not implemented"""
        raise NotImplementedError("Use async version")
    
    def _format_insight_bundle(self, bundle_data: dict) -> str:
        """Extract and format insights from a pulse metric insight bundle"""
        try:
            result = bundle_data.get("bundle_response", {}).get("result", {})
            insight_groups = result.get("insight_groups", [])
            
            if not insight_groups:
                return f"SUCCESS: {self.tool_name} returned data but no insights found:\n{json.dumps(bundle_data, indent=2)}"
            
            formatted_insights = []
            for group in insight_groups:
                insights = group.get("insights", [])
                for insight in insights:
                    insight_result = insight.get("result", {})
                    facts = insight_result.get("facts", {})
                    markup = insight_result.get("markup", "")
                    question = insight_result.get("question", "")
                    insight_type = insight.get("insight_type", "")
                    
                    # Extract key values
                    target_value = facts.get("target_period_value", {})
                    comparison_value = facts.get("comparison_period_value", {})
                    difference = facts.get("difference", {})
                    target_period = facts.get("target_time_period", {})
                    comparison_period = facts.get("comparison_time_period", {})
                    
                    # Format the insight
                    insight_text = f"INSIGHT ({insight_type}):\n"
                    if question:
                        insight_text += f"Question: {question}\n"
                    if markup:
                        insight_text += f"Summary: {markup}\n"
                    
                    # Add actual values
                    if target_period.get("label"):
                        insight_text += f"Current Period ({target_period.get('label')}): "
                        if isinstance(target_value, dict) and not target_value.get("is_nan"):
                            insight_text += f"{target_value.get('formatted', target_value.get('raw', 'N/A'))}\n"
                        else:
                            insight_text += "No data available\n"
                    
                    if comparison_period.get("label"):
                        insight_text += f"Comparison Period ({comparison_period.get('label')}): "
                        if isinstance(comparison_value, dict) and not comparison_value.get("is_nan"):
                            insight_text += f"{comparison_value.get('formatted', comparison_value.get('raw', 'N/A'))}\n"
                        else:
                            insight_text += "No data available\n"
                    
                    # Add difference if available
                    if isinstance(difference, dict):
                        abs_diff = difference.get("absolute", {})
                        rel_diff = difference.get("relative", {})
                        direction = difference.get("direction", "")
                        
                        if not abs_diff.get("is_nan") and abs_diff.get("raw") is not None:
                            insight_text += f"Absolute Change: {abs_diff.get('formatted', abs_diff.get('raw'))}\n"
                        if not rel_diff.get("is_nan") and rel_diff.get("raw") is not None:
                            insight_text += f"Relative Change: {rel_diff.get('formatted', rel_diff.get('raw'))}\n"
                        if direction:
                            insight_text += f"Direction: {direction}\n"
                    
                    formatted_insights.append(insight_text)
            
            if formatted_insights:
                return f"SUCCESS: {self.tool_name} generated {len(formatted_insights)} insight(s):\n\n" + "\n---\n\n".join(formatted_insights)
            else:
                return f"SUCCESS: {self.tool_name} returned data:\n{json.dumps(bundle_data, indent=2)}"
        except Exception as e:
            # If formatting fails, return the raw data
            return f"SUCCESS: {self.tool_name} returned data:\n{json.dumps(bundle_data, indent=2)}"
    
    def _validate_and_fix_arguments(self, kwargs: dict) -> dict:
        """Validate and fix tool arguments to ensure correct format"""
        if self.tool_name == "query-datasource":
            # Fix query-datasource argument structure
            if "datasourceLuid" in kwargs and ("fields" in kwargs or "filters" in kwargs):
                # Move fields and filters into a query object
                query = {}
                if "fields" in kwargs:
                    query["fields"] = kwargs.pop("fields")
                if "filters" in kwargs:
                    query["filters"] = kwargs.pop("filters")
                kwargs["query"] = query
        elif self.tool_name == "list-fields":
            # Ensure list-fields has datasourceLuid
            if "datasourceLuid" not in kwargs:
                # Try to find a datasource ID from other arguments
                for key, value in kwargs.items():
                    if "datasource" in key.lower() and isinstance(value, str):
                        kwargs["datasourceLuid"] = value
                        break
        elif self.tool_name == "generate-pulse-metric-value-insight-bundle":
            # MCP server requires bundleRequest.bundle_request (full object). We do not build from pulseMetricIds.
            pass
        return kwargs

    def _has_only_pulse_metric_ids(self, kwargs: dict) -> bool:
        """True if generate-pulse-metric-value-insight-bundle was called with only pulseMetricIds (no full bundle_request)."""
        if self.tool_name != "generate-pulse-metric-value-insight-bundle":
            return False
        br = kwargs.get("bundleRequest") or kwargs
        if isinstance(br, dict) and br.get("bundle_request") is not None:
            return False
        return bool(br.get("pulseMetricIds") or kwargs.get("pulseMetricIds"))

    async def _arun(self, **kwargs) -> str:
        """Execute the MCP tool with argument validation and retry logic"""
        # Validate and fix arguments
        kwargs = self._validate_and_fix_arguments(kwargs)
        
        # Avoid calling generate-pulse-metric-value-insight-bundle with only pulseMetricIds—server rejects it
        if self._has_only_pulse_metric_ids(kwargs):
            return (
                "generate-pulse-metric-value-insight-bundle requires a full bundle_request object (version, options, input.metadata, input.metric with definition/specification), not just pulseMetricIds. "
                "For 'insights from this data' or dashboard questions, use get-datasource-metadata and query-datasource with the provided datasourceLuid instead. "
                "For Pulse metrics, summarize the results from list-pulse-metrics-from-metric-definition-id or use generate-pulse-insight-brief if the question fits that tool."
            )
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = await self.mcp_client.call_tool(self.tool_name, kwargs)
                
                # MCP returns content in format: [{"type": "text", "text": "..."}]
                # Extract the actual data from the content structure
                if isinstance(result, list) and len(result) > 0:
                    # Check if it's the MCP content format
                    if isinstance(result[0], dict) and "type" in result[0] and "text" in result[0]:
                        # Extract the text content which might be JSON
                        text_content = result[0]["text"]
                        # Try to parse it as JSON to get the actual data
                        try:
                            parsed_data = json.loads(text_content)
                            
                            # Special handling for insight bundles - extract and format insights
                            if self.tool_name == "generate-pulse-metric-value-insight-bundle" and isinstance(parsed_data, dict):
                                return self._format_insight_bundle(parsed_data)
                            
                            # Return formatted JSON with clear indication of data
                            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                return f"SUCCESS: Found {len(parsed_data)} items in {self.tool_name}:\n{json.dumps(parsed_data, indent=2)}"
                            elif isinstance(parsed_data, dict):
                                return f"SUCCESS: {self.tool_name} returned data:\n{json.dumps(parsed_data, indent=2)}"
                            else:
                                return text_content
                        except json.JSONDecodeError:
                            # If it's not JSON, return the text as-is
                            return text_content
                    else:
                        # It's already a list of data objects
                        return json.dumps(result, indent=2)
                elif isinstance(result, dict):
                    return json.dumps(result, indent=2)
                else:
                    return json.dumps(result, indent=2)
            except Exception as e:
                error_msg = str(e)
                
                # If it's an argument error and we haven't tried fixing yet, try to fix
                if ("Invalid arguments" in error_msg or "-32602" in error_msg) and attempt < max_retries:
                    print(f"WARNING: Tool {self.tool_name} failed with argument error, attempting to fix...")
                    print(f"   Error: {error_msg}")
                    print(f"   Current kwargs: {kwargs}")
                    # Try different argument structures
                    if self.tool_name == "query-datasource":
                        # Try alternative structure
                        if "query" not in kwargs and ("fields" in kwargs or "filters" in kwargs):
                            kwargs = {"datasourceLuid": kwargs.get("datasourceLuid"), "query": kwargs}
                        elif "datasourceLuid" not in kwargs and "query" in kwargs:
                            # Extract datasourceLuid from query if it exists there
                            query = kwargs["query"]
                            if "datasourceLuid" in query:
                                kwargs["datasourceLuid"] = query.pop("datasourceLuid")
                    elif self.tool_name == "generate-pulse-metric-value-insight-bundle":
                        # Server requires full bundle_request; we cannot build it from IDs. Do not retry.
                        pass
                    continue
                else:
                    # Return full error message so agent can see what went wrong
                    return f"Error executing {self.tool_name}: {error_msg}"
        
        return f"Error executing {self.tool_name}: Max retries exceeded"

async def create_mcp_tools() -> List[MCPTool]:
    """Create LangChain tools from MCP server"""
    async with MCPHttpClient(MCP_SERVER_URL) as client:
        # Get available tools
        tools_data = await client.list_tools()
        
        # Create LangChain tool wrappers
        tools = []
        for tool_data in tools_data:
            tool = MCPTool(
                mcp_client=client,
                tool_name=tool_data["name"],
                tool_description=tool_data["description"],
                tool_schema=tool_data.get("inputSchema", {})
            )
            tools.append(tool)
        
        return tools

async def _resolve_datasource_luid_by_name(mcp_client: "MCPHttpClient", preferred_name: str) -> Optional[str]:
    """Call list-datasources with name filter and return the published datasource LUID (id) for the first match."""
    try:
        content = await mcp_client.call_tool("list-datasources", {"filter": f"name:eq:{preferred_name}"})
        if not content:
            return None
        # MCP content can be list of parts [{"type":"text","text":"..."}] or a single string
        text = None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    break
        elif isinstance(content, str):
            text = content
        if not text:
            return None
        parsed = json.loads(text) if isinstance(text, str) and (text.strip().startswith("[") or text.strip().startswith("{")) else None
        if not parsed:
            return None
        items = parsed if isinstance(parsed, list) else parsed.get("datasources", parsed.get("value", []))
        if not items or not isinstance(items, list):
            return None
        first = items[0]
        return first.get("id") or first.get("luid") if isinstance(first, dict) else None
    except Exception as e:
        print(f"⚠️ Could not resolve datasource LUID for name '{preferred_name}': {e}")
        return None


async def tableau_mcp_chat(
    query: str,
    conversation_history: List[Dict] = None,
    preferred_datasource_name: Optional[str] = None,
    dashboard_has_pulse_objects: bool = False,
    tableau_viewer_id: Optional[str] = None,
    viewer_email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a chat query using MCP tools and LangChain agent.
    If preferred_datasource_name is set (from extension session), resolve it to published LUID and inject so the agent uses it.
    If dashboard_has_pulse_objects is True, the agent will explain why numbers may differ from Pulse cards on the dashboard.
    If tableau_viewer_id is set (from extension workbook parameters / sheet context), inject viewer identity for personalization.
    viewer_email: when it passes is_likely_email, sent to the MCP server as X-Tableau-Jwt-Username (same as pick_viewer_email on /datasources).
    """
    if conversation_history is None:
        conversation_history = []

    async with MCPHttpClient(MCP_SERVER_URL, tableau_jwt_username=viewer_email) as mcp_client:
        # Get available tools
        tools_data = await mcp_client.list_tools()
        print(f"Found {len(tools_data)} MCP tools: {[t['name'] for t in tools_data]}")
        
        # Resolve extension datasource name -> published LUID so agent uses correct datasource
        resolved_luid = None
        if preferred_datasource_name:
            resolved_luid = await _resolve_datasource_luid_by_name(mcp_client, preferred_datasource_name)
            if resolved_luid:
                print(f"📌 Resolved dashboard datasource '{preferred_datasource_name}' -> LUID {resolved_luid}")
            else:
                print(f"⚠️ Could not resolve LUID for '{preferred_datasource_name}'; agent will discover datasources")
        
        # Create LangChain tools from MCP tools
        langchain_tools = []
        for tool_data in tools_data:
            tool = MCPTool(
                mcp_client=mcp_client,
                tool_name=tool_data["name"],
                tool_description=tool_data["description"],
                tool_schema=tool_data.get("inputSchema", {})
            )
            langchain_tools.append(tool)
        
        # Initialize OpenAI with tools and rate limiting considerations
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=1000,  # Reduce token usage
            request_timeout=120  # Extended for cold start / slow networks on deploy (was 30)
        )
        
        # Bind tools to the model
        llm_with_tools = llm.bind_tools(langchain_tools)
        
        # Create system message - keep it simple and flexible
        system_content = f"""{TABLEAU_SYSTEM_PROMPT}

Available tools:
{chr(10).join([f"- {tool['name']}: {tool['description']}" for tool in tools_data])}"""
        
        if tableau_viewer_id:
            system_content += f"""

**Viewer identity (from extension / workbook):** `{tableau_viewer_id}`. If it contains @, treat it as the viewer's email when appropriate. Use for personalization only when appropriate."""

        if resolved_luid and preferred_datasource_name:
            system_content += f"""

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

**Pulse Metric cards on the dashboard:** The dashboard may contain Pulse Metric objects. Those cards show values from specific metric definitions (fixed measure, time period, filters). Your answers use query-datasource on the same datasource with flexible queries, so **the numbers you return will not match the Pulse cards.** You MUST add one short sentence to every insight response, e.g.: "These numbers are from the same datasource with flexible queries; if your dashboard has Pulse Metric cards, their values use specific metric definitions and may differ." If the user says the numbers don't match, explain that Pulse cards are predefined metrics and this chat uses ad-hoc queries; for Pulse-based summaries they can ask "List my Pulse metrics" or "Summarize my Pulse metrics."""
        
        # Prepare conversation
        messages = [SystemMessage(content=system_content)]
        
        # Add conversation history
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        messages.append(HumanMessage(content=query))
        
        # Get response with tool calling capability
        all_tool_results = []
        iteration = 0
        consecutive_failed_tools = 0
        max_failed_attempts = 3

        def _with_viewer_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
            payload["logged_in_as"] = tableau_viewer_id
            payload["tableau_viewer_id"] = tableau_viewer_id
            return payload
        
        while iteration < MAX_MCP_ITERATIONS and consecutive_failed_tools < max_failed_attempts:
            iteration += 1
            print(f"Iteration {iteration}/{MAX_MCP_ITERATIONS}")
            
            # Get response from LLM with tools
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 Processing {len(response.tool_calls)} tool calls")
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    try:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        
                        print(f"🔧 Executing {tool_name} with args: {tool_args}")
                        
                        # Find the corresponding tool and execute it
                        for tool in langchain_tools:
                            if tool.name == tool_name:
                                result = await tool._arun(**tool_args)
                                
                                # Parse result for storage
                                try:
                                    parsed_result = json.loads(result) if result.startswith('[') or result.startswith('{') else result
                                except:
                                    parsed_result = result
                                
                                tool_result = {
                                    "tool": tool_name,
                                    "arguments": tool_args,
                                    "result": parsed_result
                                }
                                all_tool_results.append(tool_result)
                                
                                # The result from _arun is already formatted with "SUCCESS:" prefix if it has data
                                # Just use it as-is - it's already clear
                                formatted_result = result
                                
                                # Add tool result to conversation with formatted content
                                messages.append(ToolMessage(
                                    content=formatted_result,
                                    tool_call_id=tool_call["id"]
                                ))
                                
                                print(f"✅ {tool_name} completed successfully")
                                
                                # Learn from success
                                question_type = _identify_question_type(query)
                                approach = {
                                    "tool": tool_name,
                                    "datasource": tool_args.get("datasourceLuid", "unknown"),
                                    "strategy": "direct_query"
                                }
                                learning_memory.learn_success(question_type, approach, {"summary": "success"})
                                
                                consecutive_failed_tools = 0  # Reset on success
                                break
                        
                    except Exception as e:
                        print(f"❌ {tool_name} failed: {str(e)}")
                        
                        # Learn from the failure
                        question_type = _identify_question_type(query)
                        approach = {
                            "tool": tool_name,
                            "datasource": tool_args.get("datasourceLuid", "unknown"),
                            "strategy": "direct_query"
                        }
                        learning_memory.learn_failure(question_type, approach, str(e))
                        
                        # Get intelligent recovery strategy
                        recovery_strategy = learning_memory.get_error_recovery_strategy(str(e))
                        print(f"🧠 Recovery strategy: {recovery_strategy}")
                        
                        # Try alternative approach based on error type
                        alternative_result = await _try_alternative_approach(
                            tool_name, tool_args, str(e), recovery_strategy, 
                            mcp_client, langchain_tools, query
                        )
                        
                        if alternative_result:
                            print(f"✅ Alternative approach succeeded")
                            all_tool_results.append(alternative_result)
                            messages.append(ToolMessage(
                                content=json.dumps(alternative_result["result"]),
                                tool_call_id=tool_call["id"]
                            ))
                            consecutive_failed_tools = 0  # Reset on success
                        else:
                            consecutive_failed_tools += 1
                            error_result = {
                                "tool": tool_name,
                                "arguments": tool_args,
                                "error": str(e),
                                "recovery_attempted": True
                            }
                            all_tool_results.append(error_result)
                            
                            # Add error to conversation
                            messages.append(ToolMessage(
                                content=f"Error: {str(e)}. Tried alternative approach but failed.",
                                tool_call_id=tool_call["id"]
                            ))
            else:
                # No tool calls, we're done
                return _with_viewer_fields({
                    "response": response.content,
                    "tool_results": all_tool_results,
                    "iterations": iteration
                })
        
        # If we hit max iterations, ensure we always return a non-empty response
        final_text = (response.content or "").strip() if hasattr(response, 'content') else ""
        if not final_text:
            summary_parts = [f"I ran up against the iteration limit ({MAX_MCP_ITERATIONS} steps) while gathering insights."]
            if all_tool_results:
                succeeded = [t for t in all_tool_results if "error" not in t]
                if succeeded:
                    summary_parts.append(f" I successfully ran {len(succeeded)} tool call(s). For 'insights from this data', use the dashboard datasource with query-datasource; for Pulse, try 'list my pulse metrics' for a summary.")
                else:
                    summary_parts.append(" The tools I tried hit validation or server errors. Try rephrasing (e.g. 'list my pulse metrics' or 'what datasources do I have?').")
            else:
                summary_parts.append(" I wasn't able to complete the analysis. Try asking 'list my pulse metrics' or 'what data sources do I have?'")
            final_text = "".join(summary_parts)
        return _with_viewer_fields({
            "response": final_text,
            "tool_results": all_tool_results,
            "iterations": iteration
        })

# For backward compatibility
async def langchain_mcp_chat(query: str) -> Dict[str, Any]:
    """Wrapper for backward compatibility"""
    return await tableau_mcp_chat(query)
