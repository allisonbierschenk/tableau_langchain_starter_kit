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

# Configuration
MCP_SERVER_URL = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
MAX_MCP_ITERATIONS = 10  # Reduced for efficiency

# System prompt similar to e-bikes demo
TABLEAU_SYSTEM_PROMPT = """You are an expert Tableau data analyst that can answer diverse business questions using available tools.

CRITICAL EFFICIENCY RULES:
1. **Be Strategic**: Focus on the most relevant datasources first. For business questions, prioritize "Superstore" datasources.
2. **Be Comprehensive**: When asked for "all" data, query ALL relevant datasources (not just one). Look for datasources with similar names.
3. **Be Efficient**: Don't query every datasource. Use list-datasources to identify the best matches, then query all relevant ones.
4. **Be Practical**: For large datasets, start with COUNT queries to understand scale before detailed analysis.
5. **Be Direct**: Answer the specific question asked. Don't over-analyze unless requested.

TOOL USAGE SEQUENCE:
- **Discovery**: Use list-datasources to find relevant data
- **Schema**: Use list-fields to understand available fields 
- **Analysis**: Use query-datasource with appropriate filters and aggregations
- **Insights**: Provide clear, actionable answers

CRITICAL TOOL FORMATTING:
- **query-datasource**: ALWAYS use this exact structure:
  {
    "datasourceLuid": "datasource-id-here",
    "query": {
      "fields": [
        {"fieldCaption": "Product Name"},  // DIMENSION - no function
        {"fieldCaption": "Profit", "function": "SUM", "fieldAlias": "Total Profit"}  // MEASURE - with function
      ],
      "filters": [{"field": {"fieldCaption": "Field Name"}, "filterType": "TOP", "howMany": 10}]
    }
  }
- **Field Types**: 
  - DIMENSIONS (text, date): Use without functions: {"fieldCaption": "Product Name"}
  - MEASURES (numbers): Use with functions: {"fieldCaption": "Profit", "function": "SUM"}
- **list-fields**: Use {"datasourceLuid": "datasource-id-here"}
- **list-datasources**: Use {} (no arguments needed)

QUERY OPTIMIZATION:
- Use TOP filters for "top N" questions (top customers, products, etc.)
- Use appropriate aggregations (SUM, COUNT, AVG) for metrics
- Apply filters to reduce data volume
- Group by relevant dimensions

DIVERSE QUESTION HANDLING:
- **Data Discovery**: "What data do I have?" → list-datasources
- **Schema Questions**: "What fields are available?" → list-fields  
- **Business Analytics**: "Top customers by sales" → query with TOP filter
- **Comparisons**: "Compare regions" → query with GROUP BY
- **Trends**: "Sales over time" → query with date functions
- **What-if**: Use existing data to model scenarios
- **Insights**: Look for patterns, outliers, and opportunities

MULTI-DATASOURCE QUERIES:
- **"All my sales data"** → Query ALL datasources with "sales" OR "superstore" in the name
- **"All Superstore data"** → Query ALL datasources with "superstore" in the name (including "Databricks-Superstore")
- **"Using all available data"** → Query multiple relevant datasources
- **Cross-datasource analysis** → Query each datasource separately, then combine insights
- **Comprehensive analysis** → Don't limit to just one datasource when the question asks for "all" data

SUPERSTORE DATASOURCES:
- There are MULTIPLE Superstore datasources available:
  - "Superstore Datasource" (Samples project)
  - "Superstore Datasource" (default project) 
  - "Databricks-Superstore" (MCP Dashboard & Data project)
- When asked about "all sales data" or "all superstore data", query ALL of these datasources
- Combine results from all Superstore datasources for comprehensive analysis

Always provide specific, data-driven answers with actual numbers and insights."""

class MCPHttpClient:
    """HTTP-based MCP client similar to the e-bikes demo"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        
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
                        # Look for the actual result (not notifications)
                        if message.get("id") == 1 and "result" in message:
                            return message
                    except json.JSONDecodeError:
                        continue
        
        # If no proper result found, try parsing as direct JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
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
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
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
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        )
        response.raise_for_status()
        
        # Parse Server-Sent Events response
        result = self._parse_sse_response(response.text)
        
        if "error" in result:
            raise Exception(f"MCP Tool Error: {result['error']}")
            
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
        return kwargs

    async def _arun(self, **kwargs) -> str:
        """Execute the MCP tool with argument validation and retry logic"""
        # Validate and fix arguments
        kwargs = self._validate_and_fix_arguments(kwargs)
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = await self.mcp_client.call_tool(self.tool_name, kwargs)
                return json.dumps(result, indent=2)
            except Exception as e:
                error_msg = str(e)
                
                # If it's an argument error and we haven't tried fixing yet, try to fix
                if "Invalid arguments" in error_msg and attempt < max_retries:
                    print(f"⚠️ Tool {self.tool_name} failed with argument error, attempting to fix...")
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
                    continue
                else:
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

async def tableau_mcp_chat(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Process a chat query using MCP tools and LangChain agent
    Using proper LangChain tool integration
    """
    if conversation_history is None:
        conversation_history = []
    
    async with MCPHttpClient(MCP_SERVER_URL) as mcp_client:
        # Get available tools
        tools_data = await mcp_client.list_tools()
        print(f"📊 Found {len(tools_data)} MCP tools: {[t['name'] for t in tools_data]}")
        
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
            request_timeout=30  # Faster timeout
        )
        
        # Bind tools to the model
        llm_with_tools = llm.bind_tools(langchain_tools)
        
        # Create system message
        system_content = f"""{TABLEAU_SYSTEM_PROMPT}

Available tools:
{chr(10).join([f"- {tool['name']}: {tool['description']}" for tool in tools_data])}"""
        
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
        
        while iteration < MAX_MCP_ITERATIONS and consecutive_failed_tools < max_failed_attempts:
            iteration += 1
            print(f"🔄 Iteration {iteration}/{MAX_MCP_ITERATIONS}")
            
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
                                
                                tool_result = {
                                    "tool": tool_name,
                                    "arguments": tool_args,
                                    "result": json.loads(result) if result.startswith('[') or result.startswith('{') else result
                                }
                                all_tool_results.append(tool_result)
                                
                                # Add tool result to conversation
                                messages.append(ToolMessage(
                                    content=result,
                                    tool_call_id=tool_call["id"]
                                ))
                                
                                print(f"✅ {tool_name} completed successfully")
                                consecutive_failed_tools = 0  # Reset on success
                                break
                        
                    except Exception as e:
                        print(f"❌ {tool_name} failed: {str(e)}")
                        consecutive_failed_tools += 1
                        error_result = {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "error": str(e)
                        }
                        all_tool_results.append(error_result)
                        
                        # Add error to conversation
                        messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"]
                        ))
            else:
                # No tool calls, we're done
                return {
                    "response": response.content,
                    "tool_results": all_tool_results,
                    "iterations": iteration
                }
        
        # If we hit max iterations, return what we have
        return {
            "response": response.content if hasattr(response, 'content') else "Max iterations reached",
            "tool_results": all_tool_results,
            "iterations": iteration
        }

# For backward compatibility
async def langchain_mcp_chat(query: str) -> Dict[str, Any]:
    """Wrapper for backward compatibility"""
    return await tableau_mcp_chat(query)
