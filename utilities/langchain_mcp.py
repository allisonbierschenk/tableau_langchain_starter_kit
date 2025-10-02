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
                print("🔍 Trying field discovery approach...")
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
            print("🔧 Trying simplified arguments...")
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
            print("🔄 Trying alternative datasource...")
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
            print("🔄 Trying general recovery approach...")
            return await _try_general_recovery(query, mcp_client, langchain_tools)
    
    except Exception as recovery_error:
        print(f"❌ Recovery attempt failed: {str(recovery_error)}")
    
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
        print(f"❌ General recovery failed: {str(e)}")
    
    return None

def _get_contextual_guidance(query: str, question_type: str, best_approach: dict) -> str:
    """Provide contextual guidance based on question type and learned patterns"""
    
    guidance = []
    
    if question_type == "pharmacy":
        guidance.extend([
            "🏥 PHARMACY DOMAIN: This question involves medication/pharmacy data",
            "📊 RECOMMENDED DATASOURCES: Look for 'Pharmacist Inventory' or 'pharmacy' datasources",
            "🔍 KEY FIELDS: Drug Name, Expiration Date, Daily Quantity Remaining, Location",
            "💡 SPECIAL FEATURES: Use 'Expiration Buckets' field for expiration queries"
        ])
    elif question_type == "sales":
        guidance.extend([
            "💰 SALES DOMAIN: This question involves business/sales data",
            "📊 CRITICAL: Query ALL sales datasources for comprehensive analysis",
            "🔍 SALES DATASOURCES: Superstore Datasource (Samples), Superstore Datasource (default), Databricks-Superstore",
            "🔍 KEY FIELDS: Customer Name, Product Name, Sales, Profit, Region",
            "💡 ANALYSIS TIPS: Use TOP filters for 'best' questions, combine results from all datasources",
            "⚠️ IMPORTANT: For 'best sellers' or 'all sales' questions, you MUST query multiple datasources"
        ])
    elif question_type == "discovery":
        guidance.extend([
            "🔍 DISCOVERY MODE: This question is about exploring available data",
            "📊 START WITH: list-datasources to see what's available",
            "🔍 THEN EXPLORE: list-fields to understand data structure",
            "💡 STRATEGY: Start broad, then narrow down based on findings"
        ])
    else:
        guidance.extend([
            "🤔 GENERAL ANALYSIS: This question requires general data exploration",
            "📊 APPROACH: Start by discovering available datasources",
            "🔍 STRATEGY: Explore data structure, then choose appropriate analysis",
            "💡 TIP: Look for datasources that match the question context"
        ])
    
    # Add learned insights
    if best_approach.get("strategy") != "explore":
        guidance.append(f"🧠 LEARNED PATTERN: Previous successful approach was {best_approach.get('strategy')}")
    
    return "\n".join(guidance)

# Intelligent system prompt that enables reasoning and adaptation
TABLEAU_SYSTEM_PROMPT = """You are an intelligent data analyst with access to multiple datasources through MCP tools. You excel at reasoning, adapting, and learning from experience.

CORE INTELLIGENCE:
- **Dynamic Discovery**: Explore available datasources and understand their structure through reasoning
- **Context-Aware Analysis**: Choose the best approach based on question context and business domain
- **Adaptive Problem Solving**: Try different strategies when approaches fail, learn from errors
- **Business Intelligence**: Understand pharmacy, sales, and other business domains naturally
- **Continuous Learning**: Remember what works and adapt your approach over time

REASONING APPROACH:
When given a question, think step by step:
1. **Analyze the Question**: What domain is this? What data would be most relevant?
2. **Explore Available Data**: What datasources exist? What's their structure?
3. **Choose Strategy**: What approach would be most effective for this specific question?
4. **Execute and Learn**: Try your approach, learn from results, adapt if needed
5. **Provide Insights**: Give specific, data-driven answers with actual numbers

BUSINESS DOMAIN KNOWLEDGE:
- **Pharmacy/Medical**: Medications, expiration dates, inventory, patients, prescriptions
- **Sales/Business**: Customers, products, revenue, profit, regions, trends
- **Data Patterns**: Look for relevant fields, understand relationships, optimize queries

TOOL USAGE PHILOSOPHY:
- **Discovery First**: Always start by understanding what data is available
- **Context-Driven**: Choose tools based on what the question actually needs
- **Adaptive Execution**: If one approach fails, try alternative strategies
- **Learning-Oriented**: Remember successful patterns and avoid failed ones

INTELLIGENT QUERY STRATEGIES:
- **"All data" questions**: Identify all relevant datasources, query each appropriately
- **"Best sellers", "top products", "all sales"**: Query ALL sales datasources and combine results
- **Domain-specific questions**: Use business knowledge to choose the best datasource
- **Comprehensive analysis**: When asked for "best" or "top" items, query multiple datasources
- **Complex analysis**: Break down into steps, use multiple tools as needed
- **Error recovery**: When something fails, try alternative approaches

CRITICAL TOOL FORMATTING:
- **query-datasource**: Use this structure:
  {
    "datasourceLuid": "datasource-id-here",
    "query": {
      "fields": [
        {"fieldCaption": "Dimension Field"},  // DIMENSION - no function
        {"fieldCaption": "Measure Field", "function": "SUM", "fieldAlias": "Alias"}  // MEASURE - with function
      ],
      "filters": [{"field": {"fieldCaption": "Field Name"}, "filterType": "TOP", "howMany": 10}]
    }
  }
- **Field Types**: 
  - DIMENSIONS (text, date): Use without functions: {"fieldCaption": "Product Name"}
  - MEASURES (numbers): Use with functions: {"fieldCaption": "Profit", "function": "SUM"}
- **list-fields**: Use {"datasourceLuid": "datasource-id-here"}
- **list-datasources**: Use {} (no arguments needed)

Always provide specific, data-driven answers with actual numbers and insights. Think, reason, adapt, and learn."""

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
        
        # Get intelligent context and recommendations
        question_type = _identify_question_type(query)
        best_approach = learning_memory.get_best_approach(question_type)
        context_info = _get_contextual_guidance(query, question_type, best_approach)
        
        # Create system message with intelligent context
        system_content = f"""{TABLEAU_SYSTEM_PROMPT}

CONTEXTUAL GUIDANCE:
{context_info}

Available tools:
{chr(10).join([f"- {tool['name']}: {tool['description']}" for tool in tools_data])}

LEARNING INSIGHTS:
- Question type: {question_type}
- Recommended approach: {best_approach.get('strategy', 'explore')}
- Previous successes: {len(learning_memory.successful_patterns.get(question_type, []))}
- Previous failures: {len(learning_memory.failed_patterns.get(question_type, []))}"""
        
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
