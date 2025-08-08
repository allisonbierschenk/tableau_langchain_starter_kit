import os
import json
import asyncio
import aiohttp
import traceback
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Global session storage (in production, use Redis or database)
client_sessions = {}

def get_client_session_id(request: Request) -> str:
    """Get or create client session ID"""
    client_ip = request.client.host
    if client_ip not in client_sessions:
        client_sessions[client_ip] = {
            'datasource_luid': None,
            'metadata': None
        }
    return client_ip

def store_client_datasource(session_id: str, datasource_luid: str, metadata: dict):
    """Store datasource info for client session"""
    if session_id in client_sessions:
        client_sessions[session_id]['datasource_luid'] = datasource_luid
        client_sessions[session_id]['metadata'] = metadata

def get_client_datasource(session_id: str) -> tuple:
    """Get stored datasource info for client session"""
    if session_id in client_sessions:
        session = client_sessions[session_id]
        return session.get('datasource_luid'), session.get('metadata')
    return None, None

# Tableau API configuration
TABLEAU_SERVER_URL = os.getenv('TABLEAU_SERVER_URL')
TABLEAU_SITE_ID = os.getenv('TABLEAU_SITE_ID')
TABLEAU_PERSONAL_ACCESS_TOKEN_NAME = os.getenv('TABLEAU_PERSONAL_ACCESS_TOKEN_NAME')
TABLEAU_PERSONAL_ACCESS_TOKEN_SECRET = os.getenv('TABLEAU_PERSONAL_ACCESS_TOKEN_SECRET')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# MCP Configuration
MCP_SERVER_URL = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
MAX_MCP_ITERATIONS = 10

class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def list_tools(self):
        """Get available tools from MCP server"""
        async with self.session.post(
            f"{self.server_url}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 1
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to list tools: {response.status}")
            
            # Parse the streaming response
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'result' in data and 'tools' in data['result']:
                            return data['result']['tools']
                    except json.JSONDecodeError:
                        continue
            return []

    async def call_tool(self, name: str, arguments: dict):
        """Execute a tool on the MCP server"""
        async with self.session.post(
            f"{self.server_url}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments
                },
                "id": 1
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Tool execution failed: {response.status} - {error_text}")
            
            # Parse the streaming response
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'result' in data:
                            return data['result']
                    except json.JSONDecodeError:
                        continue
            return {"content": "No response received"}

async def create_openai_client():
    """Create OpenAI client"""
    import openai
    return openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

async def mcp_chat_iterative(messages: List[Dict], query: str, datasource_luid: str = None):
    """
    Iterative MCP chat implementation matching the TypeScript pattern
    """
    print(f"🚀 NEW MCP ANALYSIS STARTED")
    print(f"📋 Query: '{query}'")
    print(f"💬 Conversation history: {len(messages)} messages")
    
    try:
        # Create MCP client
        async with MCPClient(MCP_SERVER_URL) as mcp_client:
            print("📋 Getting available tools...")
            
            # Get available tools from MCP server
            tools_response = await mcp_client.list_tools()
            tools = tools_response if isinstance(tools_response, list) else []
            
            print(f"🔧 Found {len(tools)} tools: {[t.get('name', 'unknown') for t in tools]}")
            
            # Create OpenAI client
            openai_client = await create_openai_client()
            
            # Prepare OpenAI tools format
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get('name', ''),
                        "description": tool.get('description', ''),
                        "parameters": tool.get('inputSchema', {})
                    }
                })
            
            # Create system message with MCP context
            system_message = {
                "role": "system",
                "content": f"""You are a helpful assistant that can analyze Tableau data using these available tools:
{chr(10).join([f"- {tool.get('name', '')}: {tool.get('description', '')}" for tool in tools])}

CRITICAL INSTRUCTIONS:
1. When users ask questions about their data, IMMEDIATELY use the tools to get the actual data - don't just describe what you will do.
2. For data analysis questions, follow this sequence:
   - Use list-fields to understand the data structure
   - Use query-datasource to get the actual data needed to answer the question
   - Analyze the results and provide insights
3. Don't say "I will do X" - just do X immediately using the available tools.
4. Provide clear, actionable insights based on the actual data retrieved.
5. If a datasource is specified, use it. Otherwise, intelligently select the most appropriate datasource."""
            }
            
            # Prepare conversation history
            conversation_messages = [
                system_message,
                *messages,
                {"role": "user", "content": query}
            ]
            
            # Iterative tool calling with multiple rounds
            current_messages = conversation_messages.copy()
            all_tool_results = []
            final_response = ''
            iteration = 0
            last_completion = None
            
            while iteration < MAX_MCP_ITERATIONS:
                iteration += 1
                print(f"🔄 === ITERATION {iteration}/{MAX_MCP_ITERATIONS} ===")
                print(f"📝 Current conversation has {len(current_messages)} messages")
                
                # Call OpenAI with tools
                completion = await openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=current_messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )
                
                last_completion = completion
                assistant_message = completion.choices[0].message
                current_messages.append(assistant_message)
                
                print(f"🤖 OpenAI responded with {len(assistant_message.tool_calls) if assistant_message.tool_calls else 0} tool calls")
                if assistant_message.content:
                    print(f"💬 Assistant message: {assistant_message.content[:100]}...")
                
                # If no tool calls, we're done
                if not assistant_message.tool_calls or len(assistant_message.tool_calls) == 0:
                    final_response = assistant_message.content or ''
                    print(f"✅ No more tool calls needed. Final response: {final_response[:100]}...")
                    break
                
                print(f"🛠️  Executing {len(assistant_message.tool_calls)} tool call(s):")
                
                # Execute all tool calls in this iteration
                iteration_tool_results = []
                
                for tool_call in assistant_message.tool_calls:
                    try:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        print(f"🔧 Calling {tool_name} with args: {json.dumps(tool_args, indent=2)}")
                        
                        # Execute tool via MCP
                        result = await mcp_client.call_tool(tool_name, tool_args)
                        
                        tool_result = {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result
                        }
                        
                        iteration_tool_results.append(tool_result)
                        all_tool_results.append(tool_result)
                        
                        print(f"✅ {tool_name} completed successfully. Result: {json.dumps(result, indent=2)[:300]}...")
                        
                        # Add tool result to conversation
                        current_messages.append({
                            "role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": tool_call.id
                        })
                        
                    except Exception as tool_error:
                        print(f"❌ {tool_call.function.name} FAILED: {tool_error}")
                        error_result = {
                            "tool": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "error": str(tool_error)
                        }
                        
                        iteration_tool_results.append(error_result)
                        all_tool_results.append(error_result)
                        
                        # Add error to conversation
                        current_messages.append({
                            "role": "tool",
                            "content": json.dumps({"error": str(tool_error)}),
                            "tool_call_id": tool_call.id
                        })
                
                print(f"🏁 Iteration {iteration} completed:")
                print(f"   - {len(iteration_tool_results)} tool(s) executed")
                print(f"   - {len([r for r in iteration_tool_results if 'error' not in r])} successful")
                print(f"   - {len([r for r in iteration_tool_results if 'error' in r])} failed")
            
            # If we hit max iterations, make one final call for a response
            if iteration >= MAX_MCP_ITERATIONS and not final_response:
                print("Max iterations reached, getting final response")
                final_completion = await openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=current_messages
                )
                final_response = final_completion.choices[0].message.content or ''
            
            return {
                "response": final_response,
                "toolResults": all_tool_results,
                "usage": last_completion.usage.dict() if last_completion and last_completion.usage else None,
                "iterations": iteration
            }
            
    except Exception as error:
        print(f"❌ MCP Chat error: {error}")
        traceback.print_exc()
        raise Exception(f"Failed to process chat request: {str(error)}")

async def list_datasources():
    """List all available datasources"""
    try:
        # Create JWT token for Tableau authentication
        import jwt
        import time
        
        payload = {
            'iss': TABLEAU_PERSONAL_ACCESS_TOKEN_NAME,
            'exp': int(time.time()) + 3600,  # 1 hour expiration
            'aud': 'tableau',
            'sub': TABLEAU_PERSONAL_ACCESS_TOKEN_NAME,
            'scp': ['tableau:content:read', 'tableau:datasources:read']
        }
        
        token = jwt.encode(payload, TABLEAU_PERSONAL_ACCESS_TOKEN_SECRET, algorithm='HS256')
        
        # Get datasources from Tableau
        async with aiohttp.ClientSession() as session:
            headers = {
                'X-Tableau-Auth': token,
                'Content-Type': 'application/json'
            }
            
            url = f"{TABLEAU_SERVER_URL}/api/3.20/sites/{TABLEAU_SITE_ID}/datasources"
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get datasources: {response.status}")
                
                data = await response.json()
                datasources = data.get('datasources', {}).get('datasource', [])
                
                formatted_datasources = []
                for ds in datasources:
                    formatted_datasources.append({
                        'id': ds.get('id'),
                        'name': ds.get('name'),
                        'project': ds.get('project', {}).get('name', 'Unknown')
                    })
                
                response_text = f"I found {len(formatted_datasources)} datasources available to you:\n\n"
                for i, ds in enumerate(formatted_datasources, 1):
                    response_text += f"{i}. **{ds['name']}** (Project: {ds['project']})\n"
                
                return {"response": response_text, "datasources": formatted_datasources}
                
    except Exception as e:
        print(f"❌ Error listing datasources: {str(e)}")
        return {"response": f"Error retrieving datasources: {str(e)}", "datasources": []}

def select_datasource_for_query(query: str, available_datasources: list) -> dict:
    """Intelligently select the most appropriate datasource based on the query"""
    query_lower = query.lower()
    
    # Simple keyword matching - can be enhanced with more sophisticated logic
    for ds in available_datasources:
        ds_name_lower = ds['name'].lower()
        
        # Check for specific keywords in datasource names
        if 'pharmacist' in query_lower and 'pharmacist' in ds_name_lower:
            return ds
        if 'sales' in query_lower and 'sales' in ds_name_lower:
            return ds
        if 'inventory' in query_lower and 'inventory' in ds_name_lower:
            return ds
        if 'historical' in query_lower and 'historical' in ds_name_lower:
            return ds
    
    # Default to first datasource if no specific match
    return available_datasources[0] if available_datasources else None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/datasources")
async def get_datasources():
    """Get list of available datasources"""
    return await list_datasources()

@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request):
    """Enhanced chat endpoint with iterative MCP tool calling"""
    session_id = get_client_session_id(fastapi_request)
    
    print(f"\n💬 Enhanced chat request from session {session_id}")
    print(f"🧠 Message: {request.message}")
    
    try:
        # Get available datasources
        datasources_result = await list_datasources()
        available_datasources = datasources_result.get("datasources", [])
        
        # Handle datasource listing requests
        if "data sources" in request.message.lower() or "datasources" in request.message.lower() or "what data sources" in request.message.lower():
            return ChatResponse(response=datasources_result["response"])
        
        # Get or select datasource
        datasource_luid, metadata = get_client_datasource(session_id)
        
        if not datasource_luid:
            selected_datasource = select_datasource_for_query(request.message, available_datasources)
            
            if selected_datasource:
                print(f"📊 Selected datasource: {selected_datasource['name']} (ID: {selected_datasource['id']})")
                store_client_datasource(session_id, selected_datasource['id'], {
                    'name': selected_datasource['name'],
                    'project': selected_datasource['project']
                })
                datasource_luid = selected_datasource['id']
                metadata = {'name': selected_datasource['name']}
            else:
                print("⚠️ No suitable datasource found for query")
                return ChatResponse(response="I apologize, but I couldn't find a suitable data source for your query. Could you please specify which data you'd like to analyze?")
        
        print("🚀 Using iterative MCP analysis...")
        
        # Prepare conversation context
        conversation = []
        if metadata and metadata.get('name'):
            conversation.append({
                'role': 'system',
                'content': f"You are analyzing data from the {metadata.get('name')} datasource. Provide specific, data-driven answers."
            })
        
        # Use iterative MCP chat
        result = await mcp_chat_iterative(conversation, request.message, datasource_luid)
        
        response_text = result.get('response', '')
        
        if not response_text:
            response_text = "I apologize, but I was unable to process your request at this time."
        
        print(f"🎯 Final response: {response_text[:200]}...")
        return ChatResponse(response=response_text)
        
    except Exception as e:
        print("❌ Error in enhanced chat endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced analysis error: {str(e)}")

@app.post("/chat-stream")
async def chat_stream(request: ChatRequest, fastapi_request: Request):
    """Streaming chat endpoint with SSE"""
    
    async def generate():
        session_id = get_client_session_id(fastapi_request)
        
        try:
            # Send initial connection event
            yield f"event: progress\ndata: {json.dumps({'message': 'Connection established', 'step': 'init'})}\n\n"
            
            # Get available datasources
            datasources_result = await list_datasources()
            available_datasources = datasources_result.get("datasources", [])
            
            # Handle datasource listing requests
            if "data sources" in request.message.lower() or "datasources" in request.message.lower() or "what data sources" in request.message.lower():
                yield f"event: result\ndata: {json.dumps({'response': datasources_result['response']})}\n\n"
                yield f"event: done\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"
                return
            
            # Get or select datasource
            datasource_luid, metadata = get_client_datasource(session_id)
            
            if not datasource_luid:
                selected_datasource = select_datasource_for_query(request.message, available_datasources)
                
                if selected_datasource:
                    store_client_datasource(session_id, selected_datasource['id'], {
                        'name': selected_datasource['name'],
                        'project': selected_datasource['project']
                    })
                    datasource_luid = selected_datasource['id']
                    metadata = {'name': selected_datasource['name']}
                else:
                    yield f"event: result\ndata: {json.dumps({'response': 'I apologize, but I couldn\'t find a suitable data source for your query.'})}\n\n"
                    yield f"event: done\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"
                    return
            
            yield f"event: progress\ndata: {json.dumps({'message': 'Starting MCP analysis...', 'step': 'analysis-start'})}\n\n"
            
            # Prepare conversation context
            conversation = []
            if metadata and metadata.get('name'):
                conversation.append({
                    'role': 'system',
                    'content': f"You are analyzing data from the {metadata.get('name')} datasource. Provide specific, data-driven answers."
                })
            
            # Use iterative MCP chat
            result = await mcp_chat_iterative(conversation, request.message, datasource_luid)
            
            response_text = result.get('response', '')
            
            if not response_text:
                response_text = "I apologize, but I was unable to process your request at this time."
            
            # Send final result
            yield f"event: result\ndata: {json.dumps({'response': response_text, 'toolResults': result.get('toolResults', []), 'iterations': result.get('iterations', 0)})}\n\n"
            yield f"event: done\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"
            
        except Exception as e:
            print("❌ Error in streaming chat endpoint:")
            traceback.print_exc()
            yield f"event: error\ndata: {json.dumps({'error': 'Failed to process chat request', 'details': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)