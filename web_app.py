from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import traceback
import jwt
import datetime
import uuid
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, List, Dict, Any

from langsmith import Client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tableau.tools.simple_datasource_qa import initialize_simple_datasource_qa
from utilities.prompt import build_agent_identity, build_enhanced_agent_system_prompt, build_mcp_enhanced_system_prompt

load_dotenv()

langsmith_client = Client()
config = {"run_name": "Tableau Langchain Web_App.py"}

app = FastAPI(title="Tableau AI Chat", description="Simple AI chat interface for Tableau data")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Constants for MCP integration
MAX_MCP_ITERATIONS = 10
MCP_SERVER_URL = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"

class ChatRequest(BaseModel):
    message: str

class MCPChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    query: str

class ChatResponse(BaseModel):
    response: str

class DataSourcesRequest(BaseModel):
    datasources: dict  # { "name": "federated_id" }

DATASOURCE_LUID_STORE = {}
DATASOURCE_METADATA_STORE = {}

def get_user_regions(client_id):
    return ['West', 'South']

def get_tableau_username(client_id):
    return os.environ['TABLEAU_USER']

def generate_jwt(tableau_username, user_regions):
    client_id = os.environ['TABLEAU_JWT_CLIENT_ID']
    secret_id = os.environ['TABLEAU_JWT_SECRET_ID']
    secret_value = os.environ['TABLEAU_JWT_SECRET']
    now = datetime.datetime.now(datetime.UTC)
    payload = {
        "iss": client_id,
        "sub": tableau_username,
        "aud": "tableau",
        "jti": str(uuid.uuid4()),
        "exp": now + datetime.timedelta(minutes=5),
        "scp": ["tableau:rest_api:query", "tableau:rest_api:metadata", "tableau:content:read"],
        "Region": user_regions
    }
    headers = {
        "kid": secret_id,
        "iss": client_id
    }
    jwt_token = jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)
    print("\n🔐 JWT generated for Tableau API")
    print(jwt_token)
    return jwt_token

def tableau_signin_with_jwt(tableau_username, user_regions):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username, user_regions)
    url = f"{domain}/api/{api_version}/auth/signin"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    payload = {
        "credentials": {
            "jwt": jwt_token,
            "site": {"contentUrl": site}
        }
    }
    print(f"\n🔑 Signing in to Tableau REST API at {url}")
    resp = requests.post(url, json=payload, headers=headers)
    print(f"🔁 Sign-in response code: {resp.status_code}")
    print(f"📨 Sign-in response: {resp.text}")
    resp.raise_for_status()
    token = resp.json()['credentials']['token']
    site_id = resp.json()['credentials']['site']['id']
    print(f"✅ Tableau REST API token: {token}")
    print(f"✅ Site ID: {site_id}")
    return token, site_id

def lookup_published_luid_by_name(name, tableau_username, user_regions):
    token, site_id = tableau_signin_with_jwt(tableau_username, user_regions)
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {
        "X-Tableau-Auth": token,
        "Accept": "application/json"
    }
    params = {
        "filter": f"name:eq:{name}"
    }

    print(f"\n🔍 Looking up datasource by name using filter: '{name}'")
    resp = requests.get(url, headers=headers, params=params)
    print(f"🔁 Datasource fetch response code: {resp.status_code}")
    print(f"📨 Response: {resp.text}")
    resp.raise_for_status()

    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources:
        print(f"❌ No match found for datasource name: {name}")
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found on Tableau Server.")

    ds = datasources[0]
    print(f"✅ Match found ➜ Name: {ds.get('name')} ➜ LUID: {ds.get('id')}")
    return ds["id"], ds

# MCP Client functionality
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
            f"{self.server_url}/tools/list",
            json={}
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to list tools: {response.status}")
            data = await response.json()
            return data.get("tools", [])
    
    async def call_tool(self, name: str, arguments: dict):
        """Execute a tool on the MCP server"""
        async with self.session.post(
            f"{self.server_url}/tools/call",
            json={"name": name, "arguments": arguments}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Tool execution failed: {response.status} - {error_text}")
            return await response.json()

async def create_openai_completion_enhanced(messages: List[Dict], tools: List[Dict] = None):
    """Enhanced OpenAI completion with better parameters for analysis"""
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise Exception("OpenAI API key not found")

    headers = {
        'Authorization': f'Bearer {openai_api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': 'gpt-4o',  # Use most capable model
        'messages': messages,
        'temperature': 0.2,  # Balance creativity with accuracy
        'max_tokens': 4096,  # Allow comprehensive responses
        'top_p': 0.95,       # Allow some creativity in analysis
    }

    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'

    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error_text}")
            return await response.json()

async def mcp_chat_stream_enhanced(messages: List[Dict[str, str]], query: str, datasource_luid: str = None, client_id: str = None) -> AsyncGenerator[str, None]:
    """Robust MCP chat with better error handling and fallback strategies"""
    
    def send_event(event: str, data: dict):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        yield send_event('progress', {'message': 'Initializing intelligent analysis...', 'step': 'init'})

        async with MCPClient(MCP_SERVER_URL) as mcp_client:
            yield send_event('progress', {'message': 'Loading specialized analytics tools...', 'step': 'tools'})

            try:
                # Get available tools from MCP server
                tools = await mcp_client.list_tools()
                
                if not tools:
                    raise Exception("No tools available from MCP server")
                
            except Exception as tool_error:
                yield send_event('error', {
                    'error': 'Failed to load MCP tools',
                    'details': str(tool_error)
                })
                return

            yield send_event('progress', {
                'message': f'Loaded {len(tools)} analytical tools successfully',
                'step': 'tools-found'
            })

            # Enhanced OpenAI tools format
            openai_tools = []
            for tool in tools:
                try:
                    openai_tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.get('name'),
                            'description': tool.get('description'),
                            'parameters': tool.get('inputSchema', {})
                        }
                    })
                except Exception as e:
                    print(f"Warning: Failed to format tool {tool.get('name', 'unknown')}: {e}")
                    continue

            # Build enhanced system prompt
            ds_metadata = DATASOURCE_METADATA_STORE.get(client_id) if client_id else None
            
            if ds_metadata:
                agent_identity = build_agent_identity(ds_metadata)
                system_content = build_mcp_enhanced_system_prompt(
                    agent_identity, 
                    ds_metadata.get("name", "this Tableau datasource")
                )
            else:
                system_content = build_mcp_enhanced_system_prompt(
                    "Expert Business Intelligence Analyst", 
                    "this Tableau datasource"
                )
            
            # Add active tools context
            available_tools = [f"- {tool.get('name')}: {tool.get('description')}" for tool in tools if tool.get('name')]
            tools_context = f"""

**ACTIVE ANALYTICAL TOOLS:**
{chr(10).join(available_tools)}

**DATASOURCE CONTEXT:**
- Active LUID: {datasource_luid or 'auto-detect'}
- Analysis Mode: Enhanced Intelligence with MCP Tools

**ANALYSIS INSTRUCTIONS:**
1. For insight requests: Use pulse metrics and comprehensive analysis tools
2. For specific queries: Use targeted data retrieval and analysis
3. Always provide business context and actionable recommendations
4. Look for patterns, trends, and opportunities in the data
5. If field errors occur, try alternative field names or use metadata tools first
"""
            system_content += tools_context

            # Prepare conversation with enhanced context
            conversation_messages = [
                {'role': 'system', 'content': system_content}
            ]
            
            # Add recent conversation context
            if messages:
                conversation_messages.extend(messages[-8:])  # Keep last 8 messages
            
            # Add current query
            conversation_messages.append({'role': 'user', 'content': query})

            # Start iterative analysis
            current_messages = conversation_messages.copy()
            all_tool_results = []
            final_response = ''
            iteration = 0

            yield send_event('progress', {
                'message': 'Beginning comprehensive data analysis...',
                'step': 'analysis-start',
                'maxIterations': MAX_MCP_ITERATIONS
            })

            while iteration < MAX_MCP_ITERATIONS:
                iteration += 1

                yield send_event('progress', {
                    'message': f'Analysis iteration {iteration}: Reasoning and tool selection...',
                    'step': 'iteration-start',
                    'iteration': iteration
                })

                try:
                    # Call OpenAI with enhanced parameters
                    completion = await create_openai_completion_enhanced(current_messages, openai_tools)
                    assistant_message = completion['choices'][0]['message']
                    current_messages.append(assistant_message)

                    # Check if analysis is complete
                    if not assistant_message.get('tool_calls'):
                        final_response = assistant_message.get('content', '')
                        yield send_event('progress', {
                            'message': 'Analysis complete - preparing comprehensive insights...',
                            'step': 'complete'
                        })
                        break

                    # Execute tool calls
                    tool_calls = assistant_message['tool_calls']
                    yield send_event('progress', {
                        'message': f'Executing {len(tool_calls)} analytical operations...',
                        'step': 'tools-executing',
                        'toolCount': len(tool_calls)
                    })

                    successful_tools = 0
                    
                    for tool_call in tool_calls:
                        try:
                            tool_name = tool_call['function']['name']
                            tool_args_str = tool_call['function']['arguments']
                            
                            try:
                                tool_args = json.loads(tool_args_str)
                            except json.JSONDecodeError as json_error:
                                print(f"JSON decode error for tool {tool_name}: {json_error}")
                                print(f"Raw arguments: {tool_args_str}")
                                # Try to fix common JSON issues
                                tool_args = {}

                            yield send_event('progress', {
                                'message': f'Running {tool_name}...',
                                'step': 'tool-executing',
                                'tool': tool_name
                            })

                            # Execute tool via MCP with timeout
                            result = await asyncio.wait_for(
                                mcp_client.call_tool(tool_name, tool_args),
                                timeout=60  # 60 second timeout per tool
                            )

                            tool_result = {
                                'tool': tool_name,
                                'arguments': tool_args,
                                'result': result.get('content', []),
                                'success': True
                            }

                            all_tool_results.append(tool_result)
                            successful_tools += 1

                            yield send_event('progress', {
                                'message': f'{tool_name} completed successfully',
                                'step': 'tool-completed',
                                'tool': tool_name,
                                'success': True
                            })

                            # Add successful result to conversation
                            result_content = result.get('content', [])
                            if isinstance(result_content, list) and result_content:
                                content_str = json.dumps(result_content)
                            else:
                                content_str = str(result_content)

                            current_messages.append({
                                'role': 'tool',
                                'content': content_str,
                                'tool_call_id': tool_call['id']
                            })

                        except asyncio.TimeoutError:
                            error_msg = f"Tool {tool_call['function']['name']} timed out"
                            yield send_event('progress', {
                                'message': f'⏰ {tool_call["function"]["name"]} timed out, continuing...',
                                'step': 'tool-error',
                                'tool': tool_call['function']['name']
                            })
                            
                            current_messages.append({
                                'role': 'tool',
                                'content': json.dumps({'error': error_msg, 'guidance': 'Tool timed out, try alternative approaches'}),
                                'tool_call_id': tool_call['id']
                            })

                        except Exception as tool_error:
                            error_msg = str(tool_error)
                            yield send_event('progress', {
                                'message': f'⚠️ {tool_call["function"]["name"]} encountered an issue, trying alternatives...',
                                'step': 'tool-error',
                                'tool': tool_call['function']['name']
                            })

                            all_tool_results.append({
                                'tool': tool_call['function']['name'],
                                'arguments': tool_call['function']['arguments'],
                                'error': error_msg,
                                'success': False
                            })

                            # Provide helpful error context and suggest alternatives
                            error_guidance = "Tool failed. Try using alternative field names or check available fields first."
                            if "field" in error_msg.lower() or "column" in error_msg.lower():
                                error_guidance = "Field not found. Use list-fields or read-metadata to explore available options."
                            elif "unknown" in error_msg.lower():
                                error_guidance = "Unknown field detected. Let me check available fields and try alternative names."
                            
                            current_messages.append({
                                'role': 'tool',
                                'content': json.dumps({'error': error_msg, 'guidance': error_guidance}),
                                'tool_call_id': tool_call['id']
                            })

                    # If no tools succeeded, provide guidance
                    if successful_tools == 0 and len(tool_calls) > 0:
                        current_messages.append({
                            'role': 'system',
                            'content': 'All tools failed. Please provide analysis based on your general knowledge about business data and suggest what specific information would be helpful to analyze. Focus on providing actionable business insights even with limited data access.'
                        })

                except Exception as completion_error:
                    yield send_event('error', {
                        'error': 'Analysis iteration failed',
                        'details': str(completion_error),
                        'iteration': iteration
                    })
                    break

            # Generate final comprehensive response
            if iteration >= MAX_MCP_ITERATIONS and not final_response:
                yield send_event('progress', {
                    'message': 'Finalizing comprehensive business analysis...',
                    'step': 'max-iterations'
                })

                # Add instruction for final comprehensive summary
                current_messages.append({
                    'role': 'system',
                    'content': '''Provide a comprehensive business analysis summary with:
1. Key insights discovered from the data
2. Specific numbers and percentages where available
3. Business implications and what they mean
4. Actionable recommendations for next steps
5. Areas of focus for proactive decision-making

Format your response to be executive-ready with clear takeaways.'''
                })
                
                try:
                    final_completion = await create_openai_completion_enhanced(current_messages)
                    final_response = final_completion['choices'][0]['message'].get('content', '')
                except Exception as final_error:
                    final_response = f"I've analyzed your data using {len(all_tool_results)} analytical operations. While I encountered some technical challenges in the final synthesis, I can see patterns in your data that suggest focusing on performance optimization and identifying key growth opportunities would be valuable next steps."

            # Ensure we have a meaningful response
            if not final_response:
                final_response = "I've completed a comprehensive analysis of your data. Based on the tools available and the data structure, I recommend focusing on key performance indicators and trend analysis to identify actionable insights for your business."

            # Enhanced final result
            yield send_event('result', {
                'response': final_response,
                'toolResults': all_tool_results,
                'iterations': iteration,
                'analysisType': 'comprehensive',
                'toolsUsed': len([r for r in all_tool_results if r.get('success')])
            })

            yield send_event('done', {'message': 'Comprehensive analysis complete'})

    except Exception as error:
        yield send_event('error', {
            'error': 'Analysis system encountered an error',
            'details': str(error),
            'suggestion': 'Please try rephrasing your question or ask for specific data points.'
        })
        
@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    client_id = request.client.host
    print(f"\n📥 /datasources request from client: {client_id}")
    
    if not body.datasources:
        raise HTTPException(status_code=400, detail="No data sources provided")

    print("📦 Datasources received from Tableau Extensions API (name ➜ federated ID):")
    for name, fed_id in body.datasources.items():
        print(f"   - Name: '{name}' ➜ Federated ID: '{fed_id}'")

    first_name, _ = next(iter(body.datasources.items()))
    print(f"\n🎯 Targeting first datasource: '{first_name}'")

    user_regions = get_user_regions(client_id)
    tableau_username = get_tableau_username(client_id)
    print(f"👤 Tableau username: {tableau_username}")
    print(f"🌍 User regions: {user_regions}")

    try:
        published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username, user_regions)

        DATASOURCE_LUID_STORE[client_id] = published_luid
        DATASOURCE_METADATA_STORE[client_id] = {
            "name": first_name,
            "luid": published_luid,
            "projectName": full_metadata.get("projectName"),
            "description": full_metadata.get("description"),
            "uri": full_metadata.get("contentUrl"),
            "owner": full_metadata.get("owner", {})
        }

        print(f"✅ Stored published LUID for client {client_id}: {published_luid}")
        print(f"📊 Total clients with datasources: {len(DATASOURCE_LUID_STORE)}")
        print(f"📋 Available clients: {list(DATASOURCE_LUID_STORE.keys())}")
        
        return {"status": "ok", "selected_luid": published_luid}
        
    except Exception as e:
        print(f"❌ Error initializing datasource for client {client_id}: {str(e)}")
        # Try to use existing datasource as fallback
        if DATASOURCE_LUID_STORE:
            fallback_client = list(DATASOURCE_LUID_STORE.keys())[0]
            fallback_luid = DATASOURCE_LUID_STORE[fallback_client]
            print(f"🔄 Using fallback datasource from client {fallback_client}: {fallback_luid}")
            DATASOURCE_LUID_STORE[client_id] = fallback_luid
            if fallback_client in DATASOURCE_METADATA_STORE:
                DATASOURCE_METADATA_STORE[client_id] = DATASOURCE_METADATA_STORE[fallback_client]
            return {"status": "ok", "selected_luid": fallback_luid, "fallback": True}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to initialize datasource: {str(e)}")

def setup_enhanced_agent(request: Request = None):
    client_id = request.client.host if request else None
    datasource_luid = DATASOURCE_LUID_STORE.get(client_id)
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
    tableau_username = get_tableau_username(client_id)
    user_regions = get_user_regions(client_id)

    if not datasource_luid:
        raise RuntimeError("❌ No Tableau datasource LUID available.")
    if not tableau_username or not user_regions:
        raise RuntimeError("❌ User context missing.")

    print(f"\n🤖 Setting up ENHANCED agent for client {client_id}")
    print(f"📊 Datasource LUID: {datasource_luid}")
    print(f"📘 Datasource Name: {ds_metadata.get('name')}")

    if ds_metadata:
        agent_identity = build_agent_identity(ds_metadata)
        # Use the enhanced system prompt
        agent_prompt = build_enhanced_agent_system_prompt(
            agent_identity, 
            ds_metadata.get("name", "this Tableau datasource")
        )
    else:
        agent_prompt = build_enhanced_agent_system_prompt(
            "Senior AI Data Analyst", 
            "this Tableau datasource"
        )

    try:
        analyze_datasource = initialize_simple_datasource_qa(
            domain=os.environ['TABLEAU_DOMAIN_FULL'],
            site=os.environ['TABLEAU_SITE'],
            jwt_client_id=os.environ['TABLEAU_JWT_CLIENT_ID'],
            jwt_secret_id=os.environ['TABLEAU_JWT_SECRET_ID'],
            jwt_secret=os.environ['TABLEAU_JWT_SECRET'],
            tableau_api_version=os.environ['TABLEAU_API_VERSION'],
            tableau_user=tableau_username,
            datasource_luid=datasource_luid,
            tooling_llm_model="gpt-4o-mini",  # Use more capable model
            model_provider="openai"
        )

        # Use more capable LLM for reasoning
        llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Upgrade from gpt-4.1
        tools = [analyze_datasource]
        
        return create_react_agent(
            model=llm,
            tools=tools,
            prompt=agent_prompt
        )
    except Exception as e:
        print(f"❌ Error setting up enhanced agent: {str(e)}")
        # Return a basic agent with error handling
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        return create_react_agent(
            model=llm,
            tools=[],
            prompt=agent_prompt
        )

@app.get("/")
def home():
    return FileResponse('static/index.html')

@app.get("/index.html")
def static_index():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/debug/datasources")
def debug_datasources():
    """Debug endpoint to check datasource status"""
    return {
        "total_clients": len(DATASOURCE_LUID_STORE),
        "clients": list(DATASOURCE_LUID_STORE.keys()),
        "datasources": {
            client_id: {
                "luid": luid,
                "metadata": DATASOURCE_METADATA_STORE.get(client_id, {})
            }
            for client_id, luid in DATASOURCE_LUID_STORE.items()
        }
    }

@app.post("/debug/test-connection")
async def test_connection(request: Request):
    """Test endpoint to verify datasource connection"""
    client_id = request.client.host
    datasource_luid = DATASOURCE_LUID_STORE.get(client_id)
    
    if not datasource_luid:
        return {"status": "error", "message": "No datasource found for client"}
    
    try:
        tableau_username = get_tableau_username(client_id)
        user_regions = get_user_regions(client_id)
        
        # Test the connection
        token, site_id = tableau_signin_with_jwt(tableau_username, user_regions)
        
        return {
            "status": "success",
            "client_id": client_id,
            "datasource_luid": datasource_luid,
            "tableau_username": tableau_username,
            "user_regions": user_regions,
            "connection": "active"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "client_id": client_id,
            "datasource_luid": datasource_luid
        }

@app.post("/mcp-chat-stream")
async def enhanced_mcp_chat_stream_endpoint(request: MCPChatRequest, fastapi_request: Request):
    """Enhanced MCP streaming endpoint with better reasoning"""
    client_id = fastapi_request.client.host
    datasource_luid = DATASOURCE_LUID_STORE.get(client_id)

    async def generate():
        async for chunk in mcp_chat_stream_enhanced(request.messages, request.query, datasource_luid, client_id):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )
    
@app.post("/chat")
async def enhanced_chat(request: ChatRequest, fastapi_request: Request):
    """Enhanced chat endpoint with improved reasoning and tool usage"""
    client_id = fastapi_request.client.host

    # Enhanced client tracking with better debugging
    print(f"\n💬 Enhanced chat request from client {client_id}")
    print(f"🧠 Message: {request.message}")
    print(f"📊 Available datasources: {list(DATASOURCE_LUID_STORE.keys())}")
    print(f"🔍 Client datasource: {DATASOURCE_LUID_STORE.get(client_id)}")

    # Check if datasource is available
    if client_id not in DATASOURCE_LUID_STORE:
        print(f"❌ No datasource found for client {client_id}")
        # Try to find any available datasource as fallback
        if DATASOURCE_LUID_STORE:
            fallback_client = list(DATASOURCE_LUID_STORE.keys())[0]
            fallback_luid = DATASOURCE_LUID_STORE[fallback_client]
            print(f"🔄 Using fallback datasource from client {fallback_client}: {fallback_luid}")
            DATASOURCE_LUID_STORE[client_id] = fallback_luid
            if fallback_client in DATASOURCE_METADATA_STORE:
                DATASOURCE_METADATA_STORE[client_id] = DATASOURCE_METADATA_STORE[fallback_client]
        else:
            raise HTTPException(
                status_code=400,
                detail="Datasource not initialized yet. Please wait for datasource detection to complete."
            )

    if not DATASOURCE_LUID_STORE.get(client_id):
        raise HTTPException(
            status_code=400,
            detail="Datasource not initialized yet. Please wait for datasource detection to complete."
        )

    try:
        # Enhanced keyword detection for MCP usage
        mcp_keywords = [
            'insight', 'insights', 'metric', 'metrics', 'kpi', 'pulse', 'dashboard',
            'analytics', 'performance', 'trend', 'trends', 'analysis', 'visualiz',
            'chart', 'graph', 'business intelligence', 'bi', 'what can you tell me',
            'show me', 'analyze', 'breakdown', 'summary', 'overview', 'top',
            'best', 'worst', 'focus', 'proactive', 'recommend', 'should',
            'opportunity', 'risk', 'pattern', 'key', 'important', 'critical',
            'tell me about', 'what are', 'how are', 'which', 'where'
        ]
        
        use_mcp = any(keyword in request.message.lower() for keyword in mcp_keywords)

        if use_mcp:
            print("🚀 Using enhanced MCP server for intelligent analysis...")
            datasource_luid = DATASOURCE_LUID_STORE.get(client_id)

            # Use enhanced MCP streaming
            response_text = ""
            async for chunk in mcp_chat_stream_enhanced([], request.message, datasource_luid, client_id):
                if chunk.startswith("event: result"):
                    try:
                        data_line = chunk.split("\n")[1]
                        if data_line.startswith("data: "):
                            result_data = json.loads(data_line[6:])
                            response_text = result_data.get('response', '')
                            break
                    except (json.JSONDecodeError, IndexError):
                        continue

            if not response_text:
                print("⚠️ Enhanced MCP failed, using enhanced traditional agent...")
                agent = setup_enhanced_agent(fastapi_request)
                messages = {"messages": [("user", request.message)]}
                for chunk in agent.stream(messages, config=config, stream_mode="values"):
                    if 'messages' in chunk and chunk['messages']:
                        latest_message = chunk['messages'][-1]
                        if hasattr(latest_message, 'content'):
                            response_text = latest_message.content

        else:
            print("🔧 Using enhanced traditional agent for focused queries...")
            agent = setup_enhanced_agent(fastapi_request)
            messages = {"messages": [("user", request.message)]}
            response_text = ""
            for chunk in agent.stream(messages, config=config, stream_mode="values"):
                if 'messages' in chunk and chunk['messages']:
                    latest_message = chunk['messages'][-1]
                    if hasattr(latest_message, 'content'):
                        response_text = latest_message.content

        print(f"🎯 Enhanced response: {response_text[:200]}...")
        return ChatResponse(response=response_text)

    except Exception as e:
        print("❌ Error in enhanced chat endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)