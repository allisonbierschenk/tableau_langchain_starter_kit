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

# Session-based client tracking
import threading
from collections import defaultdict
import time

DATASOURCE_LUID_STORE = {}
DATASOURCE_METADATA_STORE = {}
CLIENT_SESSION_STORE = defaultdict(dict)  # Track client sessions
SESSION_LOCK = threading.Lock()

def get_client_session_id(request: Request) -> str:
    """Get a stable client session ID"""
    client_id = request.client.host
    
    # Try to get existing session ID from headers or create new one
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        # Create a session ID based on client IP and user agent
        user_agent = request.headers.get('User-Agent', '')
        session_id = f"{client_id}_{hash(user_agent) % 10000}"
    
    print(f"🔍 Session ID resolution: client_id={client_id}, session_id={session_id}")
    return session_id

def store_client_datasource(session_id: str, datasource_luid: str, metadata: dict):
    """Store datasource for a client session"""
    with SESSION_LOCK:
        CLIENT_SESSION_STORE[session_id]['datasource_luid'] = datasource_luid
        CLIENT_SESSION_STORE[session_id]['metadata'] = metadata
        CLIENT_SESSION_STORE[session_id]['timestamp'] = time.time()
        # Also store in legacy format for backward compatibility
        DATASOURCE_LUID_STORE[session_id] = datasource_luid
        DATASOURCE_METADATA_STORE[session_id] = metadata
        print(f"📊 Stored datasource for session {session_id}: {datasource_luid}")
        print(f"📋 Current sessions: {list(CLIENT_SESSION_STORE.keys())}")

def get_client_datasource(session_id: str) -> tuple:
    """Get datasource for a client session"""
    with SESSION_LOCK:
        luid = CLIENT_SESSION_STORE[session_id].get('datasource_luid')
        metadata = CLIENT_SESSION_STORE[session_id].get('metadata', {})
        timestamp = CLIENT_SESSION_STORE[session_id].get('timestamp', 0)
        print(f"🔍 Retrieved datasource for session {session_id}: luid={luid}, timestamp={timestamp}")
        return luid, metadata

def select_datasource_for_query(query: str, available_datasources: list) -> dict:
    """Intelligently select the most appropriate datasource based on the query"""
    query = query.lower()
    
    # Define topic-datasource mappings
    topic_mappings = {
        'pharmacy': ['pharmacist', 'medication', 'drug', 'prescription', 'expire', 'inventory'],
        'supply_chain': ['supply', 'chain', 'hospital', 'equipment'],
        'sales': ['sales', 'revenue', 'profit', 'customer'],
        'admin': ['user', 'permission', 'group', 'subscription']
    }
    
    # Score each datasource based on relevance to query
    scored_datasources = []
    for ds in available_datasources:
        score = 0
        name = ds['name'].lower()
        project = ds['project'].lower()
        description = ds.get('description', '').lower()
        
        # Check which topics are relevant to the query
        query_topics = set()
        for topic, keywords in topic_mappings.items():
            if any(keyword in query for keyword in keywords):
                query_topics.add(topic)
                # Check if datasource name/project matches the topic
                if any(keyword in name or keyword in project or keyword in description for keyword in keywords):
                    score += 5  # High score for direct topic match
        
        # Bonus points for enriched/enhanced versions
        if 'enriched' in name or 'enhanced' in name:
            score += 2
        
        # Penalize generic or unrelated datasources
        if 'sample' in name or 'test' in name:
            score -= 3
        
        # Store score and datasource
        scored_datasources.append((score, ds))
    
    # Sort by score (highest first) and return the best match
    if scored_datasources:
        scored_datasources.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ds = scored_datasources[0]
        
        # Only return if we have a positive score
        if best_score > 0:
            return best_ds
    
    return None

def cleanup_old_sessions():
    """Clean up sessions older than 1 hour"""
    with SESSION_LOCK:
        current_time = time.time()
        expired_sessions = []
        for session_id, data in CLIENT_SESSION_STORE.items():
            if current_time - data.get('timestamp', 0) > 3600:  # 1 hour
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del CLIENT_SESSION_STORE[session_id]
            if session_id in DATASOURCE_LUID_STORE:
                del DATASOURCE_LUID_STORE[session_id]
            if session_id in DATASOURCE_METADATA_STORE:
                del DATASOURCE_METADATA_STORE[session_id]
        
        if expired_sessions:
            print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions: {expired_sessions}")

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
    resp.raise_for_status()
    token = resp.json()['credentials']['token']
    site_id = resp.json()['credentials']['site']['id']
    print(f"✅ Tableau REST API token: {token}")
    print(f"✅ Site ID: {site_id}")
    return token, site_id

def get_all_datasources(tableau_username, user_regions):
    """Get all available datasources from Tableau Server"""
    token, site_id = tableau_signin_with_jwt(tableau_username, user_regions)
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {
        "X-Tableau-Auth": token,
        "Accept": "application/json"
    }

    print(f"\n🔍 Fetching all datasources from Tableau Server")
    resp = requests.get(url, headers=headers)
    print(f"🔁 Datasources fetch response code: {resp.status_code}")
    resp.raise_for_status()

    datasources = resp.json().get("datasources", {}).get("datasource", [])
    print(f"✅ Found {len(datasources)} datasources")
    
    # Format the datasources for display
    formatted_datasources = []
    for ds in datasources:
        formatted_datasources.append({
            "name": ds.get('name', 'Unknown'),
            "project": ds.get('project', {}).get('name', 'Unknown'),
            "id": ds.get('id'),
            "description": ds.get('description', ''),
            "contentUrl": ds.get('contentUrl', ''),
            "owner": ds.get('owner', {}).get('name', 'Unknown')
        })
    
    return formatted_datasources

class ChatRequest(BaseModel):
    message: str

class MCPChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    query: str

class ChatResponse(BaseModel):
    response: str

class DataSourcesRequest(BaseModel):
    datasources: dict  # { "name": "federated_id" }

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

async def initialize_tableau_tools(datasource_luid: str):
    """Initialize Tableau tools for the given datasource"""
    try:
        # Get authentication token
        user_regions = get_user_regions("default")
        tableau_username = get_tableau_username("default")
        token, site_id = tableau_signin_with_jwt(tableau_username, user_regions)
        
        # Initialize the simple datasource QA tool
        qa_tool = initialize_simple_datasource_qa(
            tableau_auth_token=token,
            site_id=site_id,
            datasource_luid=datasource_luid,
            server_url=os.environ['TABLEAU_DOMAIN_FULL']
        )
        
        # Add custom description to help the agent understand the tool better
        qa_tool.description = """This tool allows you to query a Tableau datasource directly.
        You can ask questions about the data and get answers based on the actual content.
        For example:
        - Count total number of records
        - Get unique values in a field
        - Find maximum/minimum values
        - Calculate averages or sums
        - Filter data based on conditions
        """
        
        return qa_tool
    except Exception as e:
        print(f"❌ Error initializing Tableau tools: {str(e)}")
        print(f"Error details: {str(e.__dict__)}")
        raise

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
                    'content': '''Please provide a clear and specific answer based on the data analysis results.
Focus on answering the user's exact question with precise details and numbers.
If the question is about specific metrics, items, or trends, provide those exact details.
Avoid generic responses - use the actual data from the analysis.'''
                })
                
                try:
                    final_completion = await create_openai_completion_enhanced(current_messages)
                    final_response = final_completion['choices'][0]['message'].get('content', '')
                except Exception as final_error:
                    final_response = "I apologize, but I encountered an error while analyzing the data. Please try rephrasing your question or ask about a different aspect of the data."

            # Ensure we have a meaningful response
            if not final_response:
                final_response = "I apologize, but I was unable to properly analyze the data for your question. Please try rephrasing your question or ask about a different aspect of the data."

            # Enhanced final result
            yield send_event('result', {
                'response': final_response,
                'toolResults': all_tool_results,
                'iterations': iteration,
                'analysisType': 'comprehensive',
                'toolsUsed': len([r for r in all_tool_results if r.get('success')])
            })

            yield send_event('done', {'message': 'Analysis complete'})

    except Exception as error:
        yield send_event('error', {
            'error': 'Analysis system encountered an error',
            'details': str(error),
            'suggestion': 'Please try rephrasing your question or ask for specific data points.'
        })

@app.get("/")
def home():
    return FileResponse('static/index.html')

@app.get("/index.html")
def static_index():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/datasources")
async def list_datasources():
    """List all available datasources from Tableau Server"""
    try:
        user_regions = get_user_regions("default")
        tableau_username = get_tableau_username("default")
        
        datasources = get_all_datasources(tableau_username, user_regions)
        
        # Format the response for display
        formatted_response = "Here are the data sources available:\n\n"
        for ds in datasources:
            formatted_response += f"Name: {ds['name']}\n"
            formatted_response += f"Project: {ds['project']}\n"
            formatted_response += f"ID: {ds['id']}\n\n"
        
        return {"response": formatted_response, "datasources": datasources}
        
    except Exception as e:
        print(f"❌ Error listing datasources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasources: {str(e)}")

@app.post("/mcp-chat-stream")
async def enhanced_mcp_chat_stream_endpoint(request: MCPChatRequest, fastapi_request: Request):
    """Enhanced MCP streaming endpoint with better reasoning"""
    session_id = get_client_session_id(fastapi_request)
    datasource_luid, _ = get_client_datasource(session_id)

    async def generate():
        async for chunk in mcp_chat_stream_enhanced(request.messages, request.query, datasource_luid, session_id):
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
    session_id = get_client_session_id(fastapi_request)

    # Enhanced client tracking with better debugging
    print(f"\n💬 Enhanced chat request from session {session_id}")
    print(f"🧠 Message: {request.message}")
    
    try:
        # Get available datasources first
        datasources_result = await list_datasources()
        available_datasources = datasources_result.get("datasources", [])
        
        # Check if the user is asking about datasources
        if "data sources" in request.message.lower() or "datasources" in request.message.lower() or "what data sources" in request.message.lower():
            return ChatResponse(response=datasources_result["response"])

        # Get current datasource or select appropriate one based on query
        datasource_luid, metadata = get_client_datasource(session_id)
        
        if not datasource_luid:
            # Select appropriate datasource based on query
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

        # Use MCP server for data analysis
        print("🚀 Using MCP server for intelligent analysis...")
        
        try:
            # Create conversation context
            conversation = []
            if metadata and metadata.get('name'):
                conversation.append({
                    'role': 'system',
                    'content': f"You are analyzing data from the {metadata.get('name')} datasource. Provide specific, data-driven answers."
                })
            
            conversation.append({
                'role': 'user',
                'content': request.message
            })
            
            # Use MCP streaming to get the response
            response_text = ""
            try:
                async for chunk in mcp_chat_stream_enhanced(conversation, request.message, datasource_luid, session_id):
                    print(f"🔍 Processing chunk: {chunk[:100]}...")
                    
                    # Check for rate limiting error
                    if "429" in chunk or "rate limit" in chunk.lower():
                        print("⚠️ Rate limiting detected, using fallback response")
                        response_text = "I'm experiencing high demand right now. Based on the Pharmacist Inventory datasource, I can tell you that this contains medication inventory data with fields like Drug Name, Inventory ID, Daily Quantity Remaining, Expiration Date, and more. The exact count would require a direct query to the datasource."
                        break
                    
                    if chunk.startswith("event: result"):
                        try:
                            data_line = chunk.split("\n")[1]
                            if data_line.startswith("data: "):
                                result_data = json.loads(data_line[6:])
                                response_text = result_data.get('response', '')
                                print(f"🎯 MCP Response: {response_text}")
                                break
                        except (json.JSONDecodeError, IndexError) as e:
                            print(f"❌ Error parsing MCP response: {e}")
                            continue
                            
            except Exception as e:
                print(f"❌ Error in MCP streaming: {e}")
                response_text = f"I encountered an error while analyzing the data: {str(e)}"

            print(f"🔍 Final response_text: '{response_text}' (length: {len(response_text) if response_text else 0})")
            if not response_text:
                # Fallback: Try direct MCP query for total medications
                if "pharmacist" in metadata.get('name', '').lower() and "total" in request.message.lower() and "medication" in request.message.lower():
                    try:
                        print("🔄 Trying direct MCP query as fallback...")
                        mcp_client = MCPClient("https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp")
                        async with mcp_client:
                            result = await mcp_client.call_tool("query-datasource", {
                            "datasourceLuid": datasource_luid,
                            "query": {
                                "fields": [
                                    {
                                        "fieldCaption": "Inventory ID",
                                        "function": "COUNTD",
                                        "fieldAlias": "Total Medications"
                                    }
                                ]
                            }
                        })
                        
                        if result and 'content' in result:
                            for content_item in result['content']:
                                if 'text' in content_item:
                                    try:
                                        data = json.loads(content_item['text'])
                                        if 'data' in data and len(data['data']) > 0:
                                            total = data['data'][0].get('Total Medications', 'unknown')
                                            response_text = f"Based on the Pharmacist Inventory datasource, there are **{total} total medications** in the inventory."
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        
                        if not response_text:
                            response_text = "Based on the Pharmacist Inventory datasource, I can see medication inventory data. The exact count would require a direct query to the datasource."
                            
                    except Exception as e:
                        print(f"❌ Fallback MCP query failed: {e}")
                        response_text = f"I apologize, but I'm unable to analyze the data from {metadata.get('name', 'the selected datasource')} at the moment. Please try rephrasing your question."
                else:
                    response_text = f"I apologize, but I'm unable to analyze the data from {metadata.get('name', 'the selected datasource')} at the moment. Please try rephrasing your question."
                
        except Exception as e:
            response_text = f"I apologize, but I encountered an error while trying to access the {metadata.get('name', 'selected')} datasource: {str(e)}"

            print(f"🎯 Enhanced response: {response_text[:200]}...")
            print(f"📤 Returning ChatResponse with response: {response_text}")
            
            # Ensure response_text is not None or empty
            if not response_text:
                response_text = "I apologize, but I was unable to process your request at this time."
            
            # Create and return the response
            chat_response = ChatResponse(response=response_text)
            print(f"📤 Final ChatResponse object: {chat_response}")
            return chat_response

    except Exception as e:
        print("❌ Error in enhanced chat endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)