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
    """Enhanced OpenAI completion with better parameters for reasoning"""
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
        'temperature': 0.1,  # Slight creativity while maintaining accuracy
        'max_tokens': 4000,  # Allow longer responses
        'top_p': 0.9,       # Focus on high-probability tokens
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
    """Enhanced MCP chat with better reasoning and tool usage"""
    
    def send_event(event: str, data: dict):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        yield send_event('progress', {'message': 'Initializing enhanced MCP analysis...', 'step': 'init'})

        async with MCPClient(MCP_SERVER_URL) as mcp_client:
            yield send_event('progress', {'message': 'Loading advanced analytics tools...', 'step': 'tools'})

            # Get available tools from MCP server
            tools = await mcp_client.list_tools()

            yield send_event('progress', {
                'message': f'Loaded {len(tools)} specialized tools for data analysis',
                'step': 'tools-found'
            })

            # Enhanced OpenAI tools format
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    'type': 'function',
                    'function': {
                        'name': tool.get('name'),
                        'description': tool.get('description'),
                        'parameters': tool.get('inputSchema', {})
                    }
                })

            # Use the enhanced MCP system prompt
            ds_metadata = DATASOURCE_METADATA_STORE.get(client_id) if client_id else None
            agent_identity = "Expert Tableau Data Analyst with MCP Tools"
            
            if ds_metadata:
                agent_identity = build_agent_identity(ds_metadata)
                system_content = build_mcp_enhanced_system_prompt(
                    agent_identity, 
                    ds_metadata.get("name", "this Tableau datasource")
                )
            else:
                system_content = build_mcp_enhanced_system_prompt(
                    agent_identity, 
                    "this Tableau datasource"
                )
            
            # Add tools context to the system prompt
            tools_context = f"""

**AVAILABLE MCP TOOLS:**
{chr(10).join([f"- {tool.get('name')}: {tool.get('description')}" for tool in tools])}

**ACTIVE DATASOURCE LUID**: {datasource_luid or 'auto-detect'}
"""
            system_content += tools_context

            # Prepare conversation with enhanced context
            conversation_messages = [
                {'role': 'system', 'content': system_content},
                *messages[-5:],  # Keep last 5 messages for context
                {'role': 'user', 'content': query}
            ]

            # Enhanced iterative processing
            current_messages = conversation_messages.copy()
            all_tool_results = []
            final_response = ''
            iteration = 0
            
            # Increase max iterations for complex analysis
            MAX_ITERATIONS = 15

            yield send_event('progress', {
                'message': 'Starting intelligent data analysis...',
                'step': 'analysis-start',
                'maxIterations': MAX_ITERATIONS
            })

            while iteration < MAX_ITERATIONS:
                iteration += 1

                yield send_event('progress', {
                    'message': f'Analysis step {iteration}: Processing and reasoning...',
                    'step': 'iteration-start',
                    'iteration': iteration
                })

                # Call OpenAI with enhanced parameters
                completion = await create_openai_completion_enhanced(current_messages, openai_tools)

                assistant_message = completion['choices'][0]['message']
                current_messages.append(assistant_message)

                # If no tool calls, we're done
                if not assistant_message.get('tool_calls'):
                    final_response = assistant_message.get('content', '')
                    yield send_event('progress', {
                        'message': 'Analysis complete - delivering insights...',
                        'step': 'complete'
                    })
                    break

                tool_calls = assistant_message['tool_calls']
                yield send_event('progress', {
                    'message': f'Executing {len(tool_calls)} analysis operations...',
                    'step': 'tools-executing',
                    'toolCount': len(tool_calls)
                })

                # Execute all tool calls with enhanced error handling
                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call['function']['name']
                        tool_args = json.loads(tool_call['function']['arguments'])

                        yield send_event('progress', {
                            'message': f'Running {tool_name}...',
                            'step': 'tool-executing',
                            'tool': tool_name
                        })

                        # Execute tool via MCP
                        result = await mcp_client.call_tool(tool_name, tool_args)

                        tool_result = {
                            'tool': tool_name,
                            'arguments': tool_args,
                            'result': result.get('content', [])
                        }

                        all_tool_results.append(tool_result)

                        yield send_event('progress', {
                            'message': f'{tool_name} completed successfully',
                            'step': 'tool-completed',
                            'tool': tool_name,
                            'success': True
                        })

                        # Add tool result to conversation
                        current_messages.append({
                            'role': 'tool',
                            'content': json.dumps(result.get('content', [])),
                            'tool_call_id': tool_call['id']
                        })

                    except Exception as tool_error:
                        error_result = {
                            'tool': tool_call['function']['name'],
                            'arguments': tool_call['function']['arguments'],
                            'error': str(tool_error)
                        }

                        all_tool_results.append(error_result)

                        yield send_event('progress', {
                            'message': f'⚠️ {tool_call["function"]["name"]} encountered an issue, trying alternatives...',
                            'step': 'tool-error',
                            'tool': tool_call['function']['name']
                        })

                        # Add error to conversation with guidance for recovery
                        error_guidance = f"Tool failed: {str(tool_error)}. Try using alternative field names or use list-fields to explore available options."
                        current_messages.append({
                            'role': 'tool',
                            'content': json.dumps({'error': str(tool_error), 'guidance': error_guidance}),
                            'tool_call_id': tool_call['id']
                        })

            # Final response generation with enhanced parameters
            if iteration >= MAX_ITERATIONS and not final_response:
                yield send_event('progress', {
                    'message': 'Finalizing comprehensive analysis...',
                    'step': 'max-iterations'
                })

                # Add instruction for final summary
                current_messages.append({
                    'role': 'system',
                    'content': 'Provide a comprehensive summary of your analysis with the data you have gathered. Focus on actionable insights and specific numbers.'
                })
                
                final_completion = await create_openai_completion_enhanced(current_messages)
                final_response = final_completion['choices'][0]['message'].get('content', '')

            # Send enhanced final result
            yield send_event('result', {
                'response': final_response,
                'toolResults': all_tool_results,
                'iterations': iteration,
                'analysisQuality': 'enhanced'
            })

            yield send_event('done', {'message': 'Enhanced analysis complete'})

    except Exception as error:
        yield send_event('error', {
            'error': 'Enhanced analysis failed',
            'details': str(error)
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

    print(f"✅ Stored published LUID for client {client_id}: {published_luid}\n")
    return {"status": "ok", "selected_luid": published_luid}

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

@app.get("/")
def home():
    return FileResponse('static/index.html')

@app.get("/index.html")
def static_index():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {"status": "ok"}

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

    if client_id not in DATASOURCE_LUID_STORE or not DATASOURCE_LUID_STORE[client_id]:
        raise HTTPException(
            status_code=400,
            detail="Datasource not initialized yet. Please wait for datasource detection to complete."
        )

    try:
        print(f"\n💬 Enhanced chat request from client {client_id}")
        print(f"🧠 Message: {request.message}")

        # Enhanced keyword detection for MCP usage
        mcp_keywords = [
            'insight', 'insights', 'metric', 'metrics', 'kpi', 'pulse', 'dashboard',
            'analytics', 'performance', 'trend', 'trends', 'analysis', 'visualiz',
            'chart', 'graph', 'business intelligence', 'bi', 'what can you tell me',
            'show me', 'analyze', 'breakdown', 'summary', 'overview'
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