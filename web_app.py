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
from utilities.prompt import build_agent_identity, build_agent_system_prompt

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

async def create_openai_completion(messages: List[Dict], tools: List[Dict] = None):
    """Create OpenAI completion with tools"""
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise Exception("OpenAI API key not found")
    
    headers = {
        'Authorization': f'Bearer {openai_api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'gpt-4o-mini',
        'messages': messages,
        'temperature': 0
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

async def mcp_chat_stream(messages: List[Dict[str, str]], query: str, datasource_luid: str = None) -> AsyncGenerator[str, None]:
    """Stream MCP chat responses with progress updates"""
    
    def send_event(event: str, data: dict):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
    
    try:
        yield send_event('progress', {'message': 'Initializing MCP connection...', 'step': 'init'})
        
        async with MCPClient(MCP_SERVER_URL) as mcp_client:
            yield send_event('progress', {'message': 'Getting available tools...', 'step': 'tools'})
            
            # Get available tools from MCP server
            tools = await mcp_client.list_tools()
            
            yield send_event('progress', {
                'message': f'Found {len(tools)} tools: {", ".join([t.get("name", "unknown") for t in tools])}',
                'step': 'tools-found'
            })
            
            # Prepare OpenAI tools format
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
            
            # Create system message with MCP context and datasource info
            system_content = f"""You are a helpful assistant that can analyze Tableau data using these available tools:
{chr(10).join([f"- {tool.get('name')}: {tool.get('description')}" for tool in tools])}

CRITICAL INSTRUCTIONS:
1. When users ask questions about their data, IMMEDIATELY use the tools to get the actual data - don't just describe what you will do.
2. ALWAYS use the datasource "eBikes Inventory and Sales" for data questions unless they specify a different datasource.
3. For data analysis questions, follow this sequence:
   - Use read-metadata or list-fields to understand the data structure
   - Use query-datasource to get the actual data needed to answer the question
   - Analyze the results and provide insights
4. Don't say "I will do X" - just do X immediately using the available tools.
5. Provide clear, actionable insights based on the actual data retrieved."""

            if datasource_luid:
                system_content += f"\n6. The current datasource LUID is: {datasource_luid}"

            system_content += """

PULSE METRICS AND KPI INSTRUCTIONS (HIGHEST PRIORITY):
7. MANDATORY: When users mention ANY of these words/phrases, use Pulse tools FIRST before any other tools:
   - "insights", "metric", "KPI", "key performance indicator", "pulse", "business performance", "dashboard"
   - "bike sales", "bike returns", "sales insights", "returns insights" (these are known Pulse metrics)
   - ANY question about insights from specific business metrics
8. For ANY insight request, ALWAYS follow this sequence:
   - FIRST: Use list-all-pulse-metric-definitions to see available metrics
   - SECOND: For any specific metric mentioned, use generate-pulse-metric-value-insight-bundle with OUTPUT_FORMAT_HTML
   - THIRD: Extract and display the actual Pulse insights AND Vega-Lite visualizations returned
   - ONLY if Pulse tools fail completely, then use regular data analysis tools
9. When displaying Pulse results, show the EXACT content returned by the tools, including HTML and Vega-Lite specs.
10. DO NOT use query-datasource for insight requests - use Pulse tools instead."""
            
            # Prepare conversation
            conversation_messages = [
                {'role': 'system', 'content': system_content},
                *messages,
                {'role': 'user', 'content': query}
            ]
            
            # Iterative tool calling
            current_messages = conversation_messages.copy()
            all_tool_results = []
            final_response = ''
            iteration = 0
            
            yield send_event('progress', {
                'message': 'Starting analysis...',
                'step': 'analysis-start',
                'maxIterations': MAX_MCP_ITERATIONS
            })
            
            while iteration < MAX_MCP_ITERATIONS:
                iteration += 1
                
                yield send_event('progress', {
                    'message': f'Iteration {iteration}/{MAX_MCP_ITERATIONS}: Analyzing and planning...',
                    'step': 'iteration-start',
                    'iteration': iteration
                })
                
                # Call OpenAI with tools
                completion = await create_openai_completion(current_messages, openai_tools)
                
                assistant_message = completion['choices'][0]['message']
                current_messages.append(assistant_message)
                
                # If no tool calls, we're done
                if not assistant_message.get('tool_calls'):
                    final_response = assistant_message.get('content', '')
                    yield send_event('progress', {
                        'message': 'Analysis complete - generating final response...',
                        'step': 'complete'
                    })
                    break
                
                tool_calls = assistant_message['tool_calls']
                yield send_event('progress', {
                    'message': f'Executing {len(tool_calls)} tool(s)...',
                    'step': 'tools-executing',
                    'toolCount': len(tool_calls)
                })
                
                # Execute all tool calls
                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call['function']['name']
                        tool_args = json.loads(tool_call['function']['arguments'])
                        
                        yield send_event('progress', {
                            'message': f'{tool_name}({", ".join([f"{k}: {json.dumps(v)}" for k, v in tool_args.items()])})',
                            'step': 'tool-executing',
                            'tool': tool_name,
                            'arguments': tool_args
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
                            'message': f'❌ {tool_call["function"]["name"]} failed: {str(tool_error)}',
                            'step': 'tool-error',
                            'tool': tool_call['function']['name'],
                            'error': str(tool_error)
                        })
                        
                        # Add error to conversation
                        current_messages.append({
                            'role': 'tool',
                            'content': json.dumps({'error': str(tool_error)}),
                            'tool_call_id': tool_call['id']
                        })
                
                yield send_event('progress', {
                    'message': f'Iteration {iteration} completed - {len(tool_calls)} tool(s) executed',
                    'step': 'iteration-complete',
                    'iteration': iteration,
                    'toolsExecuted': len(tool_calls)
                })
            
            # If we hit max iterations, make one final call
            if iteration >= MAX_MCP_ITERATIONS and not final_response:
                yield send_event('progress', {
                    'message': 'Max iterations reached - generating final response...',
                    'step': 'max-iterations'
                })
                
                final_completion = await create_openai_completion(current_messages)
                final_response = final_completion['choices'][0]['message'].get('content', '')
            
            # Send final result
            yield send_event('result', {
                'response': final_response,
                'toolResults': all_tool_results,
                'iterations': iteration
            })
            
            yield send_event('done', {'message': 'Stream complete'})
            
    except Exception as error:
        yield send_event('error', {
            'error': 'Failed to process chat request',
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

def setup_agent(request: Request = None):
    client_id = request.client.host if request else None
    datasource_luid = DATASOURCE_LUID_STORE.get(client_id)
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
    tableau_username = get_tableau_username(client_id)
    user_regions = get_user_regions(client_id)

    if not datasource_luid:
        raise RuntimeError("❌ No Tableau datasource LUID available.")
    if not tableau_username or not user_regions:
        raise RuntimeError("❌ User context missing.")

    print(f"\n🤖 Setting up agent for client {client_id}")
    print(f"📊 Datasource LUID: {datasource_luid}")
    print(f"📘 Datasource Name: {ds_metadata.get('name')}")

    if ds_metadata:
        agent_identity = build_agent_identity(ds_metadata)
        agent_prompt = build_agent_system_prompt(agent_identity, ds_metadata.get("name", "this Tableau datasource"))
    else:
        from utilities.prompt import AGENT_SYSTEM_PROMPT
        agent_prompt = AGENT_SYSTEM_PROMPT

    analyze_datasource = initialize_simple_datasource_qa(
        domain=os.environ['TABLEAU_DOMAIN_FULL'],
        site=os.environ['TABLEAU_SITE'],
        jwt_client_id=os.environ['TABLEAU_JWT_CLIENT_ID'],
        jwt_secret_id=os.environ['TABLEAU_JWT_SECRET_ID'],
        jwt_secret=os.environ['TABLEAU_JWT_SECRET'],
        tableau_api_version=os.environ['TABLEAU_API_VERSION'],
        tableau_user=tableau_username,
        datasource_luid=datasource_luid,
        tooling_llm_model="gpt-4.1-nano",
        model_provider="openai"
    )

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
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

# New MCP-powered chat endpoint with streaming
@app.post("/mcp-chat-stream")
async def mcp_chat_stream_endpoint(request: MCPChatRequest, fastapi_request: Request):
    """Stream MCP chat responses with real-time progress updates"""
    client_id = fastapi_request.client.host
    datasource_luid = DATASOURCE_LUID_STORE.get(client_id)
    
    async def generate():
        async for chunk in mcp_chat_stream(request.messages, request.query, datasource_luid):
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

# Enhanced chat endpoint that chooses between MCP and traditional approach
@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request):
    """Enhanced chat endpoint that uses MCP when available, falls back to traditional agent"""
    client_id = fastapi_request.client.host
    
    if client_id not in DATASOURCE_LUID_STORE or not DATASOURCE_LUID_STORE[client_id]:
        raise HTTPException(
            status_code=400,
            detail="Datasource not initialized yet. Please wait for datasource detection to complete."
        )
    
    try:
        print(f"\n💬 Chat request from client {client_id}")
        print(f"🧠 Message: {request.message}")
        
        # Check if we can use MCP (for insight/analytics queries)
        use_mcp = any(keyword in request.message.lower() for keyword in [
            'insight', 'metric', 'kpi', 'pulse', 'dashboard', 'analytics', 'performance',
            'trend', 'analysis', 'visualiz'
        ])
        
        if use_mcp:
            print("🔄 Using MCP server for enhanced analytics...")
            datasource_luid = DATASOURCE_LUID_STORE.get(client_id)
            
            # Use MCP for advanced analytics
            response_text = ""
            async for chunk in mcp_chat_stream([], request.message, datasource_luid):
                if chunk.startswith("event: result"):
                    try:
                        data_line = chunk.split("\n")[1]  # Get the data line
                        if data_line.startswith("data: "):
                            result_data = json.loads(data_line[6:])  # Remove "data: " prefix
                            response_text = result_data.get('response', '')
                            break
                    except (json.JSONDecodeError, IndexError):
                        continue
            
            if not response_text:
                # Fallback to traditional agent if MCP fails
                print("⚠️ MCP failed, falling back to traditional agent...")
                agent = setup_agent(fastapi_request)
                messages = {"messages": [("user", request.message)]}
                for chunk in agent.stream(messages, config=config, stream_mode="values"):
                    if 'messages' in chunk and chunk['messages']:
                        latest_message = chunk['messages'][-1]
                        if hasattr(latest_message, 'content'):
                            response_text = latest_message.content
        else:
            print("🔄 Using traditional agent for basic queries...")
            # Use traditional agent for simple queries
            agent = setup_agent(fastapi_request)
            messages = {"messages": [("user", request.message)]}
            response_text = ""
            for chunk in agent.stream(messages, config=config, stream_mode="values"):
                if 'messages' in chunk and chunk['messages']:
                    latest_message = chunk['messages'][-1]
                    if hasattr(latest_message, 'content'):
                        response_text = latest_message.content
        
        print(f"🤖 Response: {response_text}")
        return ChatResponse(response=response_text)
        
    except Exception as e:
        print("❌ Error in /chat endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)