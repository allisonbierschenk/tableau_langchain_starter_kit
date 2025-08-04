# web_app.py - FINAL VERSION
# This version correctly integrates your original authentication and frontend logic
# with a direct Python translation of the working MCP TypeScript example.

import os
import json
import uuid
import logging
import datetime
import traceback
import asyncio
from typing import Dict, Any, List, Optional

import requests
import jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI as OpenAIClient

# Use your existing prompt structure
from utilities.prompt import build_agent_identity, build_agent_system_prompt

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variables ---
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

app = FastAPI(title="Tableau AI Chat", description="A streaming AI chat interface for Tableau data.")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class DatasourceRequest(BaseModel):
    datasources: Dict[str, str]

# --- In-Memory Store ---
DATASOURCE_METADATA_STORE: Dict[str, Dict] = {}


# ==============================================================================
# === YOUR TABLEAU AUTHENTICATION & LOOKUP FUNCTIONS (RESTORED & UNCHANGED) ===
# ==============================================================================

def get_user_regions(client_id: str) -> List[str]:
    return ['West', 'South']

def get_tableau_username(client_id: str) -> str:
    return os.environ['TABLEAU_USER']

def generate_jwt(tableau_username: str, user_regions: List[str]) -> str:
    client_id = os.environ['TABLEAU_JWT_CLIENT_ID']
    secret_id = os.environ['TABLEAU_JWT_SECRET_ID']
    secret_value = os.environ['TABLEAU_JWT_SECRET']
    now = datetime.datetime.now(datetime.UTC)
    payload = {
        "iss": client_id, "sub": tableau_username, "aud": "tableau", "jti": str(uuid.uuid4()),
        "exp": now + datetime.timedelta(minutes=5),
        "scp": ["tableau:rest_api:read", "tableau:datasources:query", "tableau:content:read"],
        "Region": user_regions
    }
    headers = {"kid": secret_id, "iss": client_id}
    return jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)

def tableau_signin_with_jwt(tableau_username: str, user_regions: List[str]) -> (str, str):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username, user_regions)
    url = f"{domain}/api/{api_version}/auth/signin"
    payload = {"credentials": {"jwt": jwt_token, "site": {"contentUrl": site}}}
    logger.info(f"Signing in to Tableau REST API at {url}")
    resp = requests.post(url, json=payload, headers={'Accept': 'application/json'})
    resp.raise_for_status()
    creds = resp.json()['credentials']
    logger.info("Successfully signed in to Tableau.")
    return creds['token'], creds['site']['id']

def lookup_published_luid_by_name(name: str, tableau_username: str, user_regions: List[str]) -> (str, dict):
    token, site_id = tableau_signin_with_jwt(tableau_username, user_regions)
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {"X-Tableau-Auth": token, "Accept": "application/json"}
    params = {"filter": f"name:eq:{name}"}
    logger.info(f"Looking up datasource by name using filter: '{name}'")
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources:
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found.")
    ds = datasources[0]
    logger.info(f"Match found -> Name: {ds.get('name')} -> LUID: {ds.get('id')}")
    return ds["id"], ds


# ==============================================================================
# === MCP CLIENT AND STREAMING LOGIC (PYTHON TRANSLATION OF YOUR EXAMPLE) ===
# ==============================================================================

class MCPClient:
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream',
            'MCP-Protocol-Version': '0.1.0'
        })
        self._timeout = 180
        logger.info(f"MCP Client initialized for {server_url}")

    def _send_request(self, method: str, params: Dict = None) -> Dict:
        request_id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "method": method, "id": request_id, "params": params or {}}
        logger.info(f"Sending MCP Request -> ID: {request_id}, Method: {method}")
        try:
            with self._session.post(self._server_url, json=payload, stream=True, timeout=self._timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith('data:'):
                        data_content = line[5:].strip()
                        if data_content and data_content != '[DONE]':
                            try:
                                parsed_data = json.loads(data_content)
                                if parsed_data.get('id') == request_id:
                                    if 'error' in parsed_data: raise RuntimeError(f"MCP Error: {parsed_data['error']}")
                                    return parsed_data.get('result', {})
                            except json.JSONDecodeError: logger.warning(f"Could not decode SSE line: {data_content}")
                raise RuntimeError("Stream ended without a valid response.")
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP request failed: {e}", exc_info=True)
            raise RuntimeError(f"MCP communication error: {e}")

    def list_tools(self) -> Dict: return self._send_request("tools/list")
    def call_tool(self, tool_name: str, arguments: Dict) -> Dict: return self._send_request("tools/call", {"name": tool_name, "arguments": arguments})
    def close(self): self._session.close()

async def stream_chat_events(message: str, client_id: str):
    def sse_format(event_name: str, data: Dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

    mcp_client = None
    try:
        ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
        if not ds_metadata or "luid" not in ds_metadata:
            raise ValueError("Datasource LUID not initialized. Please refresh.")
        
        datasource_luid = ds_metadata["luid"]
        datasource_name = ds_metadata["name"]
        
        yield sse_format("progress", {"message": "Initializing AI assistant..."})
        
        mcp_server_url = os.environ.get("MCP_SERVER_URL", "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp")
        mcp_client = MCPClient(server_url=mcp_server_url)
        openai_client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))

        yield sse_format("progress", {"message": "Fetching available tools from MCP..."})
        
        tools_response = mcp_client.list_tools()
        mcp_tools = tools_response.get('tools', [])
        
        openai_tools = [{
            "type": "function",
            "function": {"name": tool['name'], "description": tool['description'], "parameters": tool['inputSchema']}
        } for tool in mcp_tools]

        agent_identity = build_agent_identity(ds_metadata)
        base_prompt = build_agent_system_prompt(agent_identity, datasource_name)
        critical_instructions = f"""
        **CRITICAL INSTRUCTIONS (HIGHEST PRIORITY):**
        1. You MUST use the datasource with the unique ID (LUID): `{datasource_luid}`.
        2. When a tool requires a `datasourceLuid`, you MUST use this value.
        3. Do NOT use `list-datasources`. You have been given the only one you need.
        4. Your first step is ALWAYS `list-fields` to see the columns, then `query-datasource`.
        """
        system_prompt = f"{base_prompt}\n{critical_instructions}"

        conversation_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]

        final_answer = ""

        # This iterative loop is a direct translation of the working TypeScript example
        for i in range(10): # Max 10 iterations
            logger.info(f"Starting AI iteration {i+1}")
            completion = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=conversation_messages,
                tools=openai_tools,
                tool_choice="auto"
            )
            assistant_message = completion.choices[0].message
            conversation_messages.append(assistant_message.model_dump(exclude_none=True))

            # If the AI responds with text, it's part of its thought process or the final answer
            if assistant_message.content:
                # Stream these thoughts/answers as tokens
                for char in assistant_message.content:
                    yield sse_format("token", {"token": char})
                    await asyncio.sleep(0.005) # Small delay for streaming effect

            # If there are no more tools to call, the loop is done
            if not assistant_message.tool_calls:
                final_answer = assistant_message.content or "Done."
                break

            # Execute tool calls
            yield sse_format("progress", {"message": f"Querying {datasource_name}..."})
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                try:
                    tool_result_content = mcp_client.call_tool(tool_name, tool_args).get('content')
                    conversation_messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": json.dumps(tool_result_content)})
                except Exception as e:
                    error_str = str(e)
                    conversation_messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": json.dumps({"error": error_str})})

        yield sse_format("result", {"response": final_answer})

    except Exception as e:
        logger.error(f"Error in chat stream: {e}", exc_info=True)
        yield sse_format("error", {"error": str(e)})
    finally:
        if mcp_client: mcp_client.close()
        yield sse_format("done", {"message": "Stream complete"})

# --- FastAPI Endpoints ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, fastapi_request: Request):
    client_id = fastapi_request.client.host
    logger.info(f"Received chat request from {client_id}: '{request.message[:100]}...'")
    return StreamingResponse(stream_chat_events(request.message, client_id), media_type="text/event-stream")

@app.post("/datasources")
async def receive_datasources(request: Request, body: DatasourceRequest):
    client_id = request.client.host
    try:
        datasources_map = body.datasources
        if not datasources_map:
            raise HTTPException(status_code=400, detail="No data sources provided")

        first_name = next(iter(datasources_map.keys()))
        user_regions = get_user_regions(client_id)
        tableau_username = get_tableau_username(client_id)
        
        published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username, user_regions)
        
        DATASOURCE_METADATA_STORE[client_id] = {
            "name": first_name, "luid": published_luid,
            "description": full_metadata.get("description", "")
        }
        return JSONResponse({"status": "success", "initialized_datasource_name": first_name})
    except Exception as e:
        logger.error(f"Datasource initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Static Files and Startup ---
@app.get("/")
async def root():
    return FileResponse('static/index.html')

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    required_vars = [
        'MCP_SERVER_URL', 'OPENAI_API_KEY', 'TABLEAU_USER', 
        'TABLEAU_JWT_CLIENT_ID', 'TABLEAU_JWT_SECRET_ID', 'TABLEAU_JWT_SECRET',
        'TABLEAU_DOMAIN_FULL', 'TABLEAU_SITE'
    ]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logger.error(f"FATAL: Missing one or more required environment variables: {missing}")
    else:
        logger.info("Application startup complete.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))