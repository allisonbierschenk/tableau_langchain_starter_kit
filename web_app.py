# web_app.py - REWRITTEN

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, create_model
import os
from dotenv import load_dotenv
import requests
import traceback
import uuid
from typing import Type, Any, Dict
import logging
import jwt
import datetime
import json

# LangChain Imports
from langsmith import Client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
app = FastAPI(title="Tableau AI Chat", description="Simple AI chat interface for Tableau data")

# --- In-Memory Store & LangSmith Config ---
DATASOURCE_METADATA_STORE = {}
try:
    langsmith_client = Client()
    config = {"run_name": "Tableau MCP Langchain Web_App.py"}
except Exception as e:
    logger.warning(f"LangSmith client initialization failed: {e}")
    langsmith_client = None
    config = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel): message: str
class ChatResponse(BaseModel): response: str
class DataSourcesRequest(BaseModel): datasources: dict

# --- THE NEW STATEFUL MCP CLIENT ---
class MCPClient:
    """
    A stateful client that manages a persistent session with the MCP server,
    mimicking the behavior of the official TypeScript SDK.
    """
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/x-ndjson',
            'Accept': 'application/x-ndjson',
            'MCP-Protocol-Version': '0.1.0'
        })
        logger.info(f"MCP Client initialized for {server_url}")

    def _send_request(self, method: str, params: Dict = None):
        """Helper to send a JSON-RPC request over the persistent session."""
        payload = {"jsonrpc": "2.0", "method": method, "id": str(uuid.uuid4())}
        if params:
            payload["params"] = params
        
        ndjson_payload = json.dumps(payload) + '\n'
        
        try:
            response = self._session.post(
                self._server_url,
                data=ndjson_payload.encode('utf-8'),
                timeout=30
            )
            response.raise_for_status()
            response_data = json.loads(response.text.strip())
            
            if 'error' in response_data:
                error_message = response_data['error'].get('message', 'Unknown error')
                raise RuntimeError(f"MCP server error for method '{method}': {error_message}")
                
            return response_data.get('result', {})
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP request failed for method '{method}': {e}")
            raise RuntimeError(f"Failed to communicate with MCP server during '{method}' call.") from e

    def connect(self):
        """
        FIXED: Establishes the connection by sending the required handshake message.
        """
        logger.info("Connecting to MCP server with handshake...")
        connect_params = {
            "name": "langchain-python-client",
            "version": "0.1.0",
            "capabilities": {
                "tools": {}
            }
        }
        # This call officially establishes the session.
        self._send_request(method="connect", params=connect_params)
        logger.info("MCP handshake successful.")

    def list_tools(self) -> Dict:
        """Fetches the list of available tools from the server."""
        logger.info("Fetching tools from MCP server...")
        return self._send_request(method="listTools")

    def call_tool(self, tool_name: str, arguments: Dict, context: Dict) -> Dict:
        """Calls a specific tool on the server."""
        logger.info(f"Calling MCP Tool '{tool_name}' with args: {arguments}")
        params = {
            "name": tool_name,
            "arguments": arguments,
            "context": context
        }
        return self._send_request(method="callTool", params=params)

    def close(self):
        """Closes the session."""
        logger.info("Closing MCP client session.")
        self._session.close()

# --- REFACTORED LANGCHAIN TOOL ---
class MCPTool(BaseTool):
    """
    A refactored tool that uses the stateful MCPClient instance
    instead of making its own stateless requests.
    """
    client: MCPClient
    tool_name: str
    datasource_luid: str

    def _run(self, **kwargs: Any) -> Any:
        """Use the connected client to call the tool."""
        context = {"datasource_luid": self.datasource_luid}
        result = self.client.call_tool(self.tool_name, kwargs, context)
        return result.get('content', '')

# --- REFACTORED AGENT SETUP ---
def setup_agent(client: MCPClient, ds_metadata: Dict):
    """
    Sets up the LangChain agent using the pre-connected MCPClient.
    """
    try:
        datasource_luid = ds_metadata['luid']
        logger.info(f"Setting up agent with Datasource LUID: {datasource_luid}")

        # Fetch tools using the connected client
        available_tools_data = client.list_tools().get('tools', [])
        if not available_tools_data:
            raise RuntimeError("MCP server returned no available tools.")

        tools = []
        for tool_data in available_tools_data:
            tool_name = tool_data['name']
            fields = {
                p_name: (str, Field(..., description=p_details.get('description', '')))
                for p_name, p_details in tool_data.get('inputSchema', {}).get('properties', {}).items()
            }
            args_schema = create_model(f'{tool_name}Args', **fields) if fields else BaseModel

            mcp_tool = MCPTool(
                client=client, # Pass the connected client instance
                tool_name=tool_name,
                datasource_luid=datasource_luid,
                name=tool_name,
                description=tool_data['description'],
                args_schema=args_schema
            )
            tools.append(mcp_tool)

        logger.info(f"Created {len(tools)} tools: {[t.name for t in tools]}")
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        system_prompt = (f"You are a helpful assistant for the Tableau datasource named '{ds_metadata.get('name', 'N/A')}'. "
                         f"Description: '{ds_metadata.get('description', 'N/A')}'. Use your tools to answer questions.")

        return create_react_agent(model=llm, tools=tools, messages_modifier=system_prompt)
    except Exception as e:
        logger.error(f"Error setting up agent: {e}")
        # Re-raise to be caught by the chat endpoint
        raise

# --- REFACTORED CHAT ENDPOINT ---
@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    """Main chat endpoint that creates a stateful client for the conversation."""
    client_id = fastapi_request.client.host
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
    if not ds_metadata:
        raise HTTPException(status_code=400, detail="Datasource not initialized.")

    logger.info(f"Chat request from client {client_id}: '{request.message}'")
    
    mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
    mcp_client = MCPClient(server_url=mcp_server_url)

    try:
        mcp_client.connect() # Establish the session
        agent = setup_agent(mcp_client, ds_metadata) # Pass the connected client
        
        messages = {"messages": [("user", request.message)]}
        response_text = ""
        for chunk in agent.stream(messages, config=config):
            if "agent" in chunk and hasattr(chunk["agent"], 'messages'):
                response_text = chunk["agent"].messages[-1].content

        logger.info(f"Final response: {response_text}")
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        mcp_client.close() # Ensure the session is always closed

# --- STATIC & BOILERPLATE (UNCHANGED) ---
# (The /datasources endpoint and other helpers can remain as they were)

def get_tableau_username(client_id): return os.environ['TABLEAU_USER']
def generate_jwt(tableau_username):
    # ... (code is correct)
    client_id = os.environ['TABLEAU_JWT_CLIENT_ID']; secret_id = os.environ['TABLEAU_JWT_SECRET_ID']; secret_value = os.environ['TABLEAU_JWT_SECRET']
    now = datetime.datetime.now(datetime.UTC)
    payload = {"iss": client_id, "sub": tableau_username, "aud": "tableau", "jti": str(uuid.uuid4()), "exp": now + datetime.timedelta(minutes=5), "scp": ["tableau:rest_api:query", "tableau:rest_api:metadata", "tableau:content:read"]}
    headers = {"kid": secret_id, "iss": client_id}
    return jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)

def tableau_signin_with_jwt(tableau_username):
    # ... (code is correct)
    domain = os.environ['TABLEAU_DOMAIN_FULL']; api_version = os.environ.get('TABLEAU_API_VERSION', '3.21'); site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username)
    url = f"{domain}/api/{api_version}/auth/signin"
    payload = {"credentials": {"jwt": jwt_token, "site": {"contentUrl": site}}}
    resp = requests.post(url, json=payload, headers={'Accept': 'application/json'}, timeout=20)
    resp.raise_for_status()
    creds = resp.json()['credentials']
    return creds['token'], creds['site']['id']

def lookup_published_luid_by_name(name, tableau_username):
    # ... (code is correct)
    token, site_id = tableau_signin_with_jwt(tableau_username); domain = os.environ['TABLEAU_DOMAIN_FULL']; api_version = os.environ.get('TABLEAU_API_VERSION', '3.26')
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {"X-Tableau-Auth": token, "Accept": "application/json"}; params = {"filter": f"name:eq:{name}"}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources: raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found.")
    return datasources[0]["id"], datasources[0]

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    # ... (code is correct)
    client_id = request.client.host; first_name, _ = next(iter(body.datasources.items()))
    tableau_username = get_tableau_username(client_id)
    published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username)
    DATASOURCE_METADATA_STORE[client_id] = {"name": first_name, "luid": published_luid, "projectName": full_metadata.get("projectName"), "description": full_metadata.get("description"), "owner_name": full_metadata.get("owner", {}).get("name")}
    return {"status": "ok", "selected_datasource_name": first_name, "selected_datasource_luid": published_luid}

@app.get("/", include_in_schema=False)
async def home(): return FileResponse('static/index.html')
if os.path.exists("static"): app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__" and not os.environ.get("VERCEL"):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))