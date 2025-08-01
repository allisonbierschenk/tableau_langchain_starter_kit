# web_app.py - REWRITTEN (keeping working datasource logic)

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

# --- IMPROVED MCP CLIENT
# Replace the MCPClient class in your web_app.py with this fixed version

# Replace the MCPClient class in your web_app.py with this fixed version

class MCPClient:
    """
    A stateful client that manages a persistent session with the MCP server,
    mimicking the behavior of the official TypeScript SDK.
    """
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',  # Server requires both
            'MCP-Protocol-Version': '0.1.0'
        })
        logger.info(f"MCP Client initialized for {server_url}")

    def _parse_sse_response(self, response_text: str) -> Dict:
        """
        Parse Server-Sent Events response format properly.
        Based on the working TypeScript implementation pattern.
        """
        try:
            response_text = response_text.strip()
            logger.info(f"Parsing SSE response: {response_text}")
            
            # Handle Server-Sent Events format
            lines = response_text.split('\n')
            current_event = ''
            data_content = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('event: '):
                    current_event = line[7:].strip()  # Remove 'event: ' prefix
                    continue
                    
                if line.startswith('data: '):
                    data_content = line[6:].strip()  # Remove 'data: ' prefix
                    break
                    
                # Handle case where there's no explicit 'data: ' prefix
                # but the line contains JSON (fallback)
                if line and not line.startswith('event:') and not line.startswith('id:') and not line.startswith('retry:'):
                    # Try to parse as JSON directly
                    try:
                        json.loads(line)
                        data_content = line
                        break
                    except json.JSONDecodeError:
                        continue
            
            # If we found data content, parse it as JSON
            if data_content:
                try:
                    parsed_data = json.loads(data_content)
                    logger.info(f"Successfully parsed JSON data: {json.dumps(parsed_data, indent=2)}")
                    return parsed_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse data content as JSON: {e}")
                    logger.error(f"Data content was: {data_content}")
                    return {
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: Invalid JSON in data field: {str(e)}"
                        }
                    }
            
            # If no data content found, try parsing the entire response as JSON (fallback)
            try:
                parsed_response = json.loads(response_text)
                logger.info(f"Parsed entire response as JSON: {json.dumps(parsed_response, indent=2)}")
                return parsed_response
            except json.JSONDecodeError:
                logger.error(f"Could not parse SSE response. Full response: {response_text}")
                return {
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: Could not extract valid JSON from SSE response"
                    }
                }
                
        except Exception as e:
            logger.error(f"Unexpected error parsing SSE response: {e}")
            return {
                "error": {
                    "code": -32603,
                    "message": f"Internal error parsing response: {str(e)}"
                }
            }

    def _send_request(self, method: str, params: Dict = None):
        """Helper to send a JSON-RPC request over the persistent session."""
        payload = {"jsonrpc": "2.0", "method": method, "id": str(uuid.uuid4())}
        if params:
            payload["params"] = params
        
        logger.info(f"Sending MCP request to URL: {self._server_url}")
        logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = self._session.post(
                self._server_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            # Debug logging
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            # Get the raw response text
            response_text = response.text.strip()
            logger.info(f"Raw response (first 500 chars): {response_text[:500]}")
            
            # Parse the SSE response
            response_data = self._parse_sse_response(response_text)
            
            # Check for errors in the parsed response
            if 'error' in response_data:
                error_info = response_data['error']
                error_message = error_info.get('message', 'Unknown error')
                error_code = error_info.get('code', -32603)
                raise RuntimeError(f"MCP server error for method '{method}' (code {error_code}): {error_message}")
                
            return response_data.get('result', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed for method '{method}': {e}")
            raise RuntimeError(f"Failed to communicate with MCP server during '{method}' call: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in _send_request for method '{method}': {e}")
            raise RuntimeError(f"Unexpected error during MCP '{method}' call: {str(e)}") from e

    def connect(self):
        """
        Establishes the connection by sending the required handshake message.
        """
        logger.info("Connecting to MCP server with handshake...")
        connect_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "langchain-python-client",
                "version": "0.1.0"
            }
        }
        result = self._send_request(method="initialize", params=connect_params)
        logger.info("MCP handshake successful.")
        return result

    def list_tools(self) -> Dict:
        """Fetches the list of available tools from the server."""
        logger.info("Fetching tools from MCP server...")
        return self._send_request(method="tools/list")

    def call_tool(self, tool_name: str, arguments: Dict, context: Dict = None) -> Dict:
        """Calls a specific tool on the server."""
        logger.info(f"Calling MCP Tool '{tool_name}' with args: {arguments}")
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        # Add context if provided
        if context:
            params["context"] = context
        return self._send_request(method="tools/call", params=params)

    def close(self):
        """Closes the session."""
        logger.info("Closing MCP client session.")
        self._session.close()
        
# --- LANGCHAIN TOOL (unchanged from your working version) ---
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

# --- AGENT SETUP (unchanged from your working version) ---
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

        # Use the basic approach that works with older LangChain versions
        return create_react_agent(llm, tools)
    except Exception as e:
        logger.error(f"Error setting up agent: {e}")
        # Re-raise to be caught by the chat endpoint
        raise

# --- CHAT ENDPOINT (unchanged from your working version) ---
@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    """Main chat endpoint that creates a stateful client for the conversation."""
    client_id = fastapi_request.client.host
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
    if not ds_metadata:
        raise HTTPException(status_code=400, detail="Datasource not initialized.")

    logger.info(f"Chat request from client {client_id}: '{request.message}'")
    
    # The endpoint needs to match your Express server's basePath
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

# --- YOUR WORKING DATASOURCE LOGIC (restored exactly as it was) ---

def get_tableau_username(client_id): 
    return os.environ['TABLEAU_USER']

def generate_jwt(tableau_username):
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
        "scp": ["tableau:rest_api:query", "tableau:rest_api:metadata", "tableau:content:read"]
    }
    headers = {"kid": secret_id, "iss": client_id}
    return jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)

def tableau_signin_with_jwt(tableau_username):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username)
    url = f"{domain}/api/{api_version}/auth/signin"
    payload = {"credentials": {"jwt": jwt_token, "site": {"contentUrl": site}}}
    resp = requests.post(url, json=payload, headers={'Accept': 'application/json'}, timeout=20)
    resp.raise_for_status()
    creds = resp.json()['credentials']
    return creds['token'], creds['site']['id']

def lookup_published_luid_by_name(name, tableau_username):
    token, site_id = tableau_signin_with_jwt(tableau_username)
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.26')
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {"X-Tableau-Auth": token, "Accept": "application/json"}
    params = {"filter": f"name:eq:{name}"}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources: 
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found.")
    return datasources[0]["id"], datasources[0]

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    client_id = request.client.host
    first_name, _ = next(iter(body.datasources.items()))
    tableau_username = get_tableau_username(client_id)
    published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username)
    DATASOURCE_METADATA_STORE[client_id] = {
        "name": first_name, 
        "luid": published_luid, 
        "projectName": full_metadata.get("projectName"), 
        "description": full_metadata.get("description"), 
        "owner_name": full_metadata.get("owner", {}).get("name")
    }
    return {
        "status": "ok", 
        "selected_datasource_name": first_name, 
        "selected_datasource_luid": published_luid
    }

@app.get("/", include_in_schema=False)
async def home(): 
    return FileResponse('static/index.html')

if os.path.exists("static"): 
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__" and not os.environ.get("VERCEL"):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))