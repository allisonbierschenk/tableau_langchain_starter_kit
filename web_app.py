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

# --- FIXED MCP CLIENT ---
class MCPClient:
    """
    Fixed MCP client that properly implements the MCP protocol over HTTP.
    """
    def __init__(self, server_url: str):
        # Ensure the URL ends with the correct endpoint path
        if not server_url.endswith('/'):
            server_url += '/'
        # Add the MCP endpoint path (adjust based on your server's basePath)
        if not server_url.endswith('tableau-mcp'):
            server_url += 'tableau-mcp'
            
        self._server_url = server_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',  # Changed from x-ndjson
            'Accept': 'application/json',
            'MCP-Protocol-Version': '2024-11-05'  # Updated to proper MCP version
        })
        self._session_id = None
        logger.info(f"MCP Client initialized for {server_url}")

    def _send_request(self, method: str, params: Dict = None):
        """Send an MCP request using proper JSON-RPC 2.0 format."""
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        if params:
            payload["params"] = params
        
        logger.info(f"Sending MCP request: {method} with payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = self._session.post(
                self._server_url,
                json=payload,  # Use json parameter instead of data
                timeout=30
            )
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()
            logger.info(f"MCP response: {json.dumps(response_data, indent=2)}")
            
            if 'error' in response_data:
                error_message = response_data['error'].get('message', 'Unknown error')
                error_code = response_data['error'].get('code', -1)
                raise RuntimeError(f"MCP server error for method '{method}' (code {error_code}): {error_message}")
                
            return response_data.get('result', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP request failed for method '{method}': {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise RuntimeError(f"Failed to communicate with MCP server during '{method}' call.") from e

    def connect(self):
        """
        Initialize the MCP connection with proper handshake.
        """
        logger.info("Connecting to MCP server with proper handshake...")
        
        # Send initialize request first
        initialize_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "langchain-python-client",
                "version": "0.1.0"
            }
        }
        
        result = self._send_request(method="initialize", params=initialize_params)
        logger.info(f"Initialize response: {result}")
        
        # Send initialized notification (no response expected)
        try:
            self._send_request(method="notifications/initialized")
        except Exception as e:
            # Notifications don't return responses, so this might "fail" - that's OK
            logger.info(f"Initialized notification sent (response: {e})")
        
        logger.info("MCP handshake successful.")

    def list_tools(self) -> Dict:
        """Fetch available tools from the MCP server."""
        logger.info("Fetching tools from MCP server...")
        return self._send_request(method="tools/list")

    def call_tool(self, tool_name: str, arguments: Dict, context: Dict = None) -> Dict:
        """Call a specific tool on the MCP server."""
        logger.info(f"Calling MCP Tool '{tool_name}' with args: {arguments}")
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        # Add context if provided (some MCP servers use this)
        if context:
            params["context"] = context
            
        return self._send_request(method="tools/call", params=params)

    def close(self):
        """Close the session."""
        logger.info("Closing MCP client session.")
        self._session.close()

# --- LANGCHAIN TOOL (unchanged) ---
class MCPTool(BaseTool):
    """
    LangChain tool that uses the MCPClient.
    """
    client: MCPClient
    tool_name: str
    datasource_luid: str

    def _run(self, **kwargs: Any) -> Any:
        """Use the connected client to call the tool."""
        context = {"datasource_luid": self.datasource_luid}
        result = self.client.call_tool(self.tool_name, kwargs, context)
        
        # Handle different response formats
        if isinstance(result, dict):
            if 'content' in result:
                return result['content']
            elif 'text' in result:
                return result['text']
            else:
                return str(result)
        else:
            return str(result)

# --- AGENT SETUP (with better error handling) ---
def setup_agent(client: MCPClient, ds_metadata: Dict):
    """
    Sets up the LangChain agent using the pre-connected MCPClient.
    """
    try:
        datasource_luid = ds_metadata['luid']
        logger.info(f"Setting up agent with Datasource LUID: {datasource_luid}")

        # Fetch tools using the connected client
        tools_response = client.list_tools()
        available_tools_data = tools_response.get('tools', [])
        
        if not available_tools_data:
            logger.error(f"No tools returned from MCP server. Response: {tools_response}")
            raise RuntimeError("MCP server returned no available tools.")

        logger.info(f"Found {len(available_tools_data)} tools from MCP server")
        
        tools = []
        for tool_data in available_tools_data:
            tool_name = tool_data['name']
            logger.info(f"Processing tool: {tool_name}")
            
            # Create dynamic args schema from tool's input schema
            input_schema = tool_data.get('inputSchema', {})
            properties = input_schema.get('properties', {})
            
            fields = {}
            for p_name, p_details in properties.items():
                # Handle different property types
                prop_type = p_details.get('type', 'string')
                description = p_details.get('description', '')
                
                if prop_type == 'string':
                    fields[p_name] = (str, Field(..., description=description))
                elif prop_type == 'integer':
                    fields[p_name] = (int, Field(..., description=description))
                elif prop_type == 'boolean':
                    fields[p_name] = (bool, Field(..., description=description))
                else:
                    # Default to string for unknown types
                    fields[p_name] = (str, Field(..., description=description))
            
            args_schema = create_model(f'{tool_name}Args', **fields) if fields else BaseModel

            mcp_tool = MCPTool(
                client=client,
                tool_name=tool_name,
                datasource_luid=datasource_luid,
                name=tool_name,
                description=tool_data.get('description', ''),
                args_schema=args_schema
            )
            tools.append(mcp_tool)

        logger.info(f"Created {len(tools)} tools: {[t.name for t in tools]}")
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        system_prompt = (
            f"You are a helpful assistant for the Tableau datasource named '{ds_metadata.get('name', 'N/A')}'. "
            f"Description: '{ds_metadata.get('description', 'N/A')}'. "
            f"Use your available tools to answer questions about the data. "
            f"Always use the datasource LUID '{datasource_luid}' when calling tools."
        )

        return create_react_agent(model=llm, tools=tools, messages_modifier=system_prompt)
        
    except Exception as e:
        logger.error(f"Error setting up agent: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# --- CHAT ENDPOINT (with better error handling) ---
@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    """Main chat endpoint that creates a stateful client for the conversation."""
    client_id = fastapi_request.client.host
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
    if not ds_metadata:
        raise HTTPException(status_code=400, detail="Datasource not initialized.")

    logger.info(f"Chat request from client {client_id}: '{request.message}'")
    
    # Updated MCP server URL - make sure this matches your Heroku deployment
    mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/"
    mcp_client = MCPClient(server_url=mcp_server_url)

    try:
        # Connect and test the connection
        mcp_client.connect()
        
        # Set up the agent
        agent = setup_agent(mcp_client, ds_metadata)
        
        # Execute the chat
        messages = {"messages": [("user", request.message)]}
        response_text = ""
        
        try:
            for chunk in agent.stream(messages, config=config):
                if "agent" in chunk and hasattr(chunk["agent"], 'messages'):
                    response_text = chunk["agent"].messages[-1].content
        except Exception as agent_error:
            logger.error(f"Agent execution error: {agent_error}")
            logger.error(f"Agent traceback: {traceback.format_exc()}")
            # Return a more helpful error message
            response_text = f"I encountered an error while processing your request: {str(agent_error)}"

        logger.info(f"Final response: {response_text}")
        return ChatResponse(response=response_text)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
    finally:
        mcp_client.close()

# --- REST OF THE CODE (unchanged) ---

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
    url = f"{domain}/api/{site_id}/datasources"
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