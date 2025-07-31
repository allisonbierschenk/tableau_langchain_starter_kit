from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, create_model
import os
from dotenv import load_dotenv
import requests
import traceback
import uuid
from typing import Type, Any
import logging
import jwt # <-- Added
import datetime # <-- Added

# LangChain Imports
from langsmith import Client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initial Setup ---
load_dotenv()

# Initialize LangSmith client
try:
    langsmith_client = Client()
    config = {"run_name": "Tableau MCP Langchain Web_App.py"}
except Exception as e:
    logger.warning(f"LangSmith client initialization failed: {e}")
    langsmith_client = None
    config = {}

app = FastAPI(title="Tableau AI Chat", description="Simple AI chat interface for Tableau data")

# Mount static files
try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
    else:
        logger.warning("Static directory not found, skipping static file mounting")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class DataSourcesRequest(BaseModel):
    datasources: dict  # { "name": "federated_id" }


# --- In-Memory Stores ---
DATASOURCE_METADATA_STORE = {} # This will now hold the name, LUID, and other metadata.


# --- Tableau REST API Helper Functions (from old code) ---
def get_tableau_username(client_id):
    # Using a fixed username from environment variables for simplicity
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
        "scp": ["tableau:rest_api:query", "tableau:rest_api:metadata", "tableau:content:read"],
    }
    headers = {"kid": secret_id, "iss": client_id}
    jwt_token = jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)
    logger.info("JWT generated for Tableau API")
    return jwt_token

def tableau_signin_with_jwt(tableau_username):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username)
    url = f"{domain}/api/{api_version}/auth/signin"
    payload = { "credentials": { "jwt": jwt_token, "site": {"contentUrl": site} } }
    logger.info(f"Signing in to Tableau REST API as '{tableau_username}'")
    resp = requests.post(url, json=payload, headers={'Accept': 'application/json'}, timeout=20)
    resp.raise_for_status()
    creds = resp.json()['credentials']
    logger.info(f"Sign-in successful. Site ID: {creds['site']['id']}")
    return creds['token'], creds['site']['id']

def lookup_published_luid_by_name(name, tableau_username):
    token, site_id = tableau_signin_with_jwt(tableau_username)
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.26')
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {"X-Tableau-Auth": token, "Accept": "application/json"}
    params = {"filter": f"name:eq:{name}"}

    logger.info(f"Looking up datasource by name using filter: '{name}'")
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()

    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources:
        logger.error(f"No match found for datasource name: {name}")
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found on Tableau Server.")

    ds = datasources[0]
    logger.info(f"Match found -> Name: {ds.get('name')}, LUID: {ds.get('id')}")
    return ds["id"], ds


# --- Custom LangChain Tool for MCP Server ---
class MCPTool(BaseTool):
    """A tool to dynamically call any method on the remote MCP server."""
    server_url: str
    tool_name: str
    datasource_luid: str # <-- MODIFICATION: Add LUID to the tool's state

    def _run(self, **kwargs: Any) -> Any:
        """Use the tool and pass the datasource_luid as context."""
        logger.info(f"Calling MCP Tool '{self.tool_name}' for datasource '{self.datasource_luid}' with args: {kwargs}")

        # MODIFICATION: Add a 'context' key to the payload
        payload = {
            "jsonrpc": "2.0",
            "method": "callTool",
            "params": {
                "name": self.tool_name,
                "arguments": kwargs,
                "context": {
                    "datasource_luid": self.datasource_luid
                }
            },
            "id": str(uuid.uuid4())
        }

        try:
            response = requests.post(self.server_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'error' in result:
                error_message = result['error'].get('message', 'Unknown error')
                logger.error(f"MCP server returned an error: {error_message}")
                return f"Error from MCP server: {error_message}"
                
            return result.get('result', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call MCP server: {e}")
            return f"Error: Could not connect to the MCP tool server at {self.server_url}."


# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def home():
    """Serves the frontend application."""
    static_file_path = 'static/index.html'
    if os.path.exists(static_file_path):
        return FileResponse(static_file_path)
    return JSONResponse({"message": "Tableau AI Chat API is running."})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    """
    Receives datasource name from the Tableau Extension, looks up its LUID via Tableau REST API,
    and stores it for the user session.
    """
    try:
        client_id = request.client.host
        logger.info(f"Datasources request from client: {client_id}")
        
        if not body.datasources:
            raise HTTPException(status_code=400, detail="No data sources provided")

        first_name, _ = next(iter(body.datasources.items()))
        logger.info(f"Targeting first datasource: '{first_name}'")

        # --- MODIFICATION: Perform LUID lookup ---
        tableau_username = get_tableau_username(client_id)
        published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username)

        # Store all relevant metadata, including the LUID
        DATASOURCE_METADATA_STORE[client_id] = {
            "name": first_name,
            "luid": published_luid, # <-- The crucial LUID is now stored
            "projectName": full_metadata.get("projectName"),
            "description": full_metadata.get("description"),
            "owner_name": full_metadata.get("owner", {}).get("name")
        }

        logger.info(f"Stored datasource context for client {client_id}: Name='{first_name}', LUID='{published_luid}'")
        return {"status": "ok", "selected_datasource_name": first_name, "selected_datasource_luid": published_luid}
    
    except Exception as e:
        logger.error(f"Error in /datasources endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def setup_agent(request: Request = None):
    """Sets up the LangChain agent to use the MCP server tools with the correct datasource context."""
    try:
        client_id = request.client.host if request else None
        ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)

        if not ds_metadata or 'luid' not in ds_metadata:
            raise RuntimeError("No Tableau datasource context (including LUID) available. Please connect via the extension first.")

        datasource_luid = ds_metadata['luid'] # <-- Get the LUID from the store
        logger.info(f"Setting up agent for client {client_id} with Datasource LUID: {datasource_luid}")

        mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
        
        logger.info(f"Fetching tools from MCP server at {mcp_server_url}")
        try:
            list_tools_payload = {"jsonrpc": "2.0", "method": "listTools", "id": str(uuid.uuid4())}
            headers = {'Accept': 'application/json'}
            resp = requests.post(mcp_server_url, json=list_tools_payload, headers=headers, timeout=20)
            resp.raise_for_status()
            available_tools_data = resp.json()['result']['tools']
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not connect to MCP server to get tools: {e}")

        tools = []
        for tool_data in available_tools_data:
            tool_name = tool_data['name']
            tool_description = tool_data['description']
            
            # Dynamically create Pydantic model for tool arguments
            fields = {
                prop_name: (
                    str, # Default to string, can be enhanced
                    Field(..., description=prop_details.get('description', ''))
                )
                for prop_name, prop_details in tool_data.get('inputSchema', {}).get('properties', {}).items()
            }
            args_schema = create_model(f'{tool_name}Args', **fields) if fields else BaseModel

            # --- MODIFICATION: Pass the LUID to the tool instance ---
            mcp_tool = MCPTool(
                server_url=mcp_server_url,
                tool_name=tool_name,
                datasource_luid=datasource_luid, # <-- Inject the LUID here
                name=tool_name,
                description=tool_description,
                args_schema=args_schema
            )
            tools.append(mcp_tool)

        logger.info(f"Created {len(tools)} tools with LUID '{datasource_luid}': {[t.name for t in tools]}")
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        system_prompt = (f"You are a helpful assistant for analyzing data from the Tableau datasource named '{ds_metadata.get('name', 'N/A')}'. "
                         f"The datasource is described as: '{ds_metadata.get('description', 'No description available.')}'. "
                         "Use your available tools to answer questions.")

        return create_react_agent(model=llm, tools=tools, messages_modifier=system_prompt)
    
    except Exception as e:
        logger.error(f"Error setting up agent: {e}")
        raise


@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    """Main chat endpoint that invokes the LangChain agent."""
    try:
        client_id = fastapi_request.client.host
        if client_id not in DATASOURCE_METADATA_STORE:
            raise HTTPException(
                status_code=400,
                detail="Datasource not initialized. Please interact with the Tableau Extension first."
            )

        logger.info(f"Chat request from client {client_id}: '{request.message}'")
        
        agent = setup_agent(fastapi_request)
        messages = {"messages": [("user", request.message)]}
        response_text = ""
        
        for chunk in agent.stream(messages, config=config):
            # The streaming output format for create_react_agent may vary slightly.
            # This logic inspects the chunk for agent responses.
            if "agent" in chunk:
                agent_step = chunk["agent"]
                if hasattr(agent_step, 'messages') and agent_step.messages:
                   response_text = agent_step.messages[-1].content

        logger.info(f"Final response: {response_text}")
        return ChatResponse(response=response_text)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__" and not os.environ.get("VERCEL"):
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)