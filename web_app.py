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

# Initialize LangSmith client with error handling
try:
    langsmith_client = Client()
    config = {"run_name": "Tableau MCP Langchain Web_App.py"}
except Exception as e:
    logger.warning(f"LangSmith client initialization failed: {e}")
    langsmith_client = None
    config = {}

app = FastAPI(title="Tableau AI Chat", description="Simple AI chat interface for Tableau data")

# Mount static files with error handling for Vercel
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
DATASOURCE_METADATA_STORE = {}


# --- Custom LangChain Tool for MCP Server ---
class MCPTool(BaseTool):
    """A tool to dynamically call any method on the remote MCP server."""
    server_url: str
    tool_name: str

    def _run(self, **kwargs: Any) -> Any:
        """Use the tool."""
        logger.info(f"Calling MCP Tool '{self.tool_name}' with args: {kwargs}")

        # This is the JSON-RPC payload your MCP server expects
        payload = {
            "jsonrpc": "2.0",
            "method": "callTool",
            "params": {
                "name": self.tool_name,
                "arguments": kwargs,
            },
            "id": str(uuid.uuid4())
        }

        try:
            response = requests.post(self.server_url, json=payload, timeout=30)  # Reduced timeout for serverless
            response.raise_for_status()
            
            result = response.json()
            if 'error' in result:
                logger.error(f"MCP server returned an error: {result['error']}")
                return f"Error from MCP server: {result['error'].get('message', 'Unknown error')}"
                
            return result.get('result', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call MCP server: {e}")
            return f"Error: Could not connect to the MCP tool server at {self.server_url}."


# --- API Endpoints ---

@app.get("/")
async def home():
    """Root endpoint with fallback for missing static files"""
    try:
        if os.path.exists('static/index.html'):
            return FileResponse('static/index.html')
        else:
            return JSONResponse({
                "message": "Tableau AI Chat API",
                "status": "running",
                "endpoints": ["/health", "/chat", "/datasources"]
            })
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        return JSONResponse({
            "message": "Tableau AI Chat API",
            "status": "running",
            "error": str(e)
        })


@app.get("/index.html")
async def static_index():
    """Static index with fallback"""
    try:
        if os.path.exists('static/index.html'):
            return FileResponse('static/index.html')
        else:
            return JSONResponse({"error": "Static files not available in this deployment"})
    except Exception as e:
        return JSONResponse({"error": f"Could not serve static file: {str(e)}"})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "environment": "vercel" if os.environ.get("VERCEL") else "local",
        "langsmith_available": langsmith_client is not None
    }


@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    """Receives datasource information from the Tableau Extension."""
    try:
        client_id = request.client.host
        logger.info(f"Datasources request from client: {client_id}")
        
        if not body.datasources:
            raise HTTPException(status_code=400, detail="No data sources provided")

        first_name, _ = next(iter(body.datasources.items()))
        logger.info(f"Targeting first datasource: '{first_name}'")

        # Store basic metadata
        DATASOURCE_METADATA_STORE[client_id] = {"name": first_name}

        logger.info(f"Stored datasource context for client {client_id}: Name='{first_name}'")
        return {"status": "ok", "selected_datasource": first_name}
    
    except Exception as e:
        logger.error(f"Error in datasources endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def setup_agent(request: Request = None):
    """Sets up the LangChain agent to use the MCP server tools."""
    try:
        client_id = request.client.host if request else None
        ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)

        if not ds_metadata:
            raise RuntimeError("No Tableau datasource context available.")

        logger.info(f"Setting up agent for client {client_id}")
        logger.info(f"Datasource Context: {ds_metadata.get('name')}")

        # MCP INTEGRATION LOGIC
        mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
        
        # Get tools from MCP server
        logger.info(f"Fetching tools from MCP server at {mcp_server_url}")
        try:
            list_tools_payload = {"jsonrpc": "2.0", "method": "listTools", "id": str(uuid.uuid4())}
            resp = requests.post(mcp_server_url, json=list_tools_payload, timeout=20)  # Reduced timeout
            resp.raise_for_status()
            available_tools_data = resp.json()['result']['tools']
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not connect to MCP server to get tools: {e}")

        # Dynamically create LangChain tools
        tools = []
        for tool_data in available_tools_data:
            tool_name = tool_data['name']
            tool_description = tool_data['description']
            
            # Create Pydantic model for tool arguments
            fields = {}
            for prop_name, prop_details in tool_data.get('inputSchema', {}).get('properties', {}).items():
                prop_type = prop_details.get('type', 'string')
                # Map JSON schema types to Python types
                if prop_type == 'string':
                    python_type = str
                elif prop_type == 'integer':
                    python_type = int
                elif prop_type == 'boolean':
                    python_type = bool
                else:
                    python_type = str
                
                fields[prop_name] = (python_type, Field(..., description=prop_details.get('description', '')))
            
            if fields:
                args_schema = create_model(f'{tool_name}Args', **fields)
            else:
                args_schema = None

            # Create custom tool instance
            mcp_tool = MCPTool(
                server_url=mcp_server_url,
                tool_name=tool_name,
                name=tool_name,
                description=tool_description,
                args_schema=args_schema or BaseModel
            )
            tools.append(mcp_tool)

        logger.info(f"Created {len(tools)} tools: {[t.name for t in tools]}")

        # Import prompt utilities with fallback
        try:
            from utilities.prompt import build_agent_identity, build_agent_system_prompt
            agent_identity = build_agent_identity(ds_metadata)
            agent_prompt = build_agent_system_prompt(agent_identity, ds_metadata.get("name", "this Tableau datasource"))
        except ImportError:
            logger.warning("Could not import prompt utilities, using default prompt")
            agent_prompt = f"You are a helpful assistant for analyzing Tableau data from datasource: {ds_metadata.get('name', 'unknown')}"
        
        # Initialize LLM
        try:
            llm = ChatOpenAI(model="gpt-4", temperature=0)  # Changed from gpt-4.1 to gpt-4
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI: {e}")
            raise RuntimeError(f"Could not initialize OpenAI model: {e}")

        # Create agent
        return create_react_agent(model=llm, tools=tools, prompt=agent_prompt)
    
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

        logger.info(f"Chat request from client {client_id}")
        logger.info(f"Message: {request.message}")
        
        agent = setup_agent(fastapi_request)
        messages = {"messages": [("user", request.message)]}
        response_text = ""
        
        # Stream the agent response
        for chunk in agent.stream(messages, config=config, stream_mode="values"):
            if 'messages' in chunk and chunk['messages']:
                latest_message = chunk['messages'][-1]
                if hasattr(latest_message, 'content'):
                    response_text = latest_message.content

        logger.info(f"Response: {response_text}")
        return ChatResponse(response=response_text)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# For Vercel, remove the if __name__ == "__main__" block or make it conditional
if __name__ == "__main__" and not os.environ.get("VERCEL"):
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)