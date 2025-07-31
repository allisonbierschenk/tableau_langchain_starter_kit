from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, create_model
import os
from dotenv import load_dotenv
import requests
import traceback
import uuid
from typing import Type, Any

# LangChain Imports
from langsmith import Client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool

# Local Imports
from utilities.prompt import build_agent_identity, build_agent_system_prompt

# --- Initial Setup ---
load_dotenv()
langsmith_client = Client()
config = {"run_name": "Tableau MCP Langchain Web_App.py"}
app = FastAPI(title="Tableau AI Chat", description="Simple AI chat interface for Tableau data")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        print(f"Calling MCP Tool '{self.tool_name}' with args: {kwargs}")

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
            response = requests.post(self.server_url, json=payload, timeout=120) # Added timeout
            response.raise_for_status()  # Raise an exception for bad status codes
            
            result = response.json()
            if 'error' in result:
                print(f"MCP server returned an error: {result['error']}")
                return f"Error from MCP server: {result['error'].get('message', 'Unknown error')}"
                
            return result.get('result', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            print(f"Failed to call MCP server: {e}")
            return f"Error: Could not connect to the MCP tool server at {self.server_url}."


# --- API Endpoints ---

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    """Receives datasource information from the Tableau Extension."""
    client_id = request.client.host
    print(f"\n📥 /datasources request from client: {client_id}")
    
    if not body.datasources:
        raise HTTPException(status_code=400, detail="No data sources provided")

    first_name, _ = next(iter(body.datasources.items()))
    print(f"\n🎯 Targeting first datasource: '{first_name}'")

    # Store basic metadata. The agent will use tools to get more details.
    DATASOURCE_METADATA_STORE[client_id] = {"name": first_name}

    print(f"✅ Stored datasource context for client {client_id}: Name='{first_name}'\n")
    return {"status": "ok", "selected_datasource": first_name}


def setup_agent(request: Request = None):
    """Sets up the LangChain agent to use the MCP server tools."""
    client_id = request.client.host if request else None
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)

    if not ds_metadata:
        raise RuntimeError("❌ No Tableau datasource context available.")

    print(f"\n🤖 Setting up agent for client {client_id}")
    print(f"📘 Datasource Context: {ds_metadata.get('name')}")

    # --- NEW MCP INTEGRATION LOGIC ---
    mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
    
    # 1. Ask the MCP server for its list of tools
    print(f"Fetching tools from MCP server at {mcp_server_url}")
    try:
        list_tools_payload = {"jsonrpc": "2.0", "method": "listTools", "id": str(uuid.uuid4())}
        resp = requests.post(mcp_server_url, json=list_tools_payload, timeout=30)
        resp.raise_for_status()
        available_tools_data = resp.json()['result']['tools']
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Could not connect to MCP server to get tools: {e}")

    # 2. Dynamically create a LangChain tool for each MCP tool
    tools = []
    for tool_data in available_tools_data:
        tool_name = tool_data['name']
        tool_description = tool_data['description']
        
        # Dynamically create the Pydantic model for the tool's arguments
        fields = {
            prop_name: (prop_details.get('type', 'string'), Field(..., description=prop_details.get('description')))
            for prop_name, prop_details in tool_data.get('inputSchema', {}).get('properties', {}).items()
        }
        args_schema = create_model(f'{tool_name}Args', **fields)

        # Create our custom tool instance
        mcp_tool = MCPTool(
            server_url=mcp_server_url,
            tool_name=tool_name,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema
        )
        tools.append(mcp_tool)

    print(f"Dynamically created {len(tools)} tools: {[t.name for t in tools]}")
    # --- END OF NEW LOGIC ---

    agent_identity = build_agent_identity(ds_metadata)
    agent_prompt = build_agent_system_prompt(agent_identity, ds_metadata.get("name", "this Tableau datasource"))
    
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # 3. Create the agent with the new dynamic tools
    return create_react_agent(model=llm, tools=tools, prompt=agent_prompt)


@app.get("/")
def home():
    return FileResponse('static/index.html')


@app.get("/index.html")
def static_index():
    return FileResponse('static/index.html')


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    """Main chat endpoint that invokes the LangChain agent."""
    client_id = fastapi_request.client.host
    if client_id not in DATASOURCE_METADATA_STORE:
        raise HTTPException(
            status_code=400,
            detail="Datasource not initialized. Please interact with the Tableau Extension first."
        )
    try:
        print(f"\n💬 Chat request from client {client_id}")
        print(f"🧠 Message: {request.message}")
        agent = setup_agent(fastapi_request)
        messages = {"messages": [("user", request.message)]}
        response_text = ""
        # Stream the agent response
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
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)