# web_app.py (REVISED)

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import requests
import traceback
import jwt
import datetime
import uuid
import json
from typing import List, Dict

# Langchain imports for a more robust agent
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_tableau import TableauDatasource

# --- Load Environment & Initialize Clients ---
load_dotenv()
langsmith_client = Client()

# --- App Setup ---
app = FastAPI(title="Tableau AI Chat", description="Simple AI chat interface for Tableau data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- In-Memory Stores (keep as is) ---
DATASOURCE_LUID_STORE: Dict[str, str] = {}
DATASOURCE_METADATA_STORE: Dict[str, Dict] = {}
TABLEAU_TOOL_STORE: Dict[str, "TableauTools"] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

class DataSourcesRequest(BaseModel):
    datasources: dict

# --- JWT and Tableau Auth Functions (keep as is) ---
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
    headers = {"kid": secret_id, "iss": client_id}
    return jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)

def tableau_signin_with_jwt(tableau_username, user_regions):
    # This function remains unchanged
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username, user_regions)
    url = f"{domain}/api/{api_version}/auth/signin"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    payload = {"credentials": {"jwt": jwt_token, "site": {"contentUrl": site}}}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    token = resp.json()['credentials']['token']
    site_id = resp.json()['credentials']['site']['id']
    return token, site_id

def lookup_published_luid_by_name(name, tableau_username, user_regions):
    # This function remains unchanged
    token, site_id = tableau_signin_with_jwt(tableau_username, user_regions)
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {"X-Tableau-Auth": token, "Accept": "application/json"}
    params = {"filter": f"name:eq:{name}"}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources:
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found on Tableau Server.")
    ds = datasources[0]
    return ds["id"], ds


# --- NEW: Tableau Tooling Class ---
# This class will hold the Tableau connection and provide the tools for the agent.
class TableauTools:
    def __init__(self, datasource_luid: str, tableau_username: str):
        self.datasource = TableauDatasource(
            domain=os.environ['TABLEAU_DOMAIN_FULL'],
            site=os.environ['TABLEAU_SITE'],
            jwt_client_id=os.environ['TABLEAU_JWT_CLIENT_ID'],
            jwt_secret_id=os.environ['TABLEAU_JWT_SECRET_ID'],
            jwt_secret=os.environ['TABLEAU_JWT_SECRET'],
            tableau_api_version=os.environ['TABLEAU_API_VERSION'],
            tableau_user=tableau_username,
            datasource_luid=datasource_luid,
            model_provider="openai"
        )
        # Cache fields on initialization to avoid repeated API calls
        try:
            print("🗂️ Caching fields for datasource...")
            self.fields = self.datasource.get_fields()
            print(f"✅ Cached {len(self.fields)} fields.")
        except Exception as e:
            print(f"❌ Error caching fields: {e}")
            self.fields = []

    @tool
    def query_data(self, tql_query: str):
        """
        Executes a Tableau Query Language (TQL) query against the data source.
        Use this to answer questions about metrics, trends, and specific data points.
        Example TQL: "SELECT [Category], SUM([Sales]) FROM [Table] GROUP BY [Category] ORDER BY SUM([Sales]) DESC LIMIT 10"
        ALWAYS use correct field names from the 'list_available_fields' tool. If a query fails, check the field names and try again.
        """
        try:
            return self.datasource.query(tql_query)
        except Exception as e:
            return f"Query failed. Error: {str(e)}. Please check your TQL syntax and ensure all field names are correct. Use the 'list_available_fields' tool to verify field names."

    @tool
    def list_available_fields(self):
        """
        Lists all available fields (columns) in the current data source.
        Use this tool FIRST to understand the data structure or if you are unsure about a field name.
        This helps you construct accurate TQL queries for the 'query_data' tool.
        """
        if not self.fields:
            return "Could not retrieve fields. The data source might be unavailable."
        return self.fields

    def get_all_tools(self):
        return [self.query_data, self.list_available_fields]

# --- NEW: Agent Setup Function ---
def setup_agent(client_id: str):
    """Initializes and returns a new LangChain agent for a given client session."""
    if client_id not in TABLEAU_TOOL_STORE:
        raise RuntimeError("Tools for this session not initialized. Please ensure datasource is selected.")

    tableau_tools = TABLEAU_TOOL_STORE[client_id].get_all_tools()
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id, {})
    
    # Using the prompt from prompt.py
    from utilities.prompt import REACT_AGENT_SYSTEM_PROMPT
    prompt_template = PromptTemplate.from_template(REACT_AGENT_SYSTEM_PROMPT)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tableau_tools,
        prompt=prompt_template
    )

    # Create the agent executor
    return AgentExecutor(
        agent=agent,
        tools=tableau_tools,
        verbose=True, # Set to True for debugging agent steps in your server logs
        handle_parsing_errors=True
    )


# --- API Endpoints ---
@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    client_id = request.client.host
    print(f"\n📥 /datasources request from client: {client_id}")

    if not body.datasources:
        raise HTTPException(status_code=400, detail="No data sources provided")

    first_name, _ = next(iter(body.datasources.items()))
    print(f"\n🎯 Targeting first datasource: '{first_name}'")

    tableau_username = get_tableau_username(client_id)
    user_regions = get_user_regions(client_id)

    # Lookup and store datasource info
    published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username, user_regions)
    DATASOURCE_LUID_STORE[client_id] = published_luid
    DATASOURCE_METADATA_STORE[client_id] = {
        "name": first_name,
        "luid": published_luid,
        "description": full_metadata.get("description", "A dataset in Tableau."),
        "owner": full_metadata.get("owner", {})
    }

    # Initialize and store the tools for this client session
    TABLEAU_TOOL_STORE[client_id] = TableauTools(
        datasource_luid=published_luid,
        tableau_username=tableau_username
    )
    
    print(f"✅ Stored published LUID and initialized tools for client {client_id}\n")
    return {"status": "ok", "selected_luid": published_luid}


@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request):
    """
    Handles all chat requests using a unified, robust ReAct agent.
    This endpoint now streams the agent's thoughts and final response.
    """
    client_id = fastapi_request.client.host
    print(f"\n💬 Chat request from client {client_id}: '{request.message}'")

    if client_id not in DATASOURCE_LUID_STORE:
        raise HTTPException(status_code=400, detail="Datasource not initialized.")

    agent_executor = setup_agent(client_id)
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id, {})
    ds_name = ds_metadata.get("name", "the current datasource")

    # The input to the agent needs to be a dictionary
    agent_input = {
        "input": request.message,
        "chat_history": [ (msg['role'], msg['content']) for msg in request.history ],
        "datasource_name": ds_name,
        "datasource_description": ds_metadata.get("description", "")
    }

    # Use a streaming response to send back agent thoughts and final answer
    async def stream_generator():
        final_response = ""
        try:
            # Use astream_log for detailed thought process
            async for chunk in agent_executor.astream_log(agent_input, include_names=["ChatOpenAI"]):
                for op in chunk.ops:
                    if op["path"] == "/logs/ChatOpenAI/streamed_output_str":
                        token = op["value"]
                        # We only want the final answer, not the intermediate thoughts
                        # The final answer is what's generated in the last step
                        pass # We will collect the final answer at the end
            
            # After streaming logs, invoke once more to get the clean output
            response_dict = await agent_executor.ainvoke(agent_input)
            final_response = response_dict.get("output", "I could not determine a final answer.")

            # Send the final result
            yield f"event: result\ndata: {json.dumps({'response': final_response})}\n\n"

        except Exception as e:
            print("❌ Error during agent execution:")
            traceback.print_exc()
            error_message = f"An error occurred: {str(e)}"
            yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n"
        
        # Signal that the stream is complete
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# --- Static and Health Check Endpoints (keep as is) ---
@app.get("/")
def home():
    return FileResponse('static/index.html')

@app.get("/index.html")
def static_index():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)