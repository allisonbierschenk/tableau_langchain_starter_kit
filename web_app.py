from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import traceback
import jwt
import datetime
import uuid

from langsmith import Client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tableau.tools.simple_datasource_qa import initialize_simple_datasource_qa
from utilities.prompt import build_agent_identity, build_agent_system_prompt

load_dotenv()

langsmith_client = Client()
config = {"run_name": "Tableau Langchain Web_App.py"}

app = FastAPI(title="Tableau AI Chat", description="Simple AI chat interface for Tableau data")
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class DataSourcesRequest(BaseModel):
    datasources: dict

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
    print("\n=== GENERATED JWT ===")
    print(jwt_token)
    print("=====================\n")
    return jwt_token

def tableau_signin_with_jwt(tableau_username, user_regions):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username, user_regions)
    print(f"Using JWT for Tableau sign-in: {jwt_token}")
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
    print(f"POST {url} with payload: {payload}")
    resp = requests.post(url, json=payload, headers=headers)
    print(f"REST API signin status: {resp.status_code}")
    print(f"REST API signin response: {resp.text}")
    resp.raise_for_status()
    token = resp.json()['credentials']['token']
    site_id = resp.json()['credentials']['site']['id']
    print(f"Received token: {token}")
    print(f"Site ID: {site_id}")
    return token, site_id

def lookup_published_datasource_metadata(datasource_name, tableau_username, user_regions):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    token, site_id = tableau_signin_with_jwt(tableau_username, user_regions)
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {
        "X-Tableau-Auth": token,
        "Accept": "application/json"
    }
    print(f"GET {url} with headers: {headers}")
    resp = requests.get(url, headers=headers)
    print(f"REST API datasources status: {resp.status_code}")
    if resp.status_code == 200:
        datasources = resp.json().get("datasources", {}).get("datasource", [])
        print(f"Datasources received: {datasources}")
        for ds in datasources:
            if ds.get("name", "").lower() == datasource_name.lower():
                print(f"Found datasource: {ds}")
                return {
                    "name": ds.get("name"),
                    "projectName": ds.get("projectName"),
                    "description": ds.get("description"),
                    "luid": ds.get("id"),
                    "uri": ds.get("contentUrl"),
                    "owner": ds.get("owner", {}),
                    "hasActiveWarning": None,
                    "extractLastRefreshTime": None,
                    "extractLastIncrementalUpdateTime": None,
                    "extractLastUpdateTime": None
                }
        print("Datasource not found in REST API response.")
    else:
        print(f"REST API error: {resp.status_code} - {resp.text}")
    return None

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    client_id = request.client.host
    print(f"Received /datasources request from client: {client_id}")
    if not body.datasources:
        raise HTTPException(status_code=400, detail="No data sources provided")
    first_name = next(iter(body.datasources.keys()))
    print(f"Datasource requested: {first_name}")
    user_regions = get_user_regions(client_id)
    tableau_username = get_tableau_username(client_id)
    print(f"User regions: {user_regions}, Tableau username: {tableau_username}")
    if not user_regions:
        raise HTTPException(status_code=403, detail="No region permissions found for this user.")
    ds_metadata = lookup_published_datasource_metadata(first_name, tableau_username, user_regions)
    if not ds_metadata:
        raise HTTPException(status_code=404, detail=f"Datasource not found: {first_name}.")
    luid = ds_metadata["luid"]
    DATASOURCE_LUID_STORE[client_id] = luid
    DATASOURCE_METADATA_STORE[client_id] = ds_metadata
    print(f"Stored LUID for client {client_id}: {luid}")
    return {"status": "ok", "selected_luid": luid}

def setup_agent(request: Request = None):
    client_id = request.client.host if request else None
    datasource_luid = DATASOURCE_LUID_STORE.get(client_id)
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
    tableau_username = get_tableau_username(client_id)
    user_regions = get_user_regions(client_id)
    if not datasource_luid:
        raise RuntimeError("No Tableau datasource LUID available.")
    if not tableau_username or not user_regions:
        raise RuntimeError("User context missing.")
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

@app.post("/chat")
def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    client_id = fastapi_request.client.host
    if client_id not in DATASOURCE_LUID_STORE or not DATASOURCE_LUID_STORE[client_id]:
        raise HTTPException(
            status_code=400,
            detail="Datasource not initialized yet. Please wait for datasource detection to complete."
        )
    try:
        agent = setup_agent(fastapi_request)
        messages = {"messages": [("user", request.message)]}
        response_text = ""
        for chunk in agent.stream(messages, config=config, stream_mode="values"):
            if 'messages' in chunk and chunk['messages']:
                latest_message = chunk['messages'][-1]
                if hasattr(latest_message, 'content'):
                    response_text = latest_message.content
        return ChatResponse(response=response_text)
    except Exception as e:
        print("Error in /chat endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
