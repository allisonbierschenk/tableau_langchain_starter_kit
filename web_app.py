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
    print(f"\n🔍 Looking up published LUID for datasource name: '{name}'")
    resp = requests.get(url, headers=headers)
    print(f"🔁 Datasources fetch response code: {resp.status_code}")
    resp.raise_for_status()
    datasources = resp.json().get("datasources", {}).get("datasource", [])
    for ds in datasources:
        print(f"🔎 Checking datasource: {ds.get('name')} ➜ LUID: {ds.get('id')}")
        if ds.get("name", "").lower() == name.lower():
            print(f"✅ Match found for '{name}' ➜ Published LUID: {ds['id']}")
            return ds["id"], ds
    print(f"❌ No match found for datasource name: {name}")
    raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found on Tableau Server.")

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    client_id = request.client.host
    print(f"\n📥 /datasources request from client: {client_id}")
    
    if not body.datasources:
        raise HTTPException(status_code=400, detail="No data sources provided")

    print("📦 Datasources received from Tableau Extensions API (name ➜ federated ID):")
    for name, fed_id in body.datasources.items():
        print(f"   - Name: '{name}' ➜ Federated ID: '{fed_id}'")

    # Use the first datasource
    first_name, _ = next(iter(body.datasources.items()))
    print(f"\n🎯 Targeting first datasource: '{first_name}'")

    user_regions = get_user_regions(client_id)
    tableau_username = get_tableau_username(client_id)
    print(f"👤 Tableau username: {tableau_username}")
    print(f"🌍 User regions: {user_regions}")

    # Look up the real LUID using the name
    published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username, user_regions)

    # Store the real LUID and metadata
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

@app.post("/chat")
def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    client_id = fastapi_request.client.host
    if client_id not in DATASOURCE_LUID_STORE or not DATASOURCE_LUID_STORE[client_id]:
        raise HTTPException(
            status_code=400,
            detail="Datasource not initialized yet. Please wait for datasource detection to complete."
        )
    try:
        print(f"\n💬 Chat request from client {client_id}")
        print(f"🧠 Message: {request.message}")
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
