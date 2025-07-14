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

# Import your existing components
from langsmith import Client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tableau.tools.simple_datasource_qa import initialize_simple_datasource_qa
from utilities.prompt import build_agent_identity, build_agent_system_prompt

# Load environment variables
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

def generate_jwt():
    client_id = os.environ['TABLEAU_JWT_CLIENT_ID']
    secret_id = os.environ['TABLEAU_JWT_SECRET_ID']
    secret_value = os.environ['TABLEAU_JWT_SECRET']
    username = os.environ['TABLEAU_USER']

    now = datetime.datetime.now(datetime.UTC)
    payload = {
        "iss": client_id,
        "sub": username,
        "aud": "tableau",
        "jti": str(uuid.uuid4()),
        "exp": now + datetime.timedelta(minutes=5),
        "scp": ["tableau:rest_api:query", "tableau:rest_api:metadata", "tableau:content:read"],
        "Region": ['South', 'East']
    }
    headers = {
        "kid": secret_id,
        "iss": client_id
    }
    jwt_token = jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)
    return jwt_token

def tableau_signin_with_jwt():
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']

    jwt_token = generate_jwt()
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
    try:
        resp = requests.post(url, json=payload, headers=headers)
        print("REST API signin status:", resp.status_code)
        print("REST API signin response:", resp.text)
        resp.raise_for_status()
        token = resp.json()['credentials']['token']
        site_id = resp.json()['credentials']['site']['id']
        return token, site_id
    except Exception as e:
        print("Exception during REST API signin:")
        traceback.print_exc()
        raise RuntimeError("Tableau REST API signin failed. Check your JWT and site.")

def log_all_published_datasources():
    """
    Use Tableau Metadata API (GraphQL) to log all published data source names.
    """
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    try:
        token, site_id = tableau_signin_with_jwt()
    except Exception as e:
        print("Failed to sign in with JWT.")
        return None

    gql_url = f"{domain}/api/metadata/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        'Accept': 'application/json'
    }
    query = {
        "query": """
        query {
            publishedDatasources {
                name
                luid
                projectName
            }
        }
        """
    }
    try:
        resp = requests.post(gql_url, json=query, headers=headers)
        print("Metadata API status (all datasources):", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            datasources = data.get("data", {}).get("publishedDatasources", [])
            print("=== Published Datasources ===")
            for ds in datasources:
                print(f"Name: '{ds['name']}', LUID: {ds['luid']}, Project: {ds.get('projectName')}")
            print("=============================")
            return datasources
        else:
            print(f"Metadata API error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print("Exception during Metadata API call (all datasources):")
        traceback.print_exc()
    return None

def lookup_published_datasource_metadata(datasource_name):
    """
    Use Tableau Metadata API (GraphQL) to get metadata for a published data source by name.
    """
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    try:
        token, site_id = tableau_signin_with_jwt()
    except Exception as e:
        print("Failed to sign in with JWT.")
        return None

    gql_url = f"{domain}/api/metadata/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        'Accept': 'application/json'
    }
    query = {
        "query": f"""
        query Find_Datasources_LUID {{
            publishedDatasources(filter: {{ name: "{datasource_name}" }}) {{
                name
                projectName
                description
                luid
                uri
                owner {{
                    username
                    name
                    email
                }}
                hasActiveWarning
                extractLastRefreshTime
                extractLastIncrementalUpdateTime
                extractLastUpdateTime
            }}
        }}
        """
    }
    try:
        resp = requests.post(gql_url, json=query, headers=headers)
        print(f"Metadata API status (filtered for '{datasource_name}'):", resp.status_code)
        print("Metadata API response (filtered):", resp.text)
        if resp.status_code == 200:
            data = resp.json()
            datasources = data.get("data", {}).get("publishedDatasources", [])
            if datasources:
                print(f"Found datasource with name '{datasource_name}':", datasources[0])
                return datasources[0]
            else:
                print(f"No published data source found with the name '{datasource_name}'.")
        else:
            print(f"Metadata API error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print("Exception during Metadata API call (filtered):")
        traceback.print_exc()
    return None

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    client_id = request.client.host
    if not body.datasources:
        raise HTTPException(status_code=400, detail="No data sources provided")
    first_name = next(iter(body.datasources.keys()))
    print(f"Received data source request for: '{first_name}' from client: {client_id}")
    try:
        # Log all published datasources for debugging
        log_all_published_datasources()
        ds_metadata = lookup_published_datasource_metadata(first_name)
        if not ds_metadata:
            print(f"Datasource '{first_name}' not found after logging all datasources.")
            raise HTTPException(
                status_code=404,
                detail=f"Datasource not found: {first_name}."
            )
        luid = ds_metadata["luid"]
        print(f"Storing LUID '{luid}' for client '{client_id}'")
        DATASOURCE_LUID_STORE[client_id] = luid
        DATASOURCE_METADATA_STORE[client_id] = ds_metadata
        return {"status": "ok", "selected_luid": luid}
    except Exception as e:
        print("Exception in receive_datasources endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error looking up datasource: {str(e)}")

def setup_agent(request: Request = None):
    client_id = request.client.host if request else None
    datasource_luid = DATASOURCE_LUID_STORE.get(client_id) if client_id else os.environ.get('DATASOURCE_LUID', '')
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id) if client_id else None

    print(f"Using datasource LUID: {datasource_luid}")
    if not datasource_luid:
        raise RuntimeError("No Tableau datasource LUID available.")

    # Build dynamic prompt
    if ds_metadata:
        agent_identity = build_agent_identity(ds_metadata)
        agent_prompt = build_agent_system_prompt(agent_identity, ds_metadata.get("name", "this Tableau datasource"))
    else:
        from utilities.prompt import AGENT_SYSTEM_PROMPT
        agent_prompt = AGENT_SYSTEM_PROMPT  # fallback

    analyze_datasource = initialize_simple_datasource_qa(
        domain=os.environ['TABLEAU_DOMAIN_FULL'],
        site=os.environ['TABLEAU_SITE'],
        jwt_client_id=os.environ['TABLEAU_JWT_CLIENT_ID'],
        jwt_secret_id=os.environ['TABLEAU_JWT_SECRET_ID'],
        jwt_secret=os.environ['TABLEAU_JWT_SECRET'],
        tableau_api_version=os.environ['TABLEAU_API_VERSION'],
        tableau_user=os.environ['TABLEAU_USER'],
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
