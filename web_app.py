# web_app.py - FINAL CORRECTED VERSION
# This version fixes the streaming format mismatch.

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import traceback
import jwt
import datetime
import uuid
import json
import logging

from langsmith import Client
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tableau.tools.simple_datasource_qa import initialize_simple_datasource_qa
from utilities.prompt import build_agent_identity, build_agent_system_prompt

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# This is optional but recommended for debugging with LangSmith
langsmith_client = Client()
config = {"configurable": {"thread_id": "tableau-thread"}}

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
        "iss": client_id, "sub": tableau_username, "aud": "tableau", "jti": str(uuid.uuid4()),
        "exp": now + datetime.timedelta(minutes=5),
        "scp": ["tableau:rest_api:read", "tableau:datasources:query", "tableau:content:read"],
        "Region": user_regions
    }
    headers = {"kid": secret_id, "iss": client_id}
    return jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)

def tableau_signin_with_jwt(tableau_username, user_regions):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username, user_regions)
    url = f"{domain}/api/{api_version}/auth/signin"
    payload = {"credentials": {"jwt": jwt_token, "site": {"contentUrl": site}}}
    resp = requests.post(url, json=payload, headers={'Accept': 'application/json'})
    resp.raise_for_status()
    creds = resp.json()['credentials']
    return creds['token'], creds['site']['id']

def lookup_published_luid_by_name(name, tableau_username, user_regions):
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
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found.")
    ds = datasources[0]
    return ds["id"], ds

@app.get("/")
def home():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    client_id = request.client.host
    try:
        if not body.datasources:
            raise HTTPException(status_code=400, detail="No data sources provided")

        first_name = next(iter(body.datasources.keys()))
        user_regions = get_user_regions(client_id)
        tableau_username = get_tableau_username(client_id)
        
        published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username, user_regions)
        
        DATASOURCE_LUID_STORE[client_id] = published_luid
        DATASOURCE_METADATA_STORE[client_id] = {
            "name": first_name, "luid": published_luid,
            "description": full_metadata.get("description", "")
        }
        return {"status": "ok", "selected_luid": published_luid}
    except Exception as e:
        logger.error(f"Datasource initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request):
    client_id = fastapi_request.client.host
    if client_id not in DATASOURCE_LUID_STORE:
        raise HTTPException(status_code=400, detail="Datasource not initialized.")
    
    def sse_format(event_name: str, data: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

    async def event_stream():
        try:
            ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
            datasource_luid = DATASOURCE_LUID_STORE.get(client_id)
            tableau_username = get_tableau_username(client_id)
            
            agent_identity = build_agent_identity(ds_metadata)
            system_prompt = build_agent_system_prompt(agent_identity, ds_metadata.get("name"))
            
            analyze_datasource = initialize_simple_datasource_qa(
                domain=os.environ['TABLEAU_DOMAIN_FULL'],
                site=os.environ['TABLEAU_SITE'],
                jwt_client_id=os.environ['TABLEAU_JWT_CLIENT_ID'],
                jwt_secret_id=os.environ['TABLEAU_JWT_SECRET_ID'],
                jwt_secret=os.environ['TABLEAU_JWT_SECRET'],
                tableau_user=tableau_username,
                datasource_luid=datasource_luid,
                tooling_llm_model="gpt-4o-mini",
                model_provider="openai"
            )
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
            tools = [analyze_datasource]
            agent = create_react_agent(model=llm, tools=tools)

            messages = [
                ("system", system_prompt),
                ("user", request.message)
            ]

            full_content = ""
            yield sse_format("progress", {"message": "AI is thinking..."})

            # THE CRITICAL FIX IS HERE: `stream_mode="values"`
            async for chunk in agent.astream({"messages": messages}, config=config, stream_mode="values"):
                # The 'values' mode gives us chunks like {'agent': {'messages': [...]}}
                if "agent" in chunk and "messages" in chunk["agent"]:
                    latest_message = chunk["agent"]["messages"][-1]
                    new_content = latest_message.content
                    
                    # LangGraph streams the full content at each step, so we find what's new
                    if len(new_content) > len(full_content):
                        token = new_content[len(full_content):]
                        full_content = new_content
                        yield sse_format("token", {"token": token})

            yield sse_format("result", {"response": full_content})

        except Exception as e:
            logger.error(f"Error during chat stream: {e}", exc_info=True)
            yield sse_format("error", {"error": str(e)})
        finally:
            yield sse_format("done", {"message": "Stream complete"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)