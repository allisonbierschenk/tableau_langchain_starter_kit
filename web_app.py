from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import traceback
import jwt
import datetime
import uuid
from typing import Optional, Dict, Any

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

# User database - replace with your actual user management system
USERS_DB = {
    "john_doe": {
        "username": "john_doe",
        "isRetailer": True,
        "regions": ["West", "South"],
        "tableau_user": "john.doe@company.com",
        "display_name": "John Doe"
    },
    "jane_smith": {
        "username": "jane_smith", 
        "isRetailer": False,
        "regions": ["East", "North"],
        "tableau_user": "jane.smith@company.com",
        "display_name": "Jane Smith"
    },
    "demo_user": {
        "username": "demo_user",
        "isRetailer": True,
        "regions": ["West"],
        "tableau_user": os.environ.get('TABLEAU_USER', 'demo@company.com'),
        "display_name": "Demo User"
    }
}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class DataSourcesRequest(BaseModel):
    datasources: dict  # { "name": "federated_id" }

class UserRequest(BaseModel):
    userId: str

DATASOURCE_LUID_STORE = {}
DATASOURCE_METADATA_STORE = {}
USER_SESSIONS = {}  # Store user session data

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user data by user ID"""
    return USERS_DB.get(user_id)

def get_user_regions(user_id: str) -> list:
    """Get user regions based on user ID"""
    user = get_user_by_id(user_id)
    return user.get('regions', ['West']) if user else ['West']

def get_tableau_username(user_id: str) -> str:
    """Get Tableau username based on user ID"""
    user = get_user_by_id(user_id)
    return user.get('tableau_user', os.environ.get('TABLEAU_USER', 'demo@company.com')) if user else os.environ.get('TABLEAU_USER', 'demo@company.com')

def is_retailer_user(user_id: str) -> bool:
    """Check if user is a retailer (should use Pulse interface)"""
    user = get_user_by_id(user_id)
    return user.get('isRetailer', False) if user else False

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
    print(f"\n🔐 JWT generated for Tableau API (user: {tableau_username})")
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
    resp.raise_for_status()
    token = resp.json()['credentials']['token']
    site_id = resp.json()['credentials']['site']['id']
    print(f"✅ Tableau REST API token acquired for {tableau_username}")
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
    params = {
        "filter": f"name:eq:{name}"
    }

    print(f"\n🔍 Looking up datasource by name using filter: '{name}'")
    resp = requests.get(url, headers=headers, params=params)
    print(f"🔁 Datasource fetch response code: {resp.status_code}")
    resp.raise_for_status()

    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources:
        print(f"❌ No match found for datasource name: {name}")
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found on Tableau Server.")

    ds = datasources[0]
    print(f"✅ Match found ➜ Name: {ds.get('name')} ➜ LUID: {ds.get('id')}")
    return ds["id"], ds

@app.get("/")
def home():
    """Serve the main page with user selection"""
    return FileResponse('static/index.html')

@app.get("/analyze/{userId}")
def analyze_page(userId: str):
    """Serve the analysis page based on user type"""
    user = get_user_by_id(userId)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{userId}' not found")
    
    # Store user session
    USER_SESSIONS[userId] = user
    
    if user.get('isRetailer'):
        # Serve Pulse interface
        return FileResponse('static/pulse.html')
    else:
        # Serve Web Authoring interface  
        return FileResponse('static/web_authoring.html')

@app.get("/api/user/{userId}")
def get_user(userId: str):
    """API endpoint to get user data"""
    user = get_user_by_id(userId)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{userId}' not found")
    return user

@app.get("/api/users")
def list_users():
    """API endpoint to list all users"""
    return {"users": list(USERS_DB.values())}

@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest, userId: str = Query(...)):
    """Receive datasources with user context"""
    user = get_user_by_id(userId)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{userId}' not found")
    
    print(f"\n📥 /datasources request from user: {userId} ({user.get('display_name')})")
    
    if not body.datasources:
        raise HTTPException(status_code=400, detail="No data sources provided")

    print("📦 Datasources received from Tableau Extensions API (name ➜ federated ID):")
    for name, fed_id in body.datasources.items():
        print(f"   - Name: '{name}' ➜ Federated ID: '{fed_id}'")

    first_name, _ = next(iter(body.datasources.items()))
    print(f"\n🎯 Targeting first datasource: '{first_name}'")

    user_regions = get_user_regions(userId)
    tableau_username = get_tableau_username(userId)
    print(f"👤 Tableau username: {tableau_username}")
    print(f"🌍 User regions: {user_regions}")

    published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username, user_regions)

    # Use userId as the key instead of client IP
    DATASOURCE_LUID_STORE[userId] = published_luid
    DATASOURCE_METADATA_STORE[userId] = {
        "name": first_name,
        "luid": published_luid,
        "projectName": full_metadata.get("projectName"),
        "description": full_metadata.get("description"),
        "uri": full_metadata.get("contentUrl"),
        "owner": full_metadata.get("owner", {}),
        "user": user
    }

    print(f"✅ Stored published LUID for user {userId}: {published_luid}\n")
    return {"status": "ok", "selected_luid": published_luid, "user": user}

def setup_agent(userId: str):
    """Setup agent with user context"""
    datasource_luid = DATASOURCE_LUID_STORE.get(userId)
    ds_metadata = DATASOURCE_METADATA_STORE.get(userId)
    user = get_user_by_id(userId)
    
    if not user:
        raise RuntimeError(f"❌ User '{userId}' not found.")
    if not datasource_luid:
        raise RuntimeError("❌ No Tableau datasource LUID available.")

    tableau_username = get_tableau_username(userId)
    user_regions = get_user_regions(userId)

    print(f"\n🤖 Setting up agent for user {userId} ({user.get('display_name')})")
    print(f"📊 Datasource LUID: {datasource_luid}")
    print(f"📘 Datasource Name: {ds_metadata.get('name') if ds_metadata else 'Unknown'}")
    print(f"🏪 Is Retailer: {user.get('isRetailer', False)}")

    if ds_metadata:
        agent_identity = build_agent_identity(ds_metadata)
        
        # Customize prompt based on user type
        if user.get('isRetailer'):
            # Enhanced prompt for retail users focusing on Pulse metrics
            agent_prompt = build_agent_system_prompt(
                agent_identity, 
                ds_metadata.get("name", "this Tableau datasource"),
                custom_instructions="""
                IMPORTANT: This user is a retailer. Focus on:
                - Business KPIs and metrics relevant to retail operations
                - Sales performance, inventory levels, customer insights
                - Use Tableau Pulse tools when available for metric insights
                - Provide actionable business intelligence for retail decision-making
                """
            )
        else:
            # Standard prompt for analyst users
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

@app.get("/index.html")
def static_index():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest, userId: str = Query(...)) -> ChatResponse:
    """Chat endpoint with user context"""
    user = get_user_by_id(userId)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{userId}' not found")
        
    if userId not in DATASOURCE_LUID_STORE or not DATASOURCE_LUID_STORE[userId]:
        raise HTTPException(
            status_code=400,
            detail="Datasource not initialized yet. Please wait for datasource detection to complete."
        )
    try:
        print(f"\n💬 Chat request from user {userId} ({user.get('display_name')})")
        print(f"🏪 User type: {'Retailer (Pulse)' if user.get('isRetailer') else 'Analyst (Web Authoring)'}")
        print(f"🧠 Message: {request.message}")
        
        agent = setup_agent(userId)
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