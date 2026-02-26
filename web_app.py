"""
Simple MCP + LangChain web application for Tableau
Using HTTP MCP client like the e-bikes demo
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional

# Web UI Libraries
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load Environment
from dotenv import load_dotenv
load_dotenv()

# Import our HTTP MCP integration
from utilities.langchain_mcp import tableau_mcp_chat

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    tool_results: List[Dict] = []
    iterations: int = 0

class DataSourcesRequest(BaseModel):
    datasources: Dict[str, str]  # { "DataSource Name": "luid" }
    dashboard_objects: Optional[List[Dict[str, Any]]] = None  # [ { "id", "type", "name" } ] from Extensions API

# Session store for Extension API: datasources sent from dashboard (session_id -> { luids, first_luid })
DATASOURCE_SESSION_STORE: Dict[str, Dict] = {}

# Create FastAPI app
app = FastAPI(
    title="Tableau MCP Chat",
    description="Simple HTTP MCP integration for Tableau data analysis"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files - use absolute path for Railway
import pathlib
static_dir = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    index_path = static_dir / "index.html"
    return FileResponse(str(index_path))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tableau-mcp-chat",
        "version": "1.0.0"
    }

@app.get("/system-prompt")
async def get_system_prompt():
    """Get system prompt for display in UI - matches e-bikes demo pattern"""
    from utilities.langchain_mcp import TABLEAU_SYSTEM_PROMPT
    return {
        "systemPrompt": TABLEAU_SYSTEM_PROMPT
    }

def _dashboard_has_pulse_objects(dashboard_objects: Optional[List[Dict[str, Any]]]) -> bool:
    """True if any dashboard object type looks like a Pulse metric (Extensions API may expose 'pulse' or similar)."""
    if not dashboard_objects:
        return False
    for obj in dashboard_objects:
        t = (obj.get("type") or "").lower()
        if "pulse" in t or t == "pulse-metric":
            return True
    return False


@app.post("/datasources")
async def receive_datasources(request: Request, body: DataSourcesRequest):
    """Receive datasource map and optional dashboard objects from Tableau Extension."""
    session_id = request.headers.get("X-Session-ID") or request.client.host
    if not body.datasources:
        raise HTTPException(status_code=400, detail="datasources cannot be empty")
    first_name, first_id = next(iter(body.datasources.items()))
    has_pulse = _dashboard_has_pulse_objects(body.dashboard_objects)
    DATASOURCE_SESSION_STORE[session_id] = {
        "datasources": body.datasources,
        "first_luid": first_id,
        "first_name": first_name,
        "dashboard_has_pulse_objects": has_pulse,
        "dashboard_objects": body.dashboard_objects or [],
    }
    obj_summary = [(o.get("id"), o.get("type"), o.get("name")) for o in (body.dashboard_objects or [])]
    print(f"📥 /datasources: session={session_id}, sources={list(body.datasources.keys())}, datasource_luid={first_id}, has_pulse_objects={has_pulse}, dashboard_objects={obj_summary}")
    return {
        "status": "ok",
        "datasource": first_name,
        "datasource_luid": first_id,
        "dashboard_has_pulse_objects": has_pulse,
    }

def _get_session_context(session_id: str) -> Dict[str, Any]:
    """Return session context: first_name, dashboard_has_pulse_objects."""
    entry = DATASOURCE_SESSION_STORE.get(session_id) or {}
    return {
        "preferred_datasource_name": entry.get("first_name"),
        "dashboard_has_pulse_objects": bool(entry.get("dashboard_has_pulse_objects")),
    }


@app.post("/mcp-chat")
async def mcp_chat_endpoint(chat_request: ChatRequest, request: Request) -> ChatResponse:
    """MCP chat endpoint (non-streaming) - matches e-bikes demo pattern"""
    try:
        print(f"💬 MCP Chat request: {chat_request.message}")
        session_id = request.headers.get("X-Session-ID") or (request.client and request.client.host) or ""
        ctx = _get_session_context(session_id)
        result = await tableau_mcp_chat(
            query=chat_request.message,
            conversation_history=chat_request.history,
            preferred_datasource_name=ctx.get("preferred_datasource_name"),
            dashboard_has_pulse_objects=ctx.get("dashboard_has_pulse_objects", False),
        )
        
        return ChatResponse(
            response=result["response"],
            tool_results=result["tool_results"],
            iterations=result["iterations"]
        )
        
    except Exception as e:
        print(f"❌ MCP Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp-chat-stream")
async def mcp_chat_stream_endpoint(chat_request: ChatRequest, request: Request):
    """MCP chat streaming endpoint with user-friendly progress"""
    session_id = request.headers.get("X-Session-ID") or (request.client and request.client.host) or ""
    ctx = _get_session_context(session_id)
    
    async def generate_stream():
        try:
            # Send initial progress
            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Connection established\", \"step\": \"init\"}}\n\n"
            
            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Starting analysis...\", \"step\": \"analysis-start\"}}\n\n"
            
            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Connecting to Tableau MCP server...\", \"step\": \"mcp-connect\"}}\n\n"
            
            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Discovering available data analysis tools...\", \"step\": \"discover-tools\"}}\n\n"
            
            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Analyzing your question and planning data exploration...\", \"step\": \"planning\"}}\n\n"
            
            result = await tableau_mcp_chat(
                query=chat_request.message,
                conversation_history=chat_request.history,
                preferred_datasource_name=ctx.get("preferred_datasource_name"),
                dashboard_has_pulse_objects=ctx.get("dashboard_has_pulse_objects", False),
            )
            
            # Send final result
            yield f"event: result\n"
            yield f"data: {json.dumps(result)}\n\n"
            
            yield f"event: done\n"
            yield f"data: {{\"message\": \"Stream complete\"}}\n\n"
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Stream error: {error_msg}")
            
            # Send error as a result so the UI can display it properly
            yield f"event: result\n"
            yield f"data: {json.dumps({'response': f'Error: {error_msg}', 'tool_results': [], 'iterations': 0, 'error': True})}\n\n"
            
            yield f"event: done\n"
            yield f"data: {{\"message\": \"Stream complete\"}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Tableau MCP Chat Server")
    print("📊 Using HTTP MCP client for hosted server")
    
    port = int(os.getenv("PORT", 8000))
    # Disable reload in production (Railway sets RAILWAY_ENVIRONMENT)
    reload = os.getenv("RAILWAY_ENVIRONMENT") is None
    
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )
