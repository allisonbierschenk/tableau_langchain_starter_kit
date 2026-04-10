"""
Admin Web Application for Tableau Cloud User and Group Management
Using HTTP MCP client for admin operations
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional

# Web UI Libraries
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# Load Environment
from dotenv import load_dotenv
load_dotenv()

# Import our Admin MCP integration
from utilities.admin_agent import admin_mcp_chat
from utilities.slack_mrkdwn import llm_markdown_to_slack_mrkdwn

# Request/Response models
class ChatRequest(BaseModel):
    """Accepts web UI (`message` + `history`) and Slack-style bodies (`text` + `messages`)."""

    message: Optional[str] = None
    text: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    messages: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def _require_user_text(self):
        if not (self.message or self.text or "").strip():
            raise ValueError("Provide 'message' or 'text' with the user's prompt.")
        return self

    def user_message(self) -> str:
        return (self.message or self.text or "").strip()

    def conversation_history(self) -> List[Dict[str, Any]]:
        if self.history:
            return self.history
        if self.messages:
            return self.messages
        return []

class ChatResponse(BaseModel):
    response: str
    tool_results: List[Dict] = []
    iterations: int = 0

# Create FastAPI app
app = FastAPI(
    title="Tableau Admin Portal",
    description="Comprehensive admin agent for managing Tableau Cloud users, groups, permissions, jobs, and operations via tableau-mcp server"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_slack_bearer_token(authorization: Optional[str]) -> None:
    """Validate bearer token for Slack bot API access."""
    expected_token = os.getenv("ADMIN_API_BEARER_TOKEN", "")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_BEARER_TOKEN is not configured on the API service.",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header.")

    provided_token = authorization.split(" ", 1)[1].strip()
    if provided_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid bearer token.")

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
        "service": "tableau-admin-portal",
        "version": "2.0.0",
        "capabilities": [
            "user-management",
            "group-management",
            "permissions-management",
            "job-management",
            "operations-management",
            "lineage-queries",
            "stale-content-reports",
            "workbook-archiving"
        ]
    }

@app.get("/system-prompt")
async def get_system_prompt():
    """Get admin system prompt for display in UI."""
    from utilities.admin_agent import ADMIN_SYSTEM_PROMPT
    return {
        "systemPrompt": ADMIN_SYSTEM_PROMPT
    }

@app.post("/mcp-chat")
async def mcp_chat_endpoint(
    request: ChatRequest,
) -> ChatResponse:
    """Admin chat endpoint (non-streaming)."""
    try:
        print(f"🔐 Admin Chat request: {request.user_message()}")

        # Process with Admin MCP Agent
        result = await admin_mcp_chat(
            query=request.user_message(),
            conversation_history=request.conversation_history(),
        )

        return ChatResponse(
            response=result["response"],
            tool_results=result["tool_results"],
            iterations=result["iterations"]
        )

    except Exception as e:
        print(f"❌ Admin Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/slack-mcp-chat")
async def slack_mcp_chat_endpoint(
    request: ChatRequest,
    authorization: Optional[str] = Header(default=None),
) -> ChatResponse:
    """Slack bot endpoint using bearer token auth (no web session required)."""
    _validate_slack_bearer_token(authorization)

    try:
        result = await admin_mcp_chat(
            query=request.user_message(),
            conversation_history=request.conversation_history(),
        )

        return ChatResponse(
            response=llm_markdown_to_slack_mrkdwn(result["response"]),
            tool_results=result["tool_results"],
            iterations=result["iterations"],
        )
    except Exception as e:
        print(f"❌ Slack Admin Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp-chat-stream")
async def mcp_chat_stream_endpoint(
    request: ChatRequest,
):
    """Admin chat streaming endpoint with progress updates."""

    async def generate_stream():
        try:
            # Send initial progress
            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Admin agent initialized\", \"step\": \"init\"}}\n\n"

            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Connecting to admin MCP server...\", \"step\": \"connect\"}}\n\n"

            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Loading admin tools (users, groups, permissions, jobs, operations)...\", \"step\": \"load-tools\"}}\n\n"

            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Processing admin request...\", \"step\": \"processing\"}}\n\n"

            # Process with Admin MCP Agent
            result = await admin_mcp_chat(
                query=request.user_message(),
                conversation_history=request.conversation_history(),
            )

            # Send final result
            yield f"event: result\n"
            yield f"data: {json.dumps(result)}\n\n"

            yield f"event: done\n"
            yield f"data: {{\"message\": \"Admin operation complete\"}}\n\n"

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Admin stream error: {error_msg}")

            # Send error as a result
            yield f"event: result\n"
            yield f"data: {json.dumps({'response': f'Admin operation failed: {error_msg}', 'tool_results': [], 'iterations': 0, 'error': True})}\n\n"

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

    print("🔐 Starting Tableau Admin Portal")
    print("📊 Admin operations: Users, Groups, Permissions, Jobs, Operations")
    print("🔧 MCP Server: " + os.getenv("ADMIN_MCP_SERVER", "Not configured"))

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
