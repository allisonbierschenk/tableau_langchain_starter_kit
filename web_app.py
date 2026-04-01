"""
Admin Web Application for Tableau Cloud User and Group Management
Using HTTP MCP client for admin operations
"""

import os
import json
import asyncio
from typing import List, Dict, Any

# Web UI Libraries
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# Load Environment
from dotenv import load_dotenv
load_dotenv()

# Import our Admin MCP integration
from utilities.admin_agent import admin_mcp_chat
from utilities.tableau_auth import (
    authenticate_with_tableau,
    signout_from_tableau,
    verify_admin_access,
    TableauAuthToken
)

# Request/Response models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str
    user_name: str = ""
    site_role: str = ""
    is_admin: bool = False

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    tool_results: List[Dict] = []
    iterations: int = 0

# Create FastAPI app
app = FastAPI(
    title="Tableau Admin Portal",
    description="Admin agent for managing Tableau Cloud users and groups via MCP"
)

# Session secret key - in production, use a secure random key from environment
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-to-a-secure-random-key-in-production")

# Add session middleware (must be added before CORS)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication helper functions
async def get_current_user(request: Request) -> TableauAuthToken:
    """
    Dependency to get the current authenticated user from session

    Raises:
        HTTPException: If user is not authenticated or session expired
    """
    auth_data = request.session.get("tableau_auth")

    if not auth_data:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login with your Tableau credentials."
        )

    try:
        auth_token = TableauAuthToken.from_dict(auth_data)

        # Check if token expired
        if auth_token.is_expired():
            request.session.clear()
            raise HTTPException(
                status_code=401,
                detail="Session expired. Please login again."
            )

        # Check if user has admin access
        if not auth_token.is_admin():
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Admin role required (you have: {auth_token.site_role})"
            )

        return auth_token

    except Exception as e:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Invalid session. Please login again.")

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
        "version": "1.0.0",
        "capabilities": [
            "user-management",
            "group-management"
        ]
    }

@app.post("/login")
async def login(request: Request, login_data: LoginRequest) -> LoginResponse:
    """
    Authenticate with Tableau credentials (username/password)

    Returns:
        LoginResponse with authentication result
    """
    try:
        # Authenticate with Tableau
        auth_token = await authenticate_with_tableau(
            username=login_data.username,
            password=login_data.password
        )

        # Check if user has admin access
        if not auth_token.is_admin():
            return LoginResponse(
                success=False,
                message=f"Access denied. You must be a Site Administrator to use this portal. Your role: {auth_token.site_role}",
                user_name=auth_token.user_name,
                site_role=auth_token.site_role,
                is_admin=False
            )

        # Store auth token in session
        request.session["tableau_auth"] = auth_token.to_dict()

        return LoginResponse(
            success=True,
            message="Successfully authenticated with Tableau",
            user_name=auth_token.user_name,
            site_role=auth_token.site_role,
            is_admin=True
        )

    except Exception as e:
        return LoginResponse(
            success=False,
            message=str(e),
            user_name="",
            site_role="",
            is_admin=False
        )


@app.post("/logout")
async def logout(request: Request):
    """Logout and invalidate Tableau session"""
    auth_data = request.session.get("tableau_auth")

    if auth_data:
        try:
            auth_token = TableauAuthToken.from_dict(auth_data)
            # Invalidate the Tableau token
            await signout_from_tableau(auth_token.token)
        except:
            pass

    # Clear session
    request.session.clear()

    return {"message": "Successfully logged out"}


@app.get("/auth/status")
async def auth_status(request: Request):
    """Check authentication status"""
    auth_data = request.session.get("tableau_auth")

    if not auth_data:
        return {
            "authenticated": False,
            "user_name": None,
            "site_role": None,
            "is_admin": False
        }

    try:
        auth_token = TableauAuthToken.from_dict(auth_data)

        if auth_token.is_expired():
            request.session.clear()
            return {
                "authenticated": False,
                "user_name": None,
                "site_role": None,
                "is_admin": False
            }

        return {
            "authenticated": True,
            "user_name": auth_token.user_name,
            "site_role": auth_token.site_role,
            "is_admin": auth_token.is_admin()
        }
    except:
        request.session.clear()
        return {
            "authenticated": False,
            "user_name": None,
            "site_role": None,
            "is_admin": False
        }


@app.get("/system-prompt")
async def get_system_prompt(current_user: TableauAuthToken = Depends(get_current_user)):
    """Get admin system prompt for display in UI (requires authentication)"""
    from utilities.admin_agent import ADMIN_SYSTEM_PROMPT
    return {
        "systemPrompt": ADMIN_SYSTEM_PROMPT
    }

@app.post("/mcp-chat")
async def mcp_chat_endpoint(
    request: ChatRequest,
    current_user: TableauAuthToken = Depends(get_current_user)
) -> ChatResponse:
    """Admin chat endpoint (non-streaming) - requires authentication"""
    try:
        print(f"🔐 Admin Chat request from {current_user.user_name} ({current_user.site_role}): {request.message}")

        # Process with Admin MCP Agent
        result = await admin_mcp_chat(
            query=request.message,
            conversation_history=request.history
        )

        return ChatResponse(
            response=result["response"],
            tool_results=result["tool_results"],
            iterations=result["iterations"]
        )

    except Exception as e:
        print(f"❌ Admin Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp-chat-stream")
async def mcp_chat_stream_endpoint(
    request: ChatRequest,
    current_user: TableauAuthToken = Depends(get_current_user)
):
    """Admin chat streaming endpoint with progress updates - requires authentication"""

    async def generate_stream():
        try:
            # Send initial progress
            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Admin agent initialized\", \"step\": \"init\"}}\n\n"

            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Connecting to admin MCP server...\", \"step\": \"connect\"}}\n\n"

            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Loading admin tools (user & group management)...\", \"step\": \"load-tools\"}}\n\n"

            yield f"event: progress\n"
            yield f"data: {{\"message\": \"Processing admin request...\", \"step\": \"processing\"}}\n\n"

            # Process with Admin MCP Agent
            result = await admin_mcp_chat(
                query=request.message,
                conversation_history=request.history
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
    print("📊 Admin operations: User & Group Management")

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
