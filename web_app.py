"""
Simple MCP + LangChain web application for Tableau
Using HTTP MCP client like the e-bikes demo
"""

import os
import json
import asyncio
from typing import List, Dict, Any

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

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

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

@app.post("/mcp-chat")
async def mcp_chat_endpoint(request: ChatRequest) -> ChatResponse:
    """MCP chat endpoint (non-streaming) - matches e-bikes demo pattern"""
    try:
        print(f"💬 MCP Chat request: {request.message}")
        
        # Process with MCP
        result = await tableau_mcp_chat(
            query=request.message,
            conversation_history=request.history
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
async def mcp_chat_stream_endpoint(request: ChatRequest):
    """MCP chat streaming endpoint with user-friendly progress"""
    
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
            
            # Process with MCP
            result = await tableau_mcp_chat(
                query=request.message,
                conversation_history=request.history
            )
            
            # Send final result
            yield f"event: result\n"
            yield f"data: {json.dumps(result)}\n\n"
            
            yield f"event: done\n"
            yield f"data: {{\"message\": \"Stream complete\"}}\n\n"
            
        except Exception as e:
            print(f"❌ Stream error: {str(e)}")
            yield f"event: error\n"
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
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
    
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
