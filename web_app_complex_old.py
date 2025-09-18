"""
Clean MCP + LangChain web application for Tableau
Following the standard pattern from The Information Lab's starter kit
"""

import os
import json
import traceback
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Web UI Libraries
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Libraries
from langchain_core.messages import HumanMessage

# Load Environment
from dotenv import load_dotenv
load_dotenv()

# Import our MCP integration
from utilities.langchain_mcp import tableau_mcp_lifespan, get_tableau_agent, format_agent_response

# Set Langfuse Tracing (optional)
try:
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
except ImportError:
    langfuse_handler = None

# Global variables for agent
agent = None

# Global async context manager for MCP connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("🚀 Starting up Tableau MCP application...")
    
    try:
        # Initialize MCP agent using our utility
        async with tableau_mcp_lifespan() as mcp_agent:
            agent = mcp_agent
            print("✅ Tableau MCP agent initialized successfully")
            yield
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        traceback.print_exc()
        raise

# Create FastAPI app with lifespan
app = FastAPI(
    title="Tableau AI Chat - MCP Edition", 
    description="Clean MCP + LangChain interface for Tableau data",
    lifespan=lifespan
)

# Add CORS middleware for Tableau extension support
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tableau-langchain-starter-kit.vercel.app",
        "http://localhost:8000",
        "http://localhost:3000",
        "https://*.tableau.com",
        "https://*.tableauonline.com",
        "https://*.tableaupublic.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

def get_client_session_id(request: Request) -> str:
    """Get client session ID for logging"""
    return request.client.host

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse('static/index.html')

@app.get("/index.html")
def static_index():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Tableau MCP AI Chat backend is running"}

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Simple chat endpoint - non-streaming"""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized. Please restart the server.")
    
    try:
        # Get response from agent
        result = await agent.chat(request.message)
        
        return ChatResponse(response=result.get('response', 'No response generated'))
        
    except Exception as e:
        print(f"❌ Error processing chat request: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/chat-stream")
async def chat_stream(request: ChatRequest, fastapi_request: Request):
    """Streaming chat endpoint with SSE (compatible with existing frontend)"""
    
    async def generate():
        session_id = get_client_session_id(fastapi_request)
        
        try:
            # Send initial connection event
            yield f"event: progress\\ndata: {json.dumps({'message': 'Connection established', 'step': 'init'})}\\n\\n"
            
            yield f"event: progress\\ndata: {json.dumps({'message': 'Starting MCP analysis...', 'step': 'analysis-start'})}\\n\\n"
            
            # Get agent and process request
            tableau_agent = await get_tableau_agent()
            result = await tableau_agent.chat(request.message)
            
            response_text = result.get('response', '')
            
            if not response_text:
                response_text = "I apologize, but I was unable to process your request at this time."
            
            # Send final result
            yield f"event: result\\ndata: {json.dumps({'response': response_text, 'toolResults': result.get('tool_calls', []), 'iterations': 1})}\\n\\n"
            yield f"event: done\\ndata: {json.dumps({'message': 'Stream complete'})}\\n\\n"
            
        except Exception as e:
            print("❌ Error in streaming chat endpoint:")
            traceback.print_exc()
            yield f"event: error\\ndata: {json.dumps({'error': 'Failed to process chat request', 'details': str(e)})}\\n\\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

# Alternative streaming endpoint that uses the agent's native streaming
@app.post("/chat-stream-native")
async def chat_stream_native(request: ChatRequest, fastapi_request: Request):
    """Native streaming chat using agent's stream capabilities"""
    
    async def generate():
        try:
            # Send initial events
            yield f"event: progress\\ndata: {json.dumps({'message': 'Connection established', 'step': 'init'})}\\n\\n"
            yield f"event: progress\\ndata: {json.dumps({'message': 'Starting MCP streaming...', 'step': 'stream-start'})}\\n\\n"
            
            # Stream from agent
            tableau_agent = await get_tableau_agent()
            
            response_chunks = []
            async for chunk in tableau_agent.stream_chat(request.message):
                response_chunks.append(chunk)
                # Send partial updates
                yield f"event: partial\\ndata: {json.dumps({'partial': chunk})}\\n\\n"
            
            # Send final result
            final_response = ''.join(response_chunks) if response_chunks else "No response generated"
            yield f"event: result\\ndata: {json.dumps({'response': final_response, 'toolResults': [], 'iterations': 1})}\\n\\n"
            yield f"event: done\\ndata: {json.dumps({'message': 'Stream complete'})}\\n\\n"
            
        except Exception as e:
            print("❌ Error in native streaming:")
            traceback.print_exc()
            yield f"event: error\\ndata: {json.dumps({'error': 'Failed to process streaming request', 'details': str(e)})}\\n\\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

# Legacy endpoint for compatibility
@app.post("/chat-legacy")
async def chat_legacy(request: ChatRequest):
    """Legacy endpoint using the old format_agent_response pattern"""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Create proper message format for LangGraph
        messages = [HumanMessage(content=request.message)]

        # Get response from agent using the legacy format
        response_text = await format_agent_response(
            agent.agent, 
            messages, 
            langfuse_handler
        )
        
        return ChatResponse(response=response_text)
        
    except Exception as e:
        print(f"❌ Error in legacy chat: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Standard port
    print(f"🚀 Starting Tableau MCP server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
