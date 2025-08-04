# web_app.py - FINAL version, translated from the working TypeScript example.

import os
import json
import uuid
import logging
import datetime
import traceback
import asyncio
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI as OpenAIClient

# Use your existing prompt structure
from utilities.prompt import build_agent_identity, build_agent_system_prompt

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variables ---
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

app = FastAPI(title="Tableau AI Chat", description="A streaming AI chat interface for Tableau data.")

# --- Pydantic Models to match the React frontend's request body ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    query: str

# --- In-Memory Store ---
# This will hold the LUID found by the original script.js so the AI can use it.
DATASOURCE_METADATA_STORE: Dict[str, Dict] = {}


# --- MCP Client (for talking to the MCP Server) ---
class MCPClient:
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream',
            'MCP-Protocol-Version': '0.1.0'
        })
        self._timeout = 180
        logger.info(f"MCP Client initialized for {server_url}")

    def _send_request(self, method: str, params: Dict = None) -> Dict:
        request_id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "method": method, "id": request_id, "params": params or {}}
        logger.info(f"Sending MCP Request -> ID: {request_id}, Method: {method}")
        try:
            with self._session.post(self._server_url, json=payload, stream=True, timeout=self._timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith('data:'):
                        data_content = line[5:].strip()
                        if data_content and data_content != '[DONE]':
                            try:
                                parsed_data = json.loads(data_content)
                                if parsed_data.get('id') == request_id:
                                    if 'error' in parsed_data: raise RuntimeError(f"MCP Error: {parsed_data['error']}")
                                    return parsed_data.get('result', {})
                            except json.JSONDecodeError: logger.warning(f"Could not decode SSE line: {data_content}")
                raise RuntimeError("Stream ended without a valid response.")
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP request failed: {e}", exc_info=True)
            raise RuntimeError(f"MCP communication error: {e}")

    def list_tools(self) -> Dict: return self._send_request("tools/list")
    def call_tool(self, tool_name: str, arguments: Dict) -> Dict: return self._send_request("tools/call", {"name": tool_name, "arguments": arguments})
    def close(self): self._session.close()

# --- Main Streaming Logic (Translated from mcp-chat.ts) ---
async def stream_mcp_chat_events(request: ChatRequest, client_id: str):
    def sse_format(event_name: str, data: Dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

    mcp_client = None
    try:
        # Get metadata that your original script.js stored
        ds_metadata = DATASOURCE_METADATA_STORE.get(client_id, {})
        datasource_luid = ds_metadata.get("luid")
        
        # --- Initialization Phase ---
        yield sse_format("progress", {"message": "Connection established", "step": "init"})
        
        yield sse_format("progress", {"message": "Initializing MCP connection...", "step": "init"})
        mcp_server_url = os.environ.get("MCP_SERVER_URL", "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp")
        mcp_client = MCPClient(server_url=mcp_server_url)
        
        openai_client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))

        # --- Tool Setup Phase ---
        yield sse_format("progress", {"message": "Getting available tools...", "step": "tools"})
        tools_response = mcp_client.list_tools()
        mcp_tools = tools_response.get('tools', [])
        tool_names = [t['name'] for t in mcp_tools]
        
        yield sse_format("progress", {"message": f"Found {len(mcp_tools)} tools: {', '.join(tool_names)}", "step": "tools-found"})
        
        openai_tools = [{
            "type": "function",
            "function": {"name": tool['name'], "description": tool['description'], "parameters": tool['inputSchema']}
        } for tool in mcp_tools]

        # --- Prompting Phase ---
        # Using your prompt.py functions to build the identity and base prompt
        agent_identity = build_agent_identity(ds_metadata)
        base_prompt = build_agent_system_prompt(agent_identity, ds_metadata.get("name", "the datasource"))

        # Add the critical instructions from the TypeScript example, including the LUID
        critical_instructions = f"""
        **CRITICAL INSTRUCTIONS (HIGHEST PRIORITY):**
        1. You are working with a datasource whose unique ID (LUID) is: `{datasource_luid}`. You MUST use this LUID for any tool that requires a `datasourceLuid`.
        2. Do NOT use `list-datasources`. You have been given the only one you need.
        3. For data analysis questions, your first step is ALWAYS `list-fields` to see the columns, then `query-datasource`.
        4. For "insights", "metric", or "KPI" questions, use Pulse tools first: `list-all-pulse-metric-definitions`, then `generate-pulse-metric-value-insight-bundle`.
        """
        system_prompt = f"{base_prompt}\n{critical_instructions}"

        # --- Main Agent Loop (Translated directly from the TypeScript example) ---
        conversation_messages = [{"role": "system", "content": system_prompt}]
        conversation_messages.extend([msg.dict() for msg in request.messages])
        conversation_messages.append({"role": "user", "content": request.query})

        max_iterations = 10
        yield sse_format("progress", {"message": "Starting analysis...", "step": "analysis-start", "maxIterations": max_iterations})
        
        all_tool_results = []
        final_response_text = ""
        last_completion = None
        
        for iteration in range(1, max_iterations + 1):
            yield sse_format("progress", {"message": f"Iteration {iteration}/{max_iterations}: Analyzing and planning...", "step": "iteration-start", "iteration": iteration})

            completion = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=conversation_messages,
                tools=openai_tools,
                tool_choice="auto"
            )
            last_completion = completion
            assistant_message = completion.choices[0].message
            conversation_messages.append(assistant_message.model_dump(exclude_none=True))

            if not assistant_message.tool_calls:
                final_response_text = assistant_message.content or "Analysis complete."
                yield sse_format("progress", {"message": "Analysis complete - generating final response...", "step": "complete"})
                break

            yield sse_format("progress", {"message": f"Executing {len(assistant_message.tool_calls)} tool(s)...", "step": "tools-executing", "toolCount": len(assistant_message.tool_calls)})

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                yield sse_format("progress", {"message": f"{tool_name}(...)", "step": "tool-executing", "tool": tool_name, "arguments": tool_args})
                
                try:
                    tool_result_content = mcp_client.call_tool(tool_name, tool_args).get('content')
                    all_tool_results.append({"tool": tool_name, "arguments": tool_args, "result": tool_result_content})
                    conversation_messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": json.dumps(tool_result_content)})
                    yield sse_format("progress", {"message": f"{tool_name} completed successfully", "step": "tool-completed", "tool": tool_name, "success": True})
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"Tool call failed for {tool_name}: {error_str}")
                    all_tool_results.append({"tool": tool_name, "arguments": tool_args, "error": error_str})
                    conversation_messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": json.dumps({"error": error_str})})
                    yield sse_format("progress", {"message": f"{tool_name} failed: {error_str}", "step": "tool-error", "tool": tool_name, "error": error_str})
            
            yield sse_format("progress", {"message": f"Iteration {iteration} completed", "step": "iteration-complete", "iteration": iteration})

        # --- Final Result Phase ---
        usage_data = last_completion.usage.model_dump() if last_completion and last_completion.usage else {}
        final_payload = {"response": final_response_text, "toolResults": all_tool_results, "usage": usage_data, "iterations": iteration}
        yield sse_format("result", final_payload)

    except Exception as e:
        logger.error(f"Error during chat stream: {e}", exc_info=True)
        yield sse_format("error", {"error": "An unexpected error occurred on the server.", "details": traceback.format_exc()})
    
    finally:
        if mcp_client: mcp_client.close()
        yield sse_format("done", {"message": "Stream complete"})

# --- FastAPI Endpoints ---
# This endpoint is for your ORIGINAL script.js to initialize the datasource LUID
@app.post("/datasources")
async def receive_datasources(request: Request, body: Dict[str, Any]):
    client_id = request.client.host
    datasources_map = body.get("datasources", {})
    if not datasources_map:
        raise HTTPException(status_code=400, detail="No datasources provided.")
    
    first_name = next(iter(datasources_map.keys()))
    first_luid = datasources_map[first_name]
    logger.info(f"Initializing LUID '{first_luid}' for datasource '{first_name}' for client {client_id}")
    
    # Store the LUID and name for this client session
    DATASOURCE_METADATA_STORE[client_id] = {"name": first_name, "luid": first_luid, "description": ""}
    return JSONResponse({"status": "success", "initialized_datasource_name": first_name})

# This endpoint is for the EXAMPLE React frontend (AIAssistent.js)
@app.post("/mcp-chat-stream")
async def mcp_chat_stream_endpoint(request: ChatRequest, fastapi_request: Request):
    client_id = fastapi_request.client.host
    logger.info(f"Received streaming chat request: '{request.query[:100]}...'")
    # A default LUID must be set for this endpoint to work without the /datasources call
    if client_id not in DATASOURCE_METADATA_STORE:
        DATASOURCE_METADATA_STORE[client_id] = {"name": "Default Datasource", "luid": "default-luid-from-env", "description": ""}
    return StreamingResponse(stream_mcp_chat_events(request, client_id), media_type="text/event-stream")

# This is the endpoint YOUR script.js is calling
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, fastapi_request: Request):
    client_id = fastapi_request.client.host
    logger.info(f"Received simple chat request from {client_id}: '{request.message[:100]}...'")
    
    # Adapt the simple request to the complex one the streaming function expects
    adapted_request = ChatRequest(messages=[], query=request.message)
    return StreamingResponse(stream_mcp_chat_events(adapted_request, client_id), media_type="text/event-stream")

# --- Static Files and Startup ---
@app.get("/")
async def root():
    return FileResponse('static/index.html')

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    required = ['MCP_SERVER_URL', 'OPENAI_API_KEY']
    if any(v not in os.environ for v in required):
        logger.error(f"FATAL: Missing one or more required environment variables: {required}")
    else:
        logger.info("Application startup complete.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))