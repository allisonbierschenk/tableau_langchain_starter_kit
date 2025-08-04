# web_app.py - FINAL VERSION
# This backend is built to work specifically with your provided script.js, html, css, and prompt.py files.

import os
import json
import uuid
import logging
import datetime
import asyncio
import traceback
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, create_model
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate

# Import the prompt building functions from your file
from prompt import build_agent_identity, build_agent_system_prompt

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variables ---
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

app = FastAPI(title="Tableau AI Chat", description="A streaming AI chat interface for Tableau data")

# --- In-Memory Store to hold the LUID from the frontend ---
DATASOURCE_METADATA_STORE: Dict[str, Dict] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class DatasourceRequest(BaseModel):
    datasources: Dict[str, str]

# --- MCP Client ---
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

# --- LangChain Tool ---
class MCPTool(BaseTool):
    client: MCPClient
    tool_name: str
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        tool_args = {}
        if args and isinstance(args[0], dict): tool_args = args[0]
        elif kwargs: tool_args = kwargs
        elif args and len(self.args_schema.model_fields) == 1: tool_args = {next(iter(self.args_schema.model_fields.keys())): args[0]}
        logger.info(f"Executing tool '{self.tool_name}' with processed args: {tool_args}")
        result = self.client.call_tool(self.tool_name, tool_args)
        content = result.get('content', '')
        if isinstance(content, list) and content and isinstance(content[0], dict) and 'text' in content[0]:
            try: return json.loads(content[0]['text'])
            except json.JSONDecodeError: return content[0]['text']
        return content
    async def _arun(self, *args: Any, **kwargs: Any) -> Any: return self._run(*args, **kwargs)

# --- Core Chat Streaming Logic ---
async def stream_chat_events(message: str, client_id: str):
    def sse_format(event_name: str, data: Dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

    mcp_client = None
    try:
        # Get the stored LUID for this client
        ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
        if not ds_metadata or "luid" not in ds_metadata:
            raise ValueError("Datasource LUID not initialized for this session. Please refresh the dashboard.")
        
        datasource_luid = ds_metadata["luid"]
        datasource_name = ds_metadata["name"]
        
        yield sse_format("progress", {"message": "Initializing AI assistant..."})
        
        mcp_server_url = os.environ.get("MCP_SERVER_URL", "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp")
        mcp_client = MCPClient(server_url=mcp_server_url)
        
        yield sse_format("progress", {"message": "Setting up AI tools..."})
        
        tools_data = mcp_client.list_tools().get('tools', [])
        tools = [MCPTool(client=mcp_client, tool_name=t['name'], name=t['name'], description=t.get('description',''), args_schema=create_model(f"{t['name']}Args", **{p: (Optional[str], None) for p in t.get('inputSchema',{}).get('properties',{})})) for t in tools_data]
        
        # Create the agent's identity using your prompt.py function
        agent_identity = build_agent_identity(ds_metadata)
        
        # This new prompt combines your prompt style with the critical LUID instruction
        system_prompt = f"""{agent_identity}

        ---
        **CRITICAL CONTEXT & INSTRUCTIONS:**
        1. You are working with a specific datasource that has already been identified. Its name is **'{datasource_name}'** and its unique ID (LUID) is **`{datasource_luid}`**.
        2. You MUST use this LUID for any tool that requires a `datasourceLuid` parameter.
        3. Do NOT use the `list-datasources` tool. The correct datasource has been provided to you.
        4. Your first step to answer a user's question should almost always be to use the `list-fields` tool with the provided LUID to understand the available data columns.
        ---

        **Response Guidelines:**
        - Start by answering the user's question clearly.
        - Base your answers ONLY on data retrieved via your tools.
        - If a query fails, explain why and guide the user to a next best step.
        - NEVER invent fields, metrics, or values not present in the data.

        Begin!

        Question: {{input}}
        Thought: {{agent_scratchpad}}"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors="Check your LUID and field names and try again.")

        final_answer = ""
        async for chunk in agent_executor.astream_log({"input": message}):
            op = chunk.ops[0] if chunk.ops else None
            if not op or op.get("op") != "add": continue
            path = op.get("path", "")
            if path.startswith("/logs/") and "streamed_output" not in path:
                log_entry = op.get('value', {})
                event_type = log_entry.get('event')

                if event_type == 'on_tool_start':
                    tool_name = log_entry.get('data', {}).get('name')
                    yield sse_format("progress", {"message": f"Querying {datasource_name}..."})

                elif event_type == 'on_llm_new_token':
                    token = log_entry.get('data', {}).get('chunk', {}).get('content', '')
                    if token: yield sse_format("token", {"token": token})

                elif event_type == 'on_chain_end' and "/agent" in path:
                    final_answer = log_entry.get('data', {}).get('output', {}).get('output', "")
        
        if final_answer:
            yield sse_format("result", {"response": final_answer})

    except Exception as e:
        logger.error(f"Error in chat stream: {e}", exc_info=True)
        yield sse_format("error", {"error": str(e)})
    finally:
        if mcp_client: mcp_client.close()
        yield sse_format("done", {"message": "Stream complete"})

# --- FastAPI Endpoints ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, fastapi_request: Request):
    client_id = fastapi_request.client.host
    logger.info(f"Received chat request from {client_id}: '{request.message[:100]}...'")
    return StreamingResponse(stream_chat_events(request.message, client_id), media_type="text/event-stream")

@app.post("/datasources")
async def receive_datasources(request: DatasourceRequest, fastapi_request: Request):
    client_id = fastapi_request.client.host
    try:
        if not request.datasources:
            raise HTTPException(status_code=400, detail="No datasources provided.")
        
        first_name = next(iter(request.datasources.keys()))
        first_luid = request.datasources[first_name]
        logger.info(f"Initializing LUID '{first_luid}' for datasource '{first_name}' for client {client_id}")

        DATASOURCE_METADATA_STORE[client_id] = {"name": first_name, "luid": first_luid}
        return JSONResponse({"status": "success", "initialized_datasource_name": first_name})

    except Exception as e:
        logger.error(f"Datasource initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- Static Files and Startup ---
@app.on_event("startup")
async def startup_event():
    required = ['MCP_SERVER_URL', 'OPENAI_API_KEY']
    if any(v not in os.environ for v in required):
        logger.error(f"FATAL: Missing one or more required environment variables: {required}")
    else:
        logger.info("Application startup complete.")

# Assumes your index.html, script.js etc. are in a folder named 'static'
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    # Serves the main index.html file
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)