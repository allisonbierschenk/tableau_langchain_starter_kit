# web_app.py - REWRITTEN FOR STREAMING

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, create_model
import os
import requests
import traceback
import uuid
import logging
import jwt
import datetime
import json
import asyncio
from typing import Dict, Any

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

app = FastAPI(title="Tableau AI Chat", description="A streaming AI chat interface for Tableau data")

# --- In-Memory Store for Datasource Metadata ---
DATASOURCE_METADATA_STORE = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

# --- Corrected MCP Client ---
class MCPClient:
    """
    A client that correctly handles the JSON-RPC over SSE protocol required by the MCP server.
    """
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._session = requests.Session()
        # The server expects this specific combination of Accept headers
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'MCP-Protocol-Version': '0.1.0'
        })
        logger.info(f"MCP Client initialized for {server_url}")

    def _parse_sse_response(self, response_text: str) -> Dict:
        """
        Reliably parses a Server-Sent Events (SSE) response chunk.
        It looks for the 'data: ' prefix and extracts the JSON payload.
        """
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('data:'):
                data_content = line[5:].strip()
                try:
                    return json.loads(data_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse SSE data as JSON: {data_content}")
                    raise ValueError(f"Invalid JSON in SSE data: {e}") from e
        logger.warning(f"No 'data:' line found in SSE response chunk: {response_text}")
        # Fallback for non-SSE JSON responses, though unlikely with MCP
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError("Response is not valid SSE or JSON.")

    def _send_request(self, method: str, params: Dict = None) -> Dict:
        """Sends a JSON-RPC request and parses the SSE response."""
        payload = {"jsonrpc": "2.0", "method": method, "id": str(uuid.uuid4())}
        if params:
            payload["params"] = params

        logger.info(f"Sending MCP Request -> Method: {method}, Params: {json.dumps(params, indent=2)}")

        try:
            with self._session.post(self._server_url, json=payload, stream=True, timeout=60) as response:
                response.raise_for_status()
                # MCP server sends the full response in a single chunk for simple calls
                full_response_text = response.text
                logger.info(f"Raw MCP Response <- {full_response_text[:500]}") # Log first 500 chars
                
                parsed_data = self._parse_sse_response(full_response_text)

                if 'error' in parsed_data:
                    error_info = parsed_data['error']
                    raise RuntimeError(f"MCP server error (code {error_info.get('code')}): {error_info.get('message')}")

                return parsed_data.get('result', {})

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed for method '{method}': {e}")
            raise RuntimeError(f"Failed to communicate with MCP server: {e}") from e
        except ValueError as e:
            logger.error(f"Error parsing response for method '{method}': {e}")
            raise RuntimeError(f"Could not parse server response: {e}") from e

    def connect(self) -> Dict:
        """Establishes the connection with the required handshake."""
        logger.info("Connecting to MCP server with handshake...")
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "python-fastapi-client", "version": "1.0.0"}
        }
        return self._send_request(method="initialize", params=params)

    def list_tools(self) -> Dict:
        """Fetches the list of available tools."""
        logger.info("Fetching tools from MCP server...")
        return self._send_request(method="tools/list")

    def call_tool(self, tool_name: str, arguments: Dict, context: Dict = None) -> Dict:
        """Calls a specific tool on the server."""
        logger.info(f"Calling MCP Tool '{tool_name}' with args: {arguments}")
        params = {"name": tool_name, "arguments": arguments}
        if context:
            params["context"] = context
        return self._send_request(method="tools/call", params=params)

    def close(self):
        """Closes the session."""
        logger.info("Closing MCP client session.")
        self._session.close()


# --- LangChain Setup ---
# We will set up the agent inside the streaming endpoint for simplicity
# This requires installing langchain, langchain-openai, langsmith

def setup_langchain_tools(client: MCPClient, ds_metadata: Dict) -> list:
    """Creates LangChain tools from the MCP server's tool list."""
    from langchain.tools import BaseTool, tool

    datasource_luid = ds_metadata['luid']
    logger.info(f"Setting up LangChain tools for Datasource LUID: {datasource_luid}")
    
    available_tools_data = client.list_tools().get('tools', [])
    if not available_tools_data:
        raise RuntimeError("MCP server returned no available tools.")

    langchain_tools = []
    for tool_data in available_tools_data:
        tool_name = tool_data['name']
        tool_desc = tool_data['description']
        
        # Dynamically create a function that we can decorate with @tool
        def create_tool_func(name, desc):
            def mcp_tool_caller(**kwargs):
                """Dynamically generated tool function"""
                try:
                    context = {"datasource_luid": datasource_luid}
                    result = client.call_tool(name, kwargs, context)
                    # Return the content, which is often a JSON string inside a list
                    content = result.get('content', '')
                    if isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
                        return content[0]['text']
                    return json.dumps(content)
                except Exception as e:
                    logger.error(f"Error calling MCP tool '{name}': {e}")
                    return f"Error: {e}"
            
            # Use a more descriptive name for the dynamic function
            mcp_tool_caller.__name__ = tool_name
            mcp_tool_caller.__doc__ = tool_desc
            
            # Use LangChain's @tool decorator to create the tool
            return tool(mcp_tool_caller)

        new_tool = create_tool_func(tool_name, tool_desc)
        langchain_tools.append(new_tool)

    logger.info(f"Created {len(langchain_tools)} LangChain tools: {[t.name for t in langchain_tools]}")
    return langchain_tools


# --- STREAMING CHAT ENDPOINT ---
async def stream_chat_events(message: str, ds_metadata: Dict):
    """
    This generator function handles the entire lifecycle of a chat request
    and streams events back to the client (browser).
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import ChatPromptTemplate

    # The MCP endpoint from your Heroku logs
    mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
    mcp_client = MCPClient(server_url=mcp_server_url)

    try:
        # 1. Send progress update: Initializing
        yield f"event: progress\ndata: {json.dumps({'message': 'Initializing MCP connection...'})}\n\n"
        mcp_client.connect()

        # 2. Setup LangChain Agent
        yield f"event: progress\ndata: {json.dumps({'message': 'Fetching available tools...'})}\n\n"
        tools = setup_langchain_tools(mcp_client, ds_metadata)
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        
        # Use a template compatible with create_react_agent
        prompt_template = """
        Answer the following questions as best you can. You have access to the following tools:
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        # 3. Stream agent execution
        yield f"event: progress\ndata: {json.dumps({'message': 'Starting AI analysis...'})}\n\n"
        final_response = ""
        # Use astream_log for detailed step-by-step streaming
        async for chunk in agent_executor.astream_log({"input": message}):
            # Each 'chunk' is a log of an agent step
            if chunk.ops:
                op = chunk.ops[0]
                if op["op"] == "add" and "/logs/" in op["path"]:
                    log_entry = op['value']
                    if log_entry['event'] == 'on_chain_end' and 'output' in log_entry['data']:
                        # Final answer chunk
                        output = log_entry['data']['output']
                        if isinstance(output, dict) and 'output' in output:
                           final_answer = output['output']
                           final_response = final_answer # Capture final answer
                           # Stream the final answer chunk by chunk for a typing effect
                           for char in final_answer:
                               yield f"event: token\ndata: {json.dumps({'token': char})}\n\n"
                               await asyncio.sleep(0.01) # small delay
                    
                    elif log_entry['event'] == 'on_tool_start':
                        # Tool call started
                        tool_name = log_entry['data']['tool_input']['tool_name']
                        tool_args = log_entry['data']['tool_input']['tool_input']
                        yield f"event: progress\ndata: {json.dumps({'message': f'Calling tool: {tool_name}({json.dumps(tool_args)})'})}\n\n"
        
        # 4. Send final result object
        yield f"event: result\ndata: {json.dumps({'response': final_response})}\n\n"

    except Exception as e:
        logger.error(f"Error during chat stream: {e}", exc_info=True)
        error_details = traceback.format_exc()
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'details': error_details})}\n\n"
    finally:
        mcp_client.close()
        # 5. Signal that the stream is done
        yield f"event: done\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request):
    """Main chat endpoint, now streaming responses."""
    client_id = fastapi_request.client.host
    ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
    if not ds_metadata:
        raise HTTPException(status_code=400, detail="Datasource not initialized. Please reload the dashboard.")

    logger.info(f"Streaming chat request from {client_id}: '{request.message}'")
    return StreamingResponse(stream_chat_events(request.message, ds_metadata), media_type="text/event-stream")


# --- Your Existing Tableau Datasource Logic (Unchanged) ---
# This part of your code for JWT and datasource lookup is correct.

def get_tableau_username(client_id):
    # This should be mapped securely in a real app, but env var is fine for this demo
    return os.environ['TABLEAU_USER']

def generate_jwt(tableau_username):
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
        "scp": ["tableau:rest_api:read", "tableau:datasources:query"] # Use modern scopes
    }
    headers = {"kid": secret_id, "iss": client_id}
    return jwt.encode(payload, secret_value, algorithm="HS256", headers=headers)

def tableau_signin_with_jwt(tableau_username):
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
    site = os.environ['TABLEAU_SITE']
    jwt_token = generate_jwt(tableau_username)
    url = f"{domain}/api/{api_version}/auth/signin"
    payload = {"credentials": {"jwt": jwt_token, "site": {"contentUrl": site}}}
    resp = requests.post(url, json=payload, headers={'Accept': 'application/json'}, timeout=20)
    resp.raise_for_status()
    creds = resp.json()['credentials']
    return creds['token'], creds['site']['id']

def lookup_published_luid_by_name(name, tableau_username):
    token, site_id = tableau_signin_with_jwt(tableau_username)
    domain = os.environ['TABLEAU_DOMAIN_FULL']
    api_version = os.environ.get('TABLEAU_API_VERSION', '3.21') # Datasources endpoint is stable
    url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
    headers = {"X-Tableau-Auth": token, "Accept": "application/json"}
    params = {"filter": f"name:eq:{name}"}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    datasources = resp.json().get("datasources", {}).get("datasource", [])
    if not datasources:
        raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found.")
    return datasources[0]["id"], datasources[0]

@app.post("/datasources")
async def receive_datasources(request: Request, body: Dict[str, Any]):
    client_id = request.client.host
    datasources_map = body.get("datasources", {})
    if not datasources_map:
         raise HTTPException(status_code=400, detail="No datasources provided in the request.")

    first_name = next(iter(datasources_map.keys()))
    tableau_username = get_tableau_username(client_id)
    published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username)

    DATASOURCE_METADATA_STORE[client_id] = {
        "name": first_name,
        "luid": published_luid,
        "projectName": full_metadata.get("projectName"),
        "description": full_metadata.get("description"),
        "owner_name": full_metadata.get("owner", {}).get("name")
    }
    logger.info(f"Initialized datasource for {client_id}: {first_name} (LUID: {published_luid})")
    return {"status": "ok", "selected_datasource_luid": published_luid}

# --- Static Files and Main Execution ---
@app.get("/", include_in_schema=False)
async def home():
    return FileResponse('static/index.html')

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")