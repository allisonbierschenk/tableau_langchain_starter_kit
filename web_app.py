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
from langchain.tools import BaseTool


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

class MCPClient:
    """
    A robust client that correctly handles JSON-RPC over a Server-Sent Events (SSE) stream.
    It tracks request IDs to differentiate between notifications and tool results.
    """
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'MCP-Protocol-Version': '0.1.0'
        })
        logger.info(f"MCP Client initialized for {server_url}")

    def _send_request(self, method: str, params: Dict = None) -> Dict:
        """
        Sends a JSON-RPC request, listens to the SSE stream, and waits specifically
        for the response message that matches the request's ID.
        """
        request_id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "method": method, "id": request_id, "params": params or {}}
        
        logger.info(f"Sending MCP Request -> ID: {request_id}, Method: {method}")

        try:
            with self._session.post(self._server_url, json=payload, stream=True, timeout=60) as response:
                response.raise_for_status()
                
                # Process the response as a stream of events
                for line in response.iter_lines():
                    # SSE messages start with 'data:'
                    if line.startswith(b'data:'):
                        data_content = line[5:].strip()
                        if not data_content:
                            continue # Skip empty data lines

                        try:
                            parsed_data = json.loads(data_content)
                            
                            # THIS IS THE CRITICAL LOGIC:
                            # We check if the message's ID matches our request ID.
                            # This ensures we get the tool result, not just a notification.
                            if parsed_data.get('id') == request_id:
                                if 'error' in parsed_data:
                                    error_info = parsed_data['error']
                                    raise RuntimeError(f"MCP server error (code {error_info.get('code')}): {error_info.get('message')}")
                                
                                logger.info(f"Received result for ID {request_id}")
                                return parsed_data.get('result', {}) # Return the actual result and stop listening
                            
                            # If it's a notification without an ID, we can log it but we don't return it
                            elif 'method' in parsed_data and parsed_data['method'] == 'notifications/message':
                                logger.info(f"Ignoring MCP Notification: {parsed_data.get('params', {}).get('message', '')[:100]}...")

                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode a line from SSE stream: {data_content}")
                            continue
                
                # If we finish the stream without finding our response, something went wrong.
                raise RuntimeError(f"Stream ended without a response for request ID {request_id}")

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed for method '{method}': {e}")
            raise RuntimeError(f"Failed to communicate with MCP server: {e}") from e

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
        return self._send_request(method="tools/list")

    def call_tool(self, tool_name: str, arguments: Dict, context: Dict = None) -> Dict:
        """Calls a specific tool on the server."""
        params = {"name": tool_name, "arguments": arguments}
        if context:
            params["context"] = context
        return self._send_request(method="tools/call", params=params)

    def close(self):
        """Closes the session."""
        logger.info("Closing MCP client session.")
        self._session.close()


class MCPTool(BaseTool):
    """A custom LangChain tool to interact with the MCP server."""
    client: MCPClient
    tool_name: str
    datasource_luid: str

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool and correctly parse the nested MCP response."""
        context = {"datasource_luid": self.datasource_luid}
        result = self.client.call_tool(self.tool_name, kwargs, context)
        content = result.get('content', '')

        # THE FIX: We now correctly parse the nested response and, crucially,
        # return the FULL, rich data (e.g., [{'name': '...', 'id': '...'}])
        # to the agent, not just the name. This gives it the LUID it needs.
        try:
            if isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
                actual_output = json.loads(content[0]['text'])
                return actual_output
        except (json.JSONDecodeError, TypeError, KeyError):
            return content
            
        return content
    
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool asynchronously."""
        return self._run(*args, **kwargs)


def setup_langchain_tools(client: MCPClient, ds_metadata: Dict) -> list:
    """
    Creates a list of robust MCPTool instances for the LangChain agent.
    """
    datasource_luid = ds_metadata['luid']
    logger.info(f"Setting up LangChain tools for Datasource LUID: {datasource_luid}")
    
    available_tools_data = client.list_tools().get('tools', [])
    if not available_tools_data:
        raise RuntimeError("MCP server returned no available tools.")

    # THIS IS THE FIX: A mapping from JSON schema types to actual Python types.
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    langchain_tools = []
    for tool_data in available_tools_data:
        tool_name = tool_data['name']
        
        # Dynamically create the arguments schema for each tool using Pydantic
        fields = {}
        # Iterate through the properties defined in the tool's input schema
        for p_name, p_details in tool_data.get('inputSchema', {}).get('properties', {}).items():
            # Get the JSON type (e.g., "string") and find its Python equivalent (e.g., str)
            json_type = p_details.get('type', 'string')
            python_type = type_mapping.get(json_type, str)  # Default to str if unknown
            # Add the field with the correct Python type to our fields dictionary
            fields[p_name] = (python_type, Field(None, description=p_details.get('description', '')))
        
        # Use Pydantic's create_model to build the class for the tool's arguments
        # If a tool has no arguments, fields will be empty, and it will correctly handle no-input calls.
        args_schema = create_model(f'{tool_name}Args', **fields)

        # Create an instance of our robust MCPTool class
        mcp_tool = MCPTool(
            client=client,
            tool_name=tool_name,
            datasource_luid=datasource_luid,
            name=tool_name,
            description=tool_data['description'],
            args_schema=args_schema
        )
        langchain_tools.append(mcp_tool)

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

    mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
    mcp_client = MCPClient(server_url=mcp_server_url)

    try:
        yield f"event: progress\ndata: {json.dumps({'message': 'Initializing MCP connection...'})}\n\n"
        mcp_client.connect()

        yield f"event: progress\ndata: {json.dumps({'message': 'Fetching available tools...'})}\n\n"
        tools = setup_langchain_tools(mcp_client, ds_metadata)
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True) # Using gpt-4o for better instruction following
        
        # THE FIX: A much more detailed prompt that explains the workflow,
        # including how to extract the 'id' (LUID) from the tool's output.
        prompt_template = """
        You are a helpful Tableau data assistant. You have access to the following tools:
        {tools}

        **CRITICAL WORKFLOW:**
        1.  When asked a question about data, you **MUST** first use the `list-datasources` tool to see what data is available.
        2.  The `list-datasources` tool will return a list of datasource objects, like `[{{'name': 'Superstore', 'id': '5f2b4ace...'}}]`.
        3.  From this list, identify the most relevant datasource and **extract its 'id'**. This ID is the `datasourceLuid`.
        4.  You **MUST** then use the `list-fields` tool, providing the `datasourceLuid` you just extracted, to see the columns in that datasource.
        5.  Once you have the field names, use the `query-datasource` tool with the correct `datasourceLuid` and `fields` to answer the user's question.
        6.  Do not make up field names or LUIDs. You must discover them with the tools.

        Use the following format for your thought process:
        Question: the input question you must answer
        Thought: your reasoning and plan for the next step based on the workflow above.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat)
        Thought: I now have enough information to provide the final answer.
        Final Answer: the final answer to the original input question.

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        yield f"event: progress\ndata: {json.dumps({'message': 'Starting AI analysis...'})}\n\n"
        final_response = ""
        
        async for chunk in agent_executor.astream_log({"input": message}):
            if chunk.ops and chunk.ops[0]["op"] == "add":
                log_entry_path = chunk.ops[0]["path"]
                if log_entry_path.startswith("/logs/") and "streamed_output" not in log_entry_path:
                    log_entry = chunk.ops[0]['value']
                    if isinstance(log_entry, dict) and 'event' in log_entry and 'data' in log_entry:
                        event_type = log_entry['event']
                        event_data = log_entry['data']
                        if event_type == 'on_chain_end' and 'output' in event_data:
                            output = event_data.get('output', {})
                            if isinstance(output, dict) and 'output' in output:
                                final_answer = output['output']
                                final_response = final_answer
                                for char in final_answer:
                                    yield f"event: token\ndata: {json.dumps({'token': char})}\n\n"
                                    await asyncio.sleep(0.01)
                        elif event_type == 'on_tool_start' and 'tool_input' in event_data:
                            tool_input = event_data.get('tool_input', {})
                            tool_name = tool_input.get('tool_name', 'Unknown Tool')
                            tool_args = tool_input.get('tool_input', {})
                            yield f"event: progress\ndata: {json.dumps({'message': f'Calling tool: {tool_name}({json.dumps(tool_args)})'})}\n\n"
        
        yield f"event: result\ndata: {json.dumps({'response': final_response})}\n\n"

    except Exception as e:
        logger.error(f"Error during chat stream: {e}", exc_info=True)
        error_details = traceback.format_exc()
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'details': error_details})}\n\n"
    finally:
        mcp_client.close()
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

# In web_app.py, replace the entire generate_jwt function with this one.

def generate_jwt(tableau_username):
    client_id = os.environ['TABLEAU_JWT_CLIENT_ID']
    secret_id = os.environ['TABLEAU_JWT_SECRET_ID']
    secret_value = os.environ['TABLEAU_JWT_SECRET']
    now = datetime.datetime.now(datetime.UTC)
    
    # THE FIX IS HERE: We are using a broader set of scopes to ensure
    # compatibility with the Tableau REST API endpoints being called.
    payload = {
        "iss": client_id,
        "sub": tableau_username,
        "aud": "tableau",
        "jti": str(uuid.uuid4()),
        "exp": now + datetime.timedelta(minutes=5),
        "scp": [
            "tableau:rest_api:read", 
            "tableau:datasources:query", 
            "tableau:content:read"
        ]
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