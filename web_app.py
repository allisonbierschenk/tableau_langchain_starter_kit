# web_app.py - IMPROVED VERSION WITH BETTER ERROR HANDLING AND DEBUGGING

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
from typing import Dict, Any, Optional
from langchain.tools import BaseTool

# --- Enhanced Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    Enhanced MCP client with better error handling and debugging capabilities.
    """
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'MCP-Protocol-Version': '0.1.0'
        })
        self._timeout = 60
        logger.info(f"MCP Client initialized for {server_url}")

    def _send_request(self, method: str, params: Dict = None, timeout: Optional[int] = None) -> Dict:
        """
        Enhanced request sender with better error handling and timeout management.
        """
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0", 
            "method": method, 
            "id": request_id, 
            "params": params or {}
        }
        
        logger.info(f"Sending MCP Request -> ID: {request_id}, Method: {method}")
        logger.debug(f"Full payload: {json.dumps(payload, indent=2)}")

        timeout_value = timeout or self._timeout

        try:
            with self._session.post(
                self._server_url, 
                json=payload, 
                stream=True, 
                timeout=timeout_value
            ) as response:
                response.raise_for_status()
                
                # Process SSE stream
                for line_num, line in enumerate(response.iter_lines(decode_unicode=True)):
                    if line_num > 1000:  # Prevent infinite loops
                        logger.warning("Reached maximum line limit in SSE stream")
                        break
                        
                    if line and line.startswith('data:'):
                        data_content = line[5:].strip()
                        if not data_content or data_content == '[DONE]':
                            continue

                        try:
                            parsed_data = json.loads(data_content)
                            
                            # Handle our response
                            if parsed_data.get('id') == request_id:
                                if 'error' in parsed_data:
                                    error_info = parsed_data['error']
                                    logger.error(f"MCP server error: {error_info}")
                                    raise RuntimeError(f"MCP server error (code {error_info.get('code')}): {error_info.get('message')}")
                                
                                logger.info(f"Received result for ID {request_id}")
                                return parsed_data.get('result', {})
                            
                            # Handle notifications (ignore but log for debugging)
                            elif 'method' in parsed_data:
                                notification_method = parsed_data.get('method', 'unknown')
                                if notification_method == 'notifications/message':
                                    # Truncate long notification messages for cleaner logs
                                    msg_preview = str(parsed_data.get('params', {}))[:200]
                                    logger.debug(f"MCP Notification ({notification_method}): {msg_preview}...")
                                else:
                                    logger.debug(f"Received notification: {notification_method}")

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode SSE line {line_num}: {data_content[:100]}... Error: {e}")
                            continue
                
                raise RuntimeError(f"Stream ended without response for request ID {request_id}")

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {timeout_value}s for method '{method}'")
            raise RuntimeError(f"MCP server request timed out after {timeout_value}s")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for method '{method}': {e}")
            raise RuntimeError(f"Failed to connect to MCP server at {self._server_url}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed for method '{method}': {e}")
            raise RuntimeError(f"MCP server communication error: {e}")

    def connect(self) -> Dict:
        """Establishes connection with enhanced error handling."""
        logger.info("Connecting to MCP server with handshake...")
        try:
            params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "python-fastapi-client", "version": "1.0.0"}
            }
            result = self._send_request(method="initialize", params=params, timeout=30)
            logger.info("MCP connection established successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to establish MCP connection: {e}")
            raise

    def list_tools(self) -> Dict:
        """Fetches available tools with error handling."""
        try:
            result = self._send_request(method="tools/list", timeout=30)
            tools = result.get('tools', [])
            logger.info(f"Retrieved {len(tools)} tools from MCP server")
            return result
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise

    def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Enhanced tool calling with better logging."""
        try:
            logger.info(f"Calling tool '{tool_name}' with args: {arguments}")
            params = {"name": tool_name, "arguments": arguments}
            result = self._send_request(method="tools/call", params=params)
            logger.info(f"Tool '{tool_name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            raise

    def close(self):
        """Closes the session."""
        logger.info("Closing MCP client session")
        try:
            self._session.close()
        except Exception as e:
            logger.warning(f"Error closing MCP session: {e}")

class MCPTool(BaseTool):
    """Enhanced LangChain tool with better argument handling."""
    client: MCPClient
    tool_name: str
    datasource_luid: str

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute tool with intelligent argument handling and enhanced error reporting.
        """
        try:
            tool_args = kwargs.copy()
            
            # Handle cases where AI passes raw values instead of structured arguments
            if not tool_args and args:
                arg_schema_fields = list(self.args_schema.model_fields.keys())
                
                if len(arg_schema_fields) == 1:
                    tool_args = {arg_schema_fields[0]: args[0]}
                elif len(args) == len(arg_schema_fields):
                    # Map positional args to schema fields in order
                    tool_args = dict(zip(arg_schema_fields, args))

            logger.debug(f"Executing tool '{self.tool_name}' with processed args: {tool_args}")
            
            result = self.client.call_tool(self.tool_name, tool_args)
            
            # Process the result content
            content = result.get('content', '')
            
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and 'text' in content[0]:
                    try:
                        # Try to parse as JSON
                        actual_output = json.loads(content[0]['text'])
                        return actual_output
                    except json.JSONDecodeError:
                        # Return raw text if not JSON
                        return content[0]['text']
                else:
                    return content[0]
            
            return content
            
        except Exception as e:
            logger.error(f"Tool '{self.tool_name}' execution failed: {e}")
            raise RuntimeError(f"Tool execution failed: {e}")
    
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool asynchronously."""
        return self._run(*args, **kwargs)

def setup_langchain_tools(client: MCPClient, ds_metadata: Dict) -> list:
    """
    Enhanced tool setup with better error handling and validation.
    """
    datasource_luid = ds_metadata['luid']
    logger.info(f"Setting up LangChain tools for Datasource LUID: {datasource_luid}")
    
    try:
        available_tools_data = client.list_tools().get('tools', [])
        if not available_tools_data:
            raise RuntimeError("MCP server returned no available tools")

        # Enhanced type mapping
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        langchain_tools = []
        
        for tool_data in available_tools_data:
            tool_name = tool_data['name']
            logger.debug(f"Processing tool: {tool_name}")
            
            # Build Pydantic schema dynamically
            fields = {}
            input_schema = tool_data.get('inputSchema', {})
            properties = input_schema.get('properties', {})
            required_fields = input_schema.get('required', [])
            
            for prop_name, prop_details in properties.items():
                json_type = prop_details.get('type', 'string')
                python_type = type_mapping.get(json_type, str)
                
                # Handle required vs optional fields
                if prop_name in required_fields:
                    fields[prop_name] = (python_type, Field(..., description=prop_details.get('description', '')))
                else:
                    fields[prop_name] = (Optional[python_type], Field(None, description=prop_details.get('description', '')))
            
            # Create the schema
            args_schema = create_model(f'{tool_name}Args', **fields) if fields else create_model(f'{tool_name}Args')

            # Create the tool instance
            mcp_tool = MCPTool(
                client=client,
                tool_name=tool_name,
                datasource_luid=datasource_luid,
                name=tool_name,
                description=tool_data.get('description', f'Tool: {tool_name}'),
                args_schema=args_schema
            )
            langchain_tools.append(mcp_tool)

        logger.info(f"Successfully created {len(langchain_tools)} LangChain tools: {[t.name for t in langchain_tools]}")
        return langchain_tools
        
    except Exception as e:
        logger.error(f"Failed to setup LangChain tools: {e}")
        raise

# --- ENHANCED STREAMING CHAT ENDPOINT ---
async def stream_chat_events(message: str, ds_metadata: Dict):
    """
    Enhanced chat streaming with better error handling and progress reporting.
    """
    mcp_client = None
    
    try:
        # Import here to avoid startup issues
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import ChatPromptTemplate

        mcp_server_url = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"
        
        yield f"event: progress\ndata: {json.dumps({'message': 'Initializing MCP connection...'})}\n\n"
        
        mcp_client = MCPClient(server_url=mcp_server_url)
        mcp_client.connect()

        yield f"event: progress\ndata: {json.dumps({'message': 'Setting up AI tools...'})}\n\n"
        
        tools = setup_langchain_tools(mcp_client, ds_metadata)
        
        # Use a more reliable model
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            streaming=True,
            request_timeout=120
        )
        
        # Enhanced prompt with clearer instructions
        prompt_template = """
        You are a helpful Tableau data assistant with access to these tools:
        {tools}

        **IMPORTANT WORKFLOW - Follow these steps exactly:**
        
        1. **First, always call `list-datasources` to see available data sources**
        2. **From the results, find the datasource that best matches the user's question**
        3. **Extract the 'id' field from that datasource - this is the datasourceLuid**
        4. **Call `list-fields` with the datasourceLuid to see available columns**
        5. **Use `query-datasource` with the datasourceLuid and appropriate fields to get data**
        6. **Analyze the results and provide a clear answer**

        **Key Rules:**
        - Never make up field names or datasource IDs
        - Always use the exact 'id' values returned by the tools
        - If you can't find relevant data, explain what datasources are available
        - Be specific about which datasource you're using

        **Response Format:**
        Question: the input question you must answer
        Thought: your reasoning about what to do next
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (use exact field names and IDs)
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation as needed)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate"
        )

        yield f"event: progress\ndata: {json.dumps({'message': 'Starting AI analysis...'})}\n\n"
        
        final_response = ""
        
        # Stream the agent execution
        async for chunk in agent_executor.astream_log({"input": message}):
            try:
                if chunk.ops and len(chunk.ops) > 0:
                    op = chunk.ops[0]
                    if op.get("op") == "add":
                        log_entry_path = op.get("path", "")
                        
                        # Handle different types of log entries
                        if log_entry_path.startswith("/logs/") and "streamed_output" not in log_entry_path:
                            log_entry = op.get('value', {})
                            
                            if isinstance(log_entry, dict) and 'event' in log_entry:
                                event_type = log_entry['event']
                                event_data = log_entry.get('data', {})
                                
                                if event_type == 'on_chain_end' and 'output' in event_data:
                                    output = event_data.get('output', {})
                                    if isinstance(output, dict) and 'output' in output:
                                        final_answer = output['output']
                                        final_response = final_answer
                                        
                                        # Stream the response character by character
                                        for char in final_answer:
                                            yield f"event: token\ndata: {json.dumps({'token': char})}\n\n"
                                            await asyncio.sleep(0.005)  # Slightly faster streaming
                                
                                elif event_type == 'on_tool_start':
                                    tool_name = event_data.get('name', 'Unknown Tool')
                                    tool_input = event_data.get('input', {})
                                    progress_msg = f'Executing: {tool_name}'
                                    yield f"event: progress\ndata: {json.dumps({'message': progress_msg})}\n\n"
                                
                                elif event_type == 'on_tool_end':
                                    tool_name = event_data.get('name', 'Tool')
                                    yield f"event: progress\ndata: {json.dumps({'message': f'Completed: {tool_name}'})}\n\n"
                                    
            except Exception as chunk_error:
                logger.warning(f"Error processing chunk: {chunk_error}")
                continue
        
        if not final_response:
            final_response = "I apologize, but I encountered an issue processing your request. Please try rephrasing your question or check if the data source is accessible."
        
        yield f"event: result\ndata: {json.dumps({'response': final_response})}\n\n"

    except Exception as e:
        logger.error(f"Error during chat stream: {e}", exc_info=True)
        error_message = f"I encountered an error while processing your request: {str(e)}"
        yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n"
    
    finally:
        if mcp_client:
            mcp_client.close()
        yield f"event: done\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request):
    """Enhanced chat endpoint with better error handling."""
    try:
        client_id = fastapi_request.client.host
        ds_metadata = DATASOURCE_METADATA_STORE.get(client_id)
        
        if not ds_metadata:
            raise HTTPException(
                status_code=400, 
                detail="Datasource not initialized. Please reload the dashboard or reinitialize the datasource."
            )

        logger.info(f"Processing chat request from {client_id}: '{request.message[:100]}...'")
        
        return StreamingResponse(
            stream_chat_events(request.message, ds_metadata), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- TABLEAU AUTHENTICATION AND DATASOURCE LOGIC ---

def get_tableau_username(client_id: str) -> str:
    """Get Tableau username for the client."""
    username = os.environ.get('TABLEAU_USER')
    if not username:
        raise RuntimeError("TABLEAU_USER environment variable not set")
    return username

def generate_jwt(tableau_username: str) -> str:
    """Generate JWT token for Tableau authentication."""
    try:
        client_id = os.environ.get('TABLEAU_JWT_CLIENT_ID')
        secret_id = os.environ.get('TABLEAU_JWT_SECRET_ID')
        secret_value = os.environ.get('TABLEAU_JWT_SECRET')
        
        if not all([client_id, secret_id, secret_value]):
            raise RuntimeError("Missing required JWT environment variables")
        
        now = datetime.datetime.now(datetime.UTC)
        
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
        
    except Exception as e:
        logger.error(f"Failed to generate JWT: {e}")
        raise RuntimeError(f"JWT generation failed: {e}")

def tableau_signin_with_jwt(tableau_username: str):
    """Sign in to Tableau using JWT."""
    try:
        domain = os.environ.get('TABLEAU_DOMAIN_FULL')
        api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
        site = os.environ.get('TABLEAU_SITE')
        
        if not all([domain, site]):
            raise RuntimeError("Missing required Tableau environment variables")
        
        jwt_token = generate_jwt(tableau_username)
        url = f"{domain}/api/{api_version}/auth/signin"
        payload = {"credentials": {"jwt": jwt_token, "site": {"contentUrl": site}}}
        
        resp = requests.post(url, json=payload, headers={'Accept': 'application/json'}, timeout=30)
        resp.raise_for_status()
        
        creds = resp.json()['credentials']
        return creds['token'], creds['site']['id']
        
    except Exception as e:
        logger.error(f"Tableau signin failed: {e}")
        raise RuntimeError(f"Tableau authentication failed: {e}")

def lookup_published_luid_by_name(name: str, tableau_username: str):
    """Look up datasource LUID by name."""
    try:
        token, site_id = tableau_signin_with_jwt(tableau_username)
        domain = os.environ.get('TABLEAU_DOMAIN_FULL')
        api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
        
        url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
        headers = {"X-Tableau-Auth": token, "Accept": "application/json"}
        params = {"filter": f"name:eq:{name}"}
        
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        
        datasources = resp.json().get("datasources", {}).get("datasource", [])
        if not datasources:
            raise HTTPException(status_code=404, detail=f"Datasource '{name}' not found")
        
        return datasources[0]["id"], datasources[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Datasource lookup failed: {e}")
        raise RuntimeError(f"Failed to lookup datasource '{name}': {e}")

def debug_datasource_publishing_status(tableau_username: str, datasource_name: str):
    """
    Debug function to check if a datasource has published fields and why it might not.
    """
    try:
        token, site_id = tableau_signin_with_jwt(tableau_username)
        domain = os.environ.get('TABLEAU_DOMAIN_FULL')
        api_version = os.environ.get('TABLEAU_API_VERSION', '3.21')
        
        # Get datasource details
        url = f"{domain}/api/{api_version}/sites/{site_id}/datasources"
        headers = {"X-Tableau-Auth": token, "Accept": "application/json"}
        params = {"filter": f"name:eq:{datasource_name}"}
        
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        
        datasources = resp.json().get("datasources", {}).get("datasource", [])
        if not datasources:
            logger.error(f"Datasource '{datasource_name}' not found")
            return
            
        ds = datasources[0]
        ds_id = ds["id"]
        
        logger.info(f"Datasource found: {ds}")
        logger.info(f"Datasource ID: {ds_id}")
        logger.info(f"Content URL: {ds.get('contentUrl', 'N/A')}")
        logger.info(f"Project: {ds.get('project', {}).get('name', 'N/A')}")
        logger.info(f"Owner: {ds.get('owner', {}).get('name', 'N/A')}")
        
        # Try to get datasource connections
        connections_url = f"{domain}/api/{api_version}/sites/{site_id}/datasources/{ds_id}/connections"
        try:
            conn_resp = requests.get(connections_url, headers=headers, timeout=30)
            if conn_resp.status_code == 200:
                connections = conn_resp.json()
                logger.info(f"Datasource connections: {connections}")
            else:
                logger.warning(f"Could not get connections (status {conn_resp.status_code})")
        except Exception as e:
            logger.warning(f"Error getting connections: {e}")
            
        # Check if datasource has any workbooks using it
        workbooks_url = f"{domain}/api/{api_version}/sites/{site_id}/workbooks"
        try:
            wb_resp = requests.get(workbooks_url, headers=headers, timeout=30)
            if wb_resp.status_code == 200:
                workbooks = wb_resp.json().get("workbooks", {}).get("workbook", [])
                using_workbooks = [wb for wb in workbooks if any(
                    conn.get("datasource", {}).get("id") == ds_id 
                    for conn in wb.get("connections", {}).get("connection", [])
                )]
                logger.info(f"Workbooks using this datasource: {len(using_workbooks)}")
        except Exception as e:
            logger.warning(f"Error checking workbooks: {e}")
            
        return ds_id, ds
        
    except Exception as e:
        logger.error(f"Debug check failed: {e}")
        raise

# Add this to your datasource initialization
@app.post("/datasources")
async def receive_datasources(request: Request, body: Dict[str, Any]):
    """Enhanced datasource initialization endpoint with debugging."""
    try:
        client_id = request.client.host
        datasources_map = body.get("datasources", {})
        
        if not datasources_map:
            raise HTTPException(status_code=400, detail="No datasources provided in the request")

        first_name = next(iter(datasources_map.keys()))
        logger.info(f"Initializing datasource '{first_name}' for client {client_id}")
        
        tableau_username = get_tableau_username(client_id)
        
        # Add debugging
        logger.info("=== DEBUGGING DATASOURCE STATUS ===")
        debug_datasource_publishing_status(tableau_username, first_name)
        logger.info("=== END DEBUG ===")
        
        published_luid, full_metadata = lookup_published_luid_by_name(first_name, tableau_username)

        DATASOURCE_METADATA_STORE[client_id] = {
            "name": first_name,
            "luid": published_luid,
            "projectName": full_metadata.get("projectName"),
            "description": full_metadata.get("description"),
            "owner_name": full_metadata.get("owner", {}).get("name")
        }
        
        logger.info(f"Successfully initialized datasource for {client_id}: {first_name} (LUID: {published_luid})")
        return {"status": "success", "selected_datasource_luid": published_luid, "datasource_name": first_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Datasource initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize datasource: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/", include_in_schema=False)
async def home():
    """Serve the main page."""
    return FileResponse('static/index.html')

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# --- STARTUP VALIDATION ---
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    required_env_vars = [
        'TABLEAU_USER', 
        'TABLEAU_JWT_CLIENT_ID', 
        'TABLEAU_JWT_SECRET_ID', 
        'TABLEAU_JWT_SECRET',
        'TABLEAU_DOMAIN_FULL',
        'TABLEAU_SITE'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise RuntimeError(f"Missing environment variables: {missing_vars}")
    
    logger.info("Application startup complete - all required environment variables present")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)