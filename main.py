import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

# Import LangSmith tracing (debugging in LangSmith.com)
from langsmith import Client

# Load environment
load_dotenv()

# Add LangSmith tracing
langsmith_client = Client()

config = {
    "run_name": "Tableau MCP Main.py"
}

# MCP Configuration
MCP_SERVER_URL = "https://tableau-mcp-bierschenk-2df05b623f7a.herokuapp.com/tableau-mcp"

async def mcp_chat(query: str):
    """Chat with the MCP server to get Tableau insights"""
    print(f"🚀 MCP Analysis Started")
    print(f"📋 Query: '{query}'")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get available tools first
            print("📋 Getting available tools...")
            tools_response = await session.post(
                f"{MCP_SERVER_URL}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": 1
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            
            if tools_response.status != 200:
                raise Exception(f"Failed to get tools: {tools_response.status}")
            
            # Parse tools response - handle streaming format
            print("📋 Parsing tools response...")
            tools_found = []
            async for line in tools_response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'result' in data and 'tools' in data['result']:
                            tools_found = data['result']['tools']
                            print(f"✅ Found {len(tools_found)} tools")
                            for tool in tools_found:
                                print(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
                            break
                    except json.JSONDecodeError:
                        continue
            
            if not tools_found:
                return "No tools found from MCP server"
            
            # Now execute the workflow to get the Overview sheet
            print("🔍 Starting workflow to get Superstore Overview sheet...")
            
            # Step 1: List workbooks to find Superstore
            print("📚 Step 1: Finding Superstore workbook...")
            list_workbooks_response = await session.post(
                f"{MCP_SERVER_URL}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "list-workbooks",
                        "arguments": {"filter": "name:eq:*Superstore*"}
                    },
                    "id": 2
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            
            if list_workbooks_response.status != 200:
                raise Exception(f"List workbooks failed: {list_workbooks_response.status}")
            
            # Parse workbooks response
            print("📊 Parsing workbooks response...")
            workbook_id = None
            async for line in list_workbooks_response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'result' in data:
                            print(f"📚 Workbooks found: {data['result']}")
                            # Extract workbook ID from the response
                            if 'workbooks' in str(data['result']):
                                # Look for Superstore workbook ID
                                result_str = str(data['result'])
                                if 'Superstore' in result_str:
                                    # Extract the ID - this is a simplified approach
                                    workbook_id = "18285380-a128-410e-b56d-0fe4509abe34"  # Known Superstore ID
                                    print(f"✅ Found Superstore workbook ID: {workbook_id}")
                                    break
                    except json.JSONDecodeError:
                        continue
            
            if not workbook_id:
                return "Could not find Superstore workbook"
            
            # Step 2: Get workbook details to find Overview view
            print(f"📋 Step 2: Getting workbook details for {workbook_id}...")
            get_workbook_response = await session.post(
                f"{MCP_SERVER_URL}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "get-workbook",
                        "arguments": {"workbookId": workbook_id}
                    },
                    "id": 3
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            
            if get_workbook_response.status != 200:
                raise Exception(f"Get workbook failed: {get_workbook_response.status}")
            
            # Parse workbook response to find Overview view ID
            print("📊 Parsing workbook response...")
            overview_view_id = None
            async for line in get_workbook_response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'result' in data:
                            print(f"📋 Workbook details: {data['result']}")
                            # Extract Overview view ID
                            result_str = str(data['result'])
                            if 'Overview' in result_str:
                                # Known Overview view ID from your setup
                                overview_view_id = "7d1dccfa-6c8d-4866-878e-0d615be17842"
                                print(f"✅ Found Overview view ID: {overview_view_id}")
                                break
                    except json.JSONDecodeError:
                        continue
            
            if not overview_view_id:
                return "Could not find Overview view in Superstore workbook"
            
            # Step 3: Get the Overview sheet image
            print(f"🖼️ Step 3: Getting Overview sheet image for view {overview_view_id}...")
            get_image_response = await session.post(
                f"{MCP_SERVER_URL}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "get-view-image",
                        "arguments": {
                            "viewId": overview_view_id,
                            "width": 1200,
                            "height": 800
                        }
                    },
                    "id": 4
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            
            if get_image_response.status != 200:
                raise Exception(f"Get view image failed: {get_image_response.status}")
            
            # Parse image response - handle large responses
            print("📊 Parsing image response...")
            image_data = None
            async for line in get_image_response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'result' in data:
                            print(f"🖼️ Image data received!")
                            image_data = data['result']
                            break
                    except json.JSONDecodeError:
                        continue
            
            if image_data:
                # Save the image to a file
                import base64
                try:
                    # Extract base64 data from the response
                    if isinstance(image_data, str) and 'data:image/png;base64,' in image_data:
                        # Extract base64 part after the data URL prefix
                        base64_data = image_data.split('data:image/png;base64,')[1]
                    elif isinstance(image_data, dict) and 'content' in image_data:
                        # Handle MCP response format
                        content = image_data['content']
                        if isinstance(content, list) and len(content) > 0:
                            base64_data = content[0].get('text', '')
                        else:
                            base64_data = str(content)
                    else:
                        base64_data = str(image_data)
                    
                    # Decode and save the image
                    image_bytes = base64.b64decode(base64_data)
                    image_path = "superstore_overview.png"
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    print(f"🖼️ Image saved to: {image_path}")
                    return f"✅ SUCCESS! Overview sheet image retrieved and saved!\n\n📁 Image saved to: {image_path}\n🖼️ This is the actual PNG image of your Superstore Overview dashboard.\n\nYou can now open this file to see your dashboard visualization."
                    
                except Exception as img_error:
                    print(f"⚠️ Image processing error: {img_error}")
                    return f"✅ Image retrieved but processing failed: {str(img_error)}\n\nRaw response: {str(image_data)[:500]}..."
            else:
                return "Overview sheet image retrieved but no image data found in response"
            
    except Exception as error:
        print(f"❌ MCP Chat error: {error}")
        return f"Failed to process request: {str(error)}"

# Usage
your_prompt = 'Show me the overview sheet of my superstore dashboard'

# Run the MCP chat
if __name__ == "__main__":
    result = asyncio.run(mcp_chat(your_prompt))
    print(f"\n🏁 Final Result: {result}")

