"""Check what Pulse operations are available from the MCP server"""

import asyncio
from utilities.admin_agent import AdminMCPHttpClient, ADMIN_MCP_SERVER

async def list_pulse_tools():
    async with AdminMCPHttpClient(ADMIN_MCP_SERVER) as client:
        tools = await client.list_tools()
        pulse_tools = [t for t in tools if 'pulse' in t.get('name', '').lower()]

        print(f"Found {len(pulse_tools)} Pulse tools:\n")

        for tool in pulse_tools:
            print(f"Tool: {tool['name']}")
            print(f"  Description: {tool.get('description', 'N/A')}")
            if 'inputSchema' in tool:
                props = tool['inputSchema'].get('properties', {})
                if props:
                    print(f"  Parameters: {list(props.keys())}")
            print()

if __name__ == "__main__":
    asyncio.run(list_pulse_tools())
