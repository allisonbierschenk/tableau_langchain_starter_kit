"""Route tools/call to the correct MCP client by server id."""

from __future__ import annotations

from typing import Any, Dict, Optional

from orchestration.types import MCPServerId


class MCPRouter:
    def __init__(self, clients: Dict[MCPServerId, Any]):
        if MCPServerId.TABLEAU_MCP not in clients:
            raise ValueError("MCPRouter requires a tableau_mcp client")
        self._clients = dict(clients)

    @property
    def clients(self) -> Dict[MCPServerId, Any]:
        return self._clients

    async def call_tool(self, server_id: MCPServerId, tool_name: str, arguments: dict) -> Any:
        client = self._clients.get(server_id)
        if not client:
            raise ValueError(f"No MCP client registered for {server_id}")
        return await client.call_tool(tool_name, arguments)

    async def list_tools_for(self, server_id: MCPServerId):
        client = self._clients.get(server_id)
        if not client:
            return []
        return await client.list_tools()

    @classmethod
    def with_tableau_and_optional_slack(cls, tableau_client: Any, slack_client: Optional[Any]) -> "MCPRouter":
        m: Dict[MCPServerId, Any] = {MCPServerId.TABLEAU_MCP: tableau_client}
        if slack_client is not None:
            m[MCPServerId.SLACK_MCP] = slack_client
        return cls(m)
