"""
Generic HTTP JSON-RPC MCP client (tools/list, tools/call) with SSE-shaped responses.

Used for Slack MCP (or any second MCP) with optional auth headers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx


class MCPJsonRpcSseClient:
    """Minimal async MCP over HTTP+SSE-style lines (data: {...})."""

    def __init__(
        self,
        server_url: str,
        *,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 45.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        self.timeout_seconds = timeout_seconds
        self._session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "MCPJsonRpcSseClient":
        self._session = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None

    def _parse_sse_response(self, response_text: str) -> Any:
        for line in response_text.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "result" in data:
                        return data["result"]
                except json.JSONDecodeError:
                    continue
        return None

    async def _post_jsonrpc(self, method: str, params: dict) -> Any:
        if not self._session:
            raise RuntimeError("MCPJsonRpcSseClient not initialized (use async with)")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.extra_headers,
        }
        response = await self._session.post(
            f"{self.server_url}/",
            json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
            headers=headers,
        )
        if response.status_code != 200:
            raise RuntimeError(f"MCP {method} failed: HTTP {response.status_code}")
        return self._parse_sse_response(response.text)

    async def list_tools(self) -> List[Dict[str, Any]]:
        result = await self._post_jsonrpc("tools/list", {})
        if not result:
            return []
        return result.get("tools", []) if isinstance(result, dict) else []

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        return await self._post_jsonrpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
