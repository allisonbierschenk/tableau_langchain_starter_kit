import os
from typing import Any, Dict, List

import httpx


class AdminApiClient:
    """Small client for repo admin-agent API calls."""

    def __init__(self) -> None:
        self.base_url = os.getenv("ADMIN_API_BASE_URL", "").rstrip("/")
        self.path = os.getenv("ADMIN_API_PATH", "/mcp-chat")
        self.bearer_token = os.getenv("ADMIN_API_BEARER_TOKEN", "")
        timeout_value = float(os.getenv("ADMIN_API_TIMEOUT_SECONDS", "45"))
        self.timeout = httpx.Timeout(timeout_value)

        if not self.base_url:
            raise ValueError("ADMIN_API_BASE_URL is required")

    async def chat(self, message: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        url = f"{self.base_url}{self.path}"
        payload = {
            "message": message,
            "history": history,
        }

        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)

        # Raise with body context when possible to help demo troubleshooting.
        if response.status_code >= 400:
            body_text = response.text[:500]
            raise RuntimeError(
                f"Admin API call failed ({response.status_code}): {body_text}"
            )

        data = response.json()
        return {
            "response": data.get("response", ""),
            "tool_results": data.get("tool_results", []),
            "iterations": data.get("iterations", 0),
        }
