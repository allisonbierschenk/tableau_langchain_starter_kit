import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from utilities.slack_admin_api_client import AdminApiClient

load_dotenv()


def _parse_allowed_users(raw_ids: str) -> set[str]:
    return {user_id.strip() for user_id in raw_ids.split(",") if user_id.strip()}


def _is_destructive_command(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "delete user",
        "remove user",
        "delete group",
        "remove group",
        "remove users",
        "delete users",
    ]
    return any(keyword in lowered for keyword in keywords)


def _strip_app_mention(text: str) -> str:
    # Removes first mention token from app_mention events.
    if not text:
        return ""
    parts = text.split(">", 1)
    if len(parts) == 2 and parts[0].startswith("<@"):
        return parts[1].strip()
    return text.strip()


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_ALLOWED_USER_IDS_RAW = os.getenv("SLACK_ALLOWED_USER_IDS", "")
ALLOW_ALL_USERS = SLACK_ALLOWED_USER_IDS_RAW.strip() == "*"
SLACK_ALLOWED_USER_IDS = _parse_allowed_users(SLACK_ALLOWED_USER_IDS_RAW)
MAX_HISTORY_ITEMS = int(os.getenv("SLACK_MAX_HISTORY_ITEMS", "20"))

if not SLACK_BOT_TOKEN:
    raise ValueError("SLACK_BOT_TOKEN is required")
if not SLACK_APP_TOKEN:
    raise ValueError("SLACK_APP_TOKEN is required for Socket Mode")

if not ALLOW_ALL_USERS and not SLACK_ALLOWED_USER_IDS:
    raise ValueError(
        "SLACK_ALLOWED_USER_IDS is required (comma-separated IDs) or set to '*'"
    )


app = AsyncApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
admin_client = AdminApiClient()

# In-memory state is enough for demo usage.
conversation_history: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
pending_confirmations: Dict[str, str] = {}


async def _run_admin_request(say, user_id: str, channel_id: str, prompt_text: str) -> None:
    key = (channel_id, user_id)
    history = conversation_history[key]

    try:
        result = await admin_client.chat(message=prompt_text, history=history)
    except Exception as exc:
        await say(
            text=(
                "I could not reach the admin API. "
                f"Error: {exc}"
            )
        )
        return

    response_text = result.get("response", "").strip() or "No response returned."
    await say(text=response_text)

    # Keep short rolling memory for the demo.
    history.append({"role": "user", "content": prompt_text})
    history.append({"role": "assistant", "content": response_text})
    if len(history) > MAX_HISTORY_ITEMS:
        conversation_history[key] = history[-MAX_HISTORY_ITEMS:]


async def _handle_text(say, user_id: str, channel_id: str, raw_text: str) -> None:
    if not ALLOW_ALL_USERS and user_id not in SLACK_ALLOWED_USER_IDS:
        await say(
            text=(
                "You are not authorized for this demo bot. "
                "Ask the owner to add your Slack user ID to `SLACK_ALLOWED_USER_IDS`."
            )
        )
        return

    text = raw_text.strip()
    if not text:
        await say(text="Send a user/group admin request, for example: `list all users`.")
        return

    if user_id in pending_confirmations:
        lowered = text.lower()
        original_request = pending_confirmations[user_id]
        if lowered in {"yes", "y", "confirm"}:
            del pending_confirmations[user_id]
            await say(text=f"Confirmed. Running: `{original_request}`")
            await _run_admin_request(say, user_id, channel_id, original_request)
            return
        if lowered in {"no", "n", "cancel"}:
            del pending_confirmations[user_id]
            await say(text="Canceled.")
            return
        await say(text="Reply `yes` to confirm, or `no` to cancel.")
        return

    if _is_destructive_command(text):
        pending_confirmations[user_id] = text
        await say(
            text=(
                "This looks destructive (remove/delete). "
                "Reply `yes` to confirm or `no` to cancel."
            )
        )
        return

    await _run_admin_request(say, user_id, channel_id, text)


@app.event("app_mention")
async def handle_app_mention(event, say):
    user_id = event.get("user", "")
    channel_id = event.get("channel", "")
    text = _strip_app_mention(event.get("text", ""))
    await _handle_text(say, user_id, channel_id, text)


@app.event("message")
async def handle_message_events(event, say):
    # Handle only direct messages from users.
    if event.get("channel_type") != "im":
        return
    if event.get("subtype"):
        return

    user_id = event.get("user", "")
    channel_id = event.get("channel", "")
    text = event.get("text", "").strip()
    await _handle_text(say, user_id, channel_id, text)


async def main() -> None:
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
