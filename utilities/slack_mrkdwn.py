"""
Slack uses mrkdwn: *bold*, not Markdown **bold**.
"""

import re


def llm_markdown_to_slack_mrkdwn(text: str) -> str:
    if not text:
        return text
    out = text
    for _ in range(100):
        next_out = re.sub(r"\*\*(.+?)\*\*", r"*\1*", out, flags=re.DOTALL)
        if next_out == out:
            break
        out = next_out
    return out
