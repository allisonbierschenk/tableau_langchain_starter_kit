"""Shared viewer email detection for web app and MCP client headers."""

import re
from typing import Optional


def is_likely_email(value: Optional[str]) -> bool:
    if not value or not str(value).strip():
        return False
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", str(value).strip()))


def pick_viewer_email(
    workbook_user_email: Optional[str], workbook_current_user: Optional[str]
) -> Optional[str]:
    for candidate in (workbook_user_email, workbook_current_user):
        if candidate and is_likely_email(candidate):
            return str(candidate).strip()
    return None
