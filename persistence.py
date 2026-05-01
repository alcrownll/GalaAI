"""
persistence.py — File-based chat history persistence for Gala.AI.
Replaces the fragile localStorage + query-param approach.
"""

import json
import pathlib

CHAT_FILE = pathlib.Path(__file__).parent / "gala_chats.json"


def load_chats() -> list:
    """Load all saved chats from disk. Returns empty list on any error."""
    if not CHAT_FILE.exists():
        return []
    try:
        data = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_chats(chats: list) -> None:
    """Persist all chats to disk. Silently swallows errors."""
    try:
        CHAT_FILE.write_text(
            json.dumps(chats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass