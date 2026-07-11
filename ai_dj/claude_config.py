"""Claude API key storage for the AI DJ service — a small JSON file next to
the service, written by the Running app's Settings page (POST /settings/ai-dj)
and read here so the mixer can use it without an env var.
"""

import json
import os

_CONFIG_PATH = os.environ.get(
    "AI_DJ_CLAUDE_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "claude-config.json"),
)


def load_claude_api_key() -> str | None:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        key = data.get("apiKey")
        return key or None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_claude_api_key(api_key: str) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"apiKey": api_key}, f)
