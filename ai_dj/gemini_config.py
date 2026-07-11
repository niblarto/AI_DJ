"""Gemini API key storage — same pattern as claude_config.py: a small JSON
file next to the service, written by the Running app's Settings page and
read here so the mixer can use it without an env var.
"""

import json
import os

_CONFIG_PATH = os.environ.get(
    "AI_DJ_GEMINI_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gemini-config.json"),
)


def load_gemini_api_key() -> str | None:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        key = data.get("apiKey")
        return key or None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_gemini_api_key(api_key: str) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"apiKey": api_key}, f)
