"""Claude API key storage for the AI DJ service — a row in the Running app's
shared pacesync.db (kv_config table, key "claude_config"), written by the
Running app's Settings page (POST /settings/ai-dj/claude-key) and read here
so the mixer can use it without an env var.
"""

import json
import os
import sqlite3

_DB_PATH = os.environ.get(
    "AI_DJ_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pacesync.db"),
)
_KEY = "claude_config"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=5)
    conn.execute("PRAGMA busy_timeout = 5000")
    # Idempotent — matches lib/db.ts's schema exactly, in case this
    # subprocess runs before the Node app has ever opened the DB.
    conn.execute("CREATE TABLE IF NOT EXISTS kv_config (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
    return conn


def load_claude_api_key() -> str | None:
    try:
        conn = _connect()
        try:
            row = conn.execute("SELECT value_json FROM kv_config WHERE key = ?", (_KEY,)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        data = json.loads(row[0])
        key = data.get("apiKey")
        return key or None
    except (sqlite3.Error, FileNotFoundError, json.JSONDecodeError):
        return None


def save_claude_api_key(api_key: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO kv_config (key, value_json) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
            (_KEY, json.dumps({"apiKey": api_key})),
        )
        conn.commit()
    finally:
        conn.close()
