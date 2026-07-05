"""Minimal Ollama chat client (JSON-mode responses only)."""

import json
import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_DJ_MODEL", "mistral-nemo:12b")


class OllamaError(RuntimeError):
    pass


def chat_json(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: int = 300,
) -> dict:
    """Send a chat request with format=json and return the parsed JSON reply."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                # num_ctx: the default 4096 truncates large candidate-pool prompts,
                # which makes the model loop; num_predict caps runaway JSON output.
                # 9728 is the measured ceiling for mistral-nemo:12b to stay 100% on a
                # 10GB GPU (10240 already spills ~7% to CPU) - see README Setup notes.
                "options": {
                    "temperature": temperature,
                    "num_ctx": 9728,
                    "num_predict": 4096,
                },
            },
            timeout=timeout,
        )
    except requests.ConnectionError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_URL} - is it running? (`ollama serve` "
            "or launch the Ollama app)"
        ) from e

    if resp.status_code == 404:
        raise OllamaError(f"Model '{model}' not found - run: ollama pull {model}")
    resp.raise_for_status()

    content = resp.json().get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise OllamaError(f"Model returned non-JSON output: {content[:200]!r}") from e
