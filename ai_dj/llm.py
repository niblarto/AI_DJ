"""Chat-JSON client for the three backends the mixer can use: a local Ollama
model, the Claude API, or the Gemini API. All expose the same
chat_json(system, user, model, temperature) -> dict shape, so callers
(selector.py) don't need to know which backend is live.
"""

import json
import os
import threading

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_DJ_MODEL", "qwen3.5:9b")

# Claude models selectable from Settings. Sonnet 5 is the default per
# SKILL.md guidance for cost-sensitive production workloads; Opus is offered
# for users who want the ceiling.
CLAUDE_MODELS = {
    "claude-sonnet-5": "Sonnet 5",
    "claude-opus-4-8": "Opus 4.8",
    "claude-haiku-4-5": "Haiku 4.5",
}
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_CLAUDE_EFFORT = "medium"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Gemini models selectable from Settings — Flash tier only: gemini-2.5-pro
# has no free-tier access, and these two cover fast/cheap vs a bit more
# capable while staying on the free tier.
GEMINI_MODELS = {
    "gemini-2.5-flash": "Flash 2.5",
    "gemini-2.5-flash-lite": "Flash 2.5 Lite",
}
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Rough $/1M token pricing for the session cost estimate shown in Settings —
# not billing-accurate (no cache discount applied), just a ballpark. Gemini
# free-tier calls cost $0; these are the paid-tier rates for reference if
# usage moves off the free tier.
_CLAUDE_PRICING = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
_GEMINI_PRICING = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
_PRICING = {**_CLAUDE_PRICING, **_GEMINI_PRICING}


class OllamaError(RuntimeError):
    pass


class ClaudeError(RuntimeError):
    pass


class GeminiError(RuntimeError):
    pass


def is_claude_model(model: str) -> bool:
    return model in CLAUDE_MODELS


def is_gemini_model(model: str) -> bool:
    return model in GEMINI_MODELS


# ── Usage tracking (hosted providers: Claude, Gemini) ────────────────────────
# Persisted to a JSON file next to claude-config.json/gemini-config.json
# rather than kept in memory: mixes run as a short-lived `ai_dj_bridge.py`
# subprocess per mix (no long-running service process to hold counters
# across calls) when built on the Pi directly, so a file is the only thing
# that accumulates across mixes. "Session" here means "since this file was
# last cleared", not a provider-account concept — the mixer calls the API
# with a key, which has no claude.ai/aistudio-style session limit;
# token/cost counters are what applies. Keyed by model, since each has
# different per-token pricing.

_usage_lock = threading.Lock()
_usage_path = os.environ.get(
    "AI_DJ_USAGE_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai-dj-usage.json"),
)


def _record_usage(model: str, input_tokens: int, output_tokens: int):
    with _usage_lock:
        usage = get_usage()
        u = usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "requests": 0, "errors": 0})
        u["input_tokens"] += input_tokens
        u["output_tokens"] += output_tokens
        u["requests"] += 1
        try:
            with open(_usage_path, "w", encoding="utf-8") as f:
                json.dump(usage, f)
        except OSError:
            pass  # usage tracking is best-effort — never fail the mix over it


def _record_error(model: str, message: str):
    # A failed call (rate limit, quota, network) never reaches _record_usage
    # since there's no response to read tokens from — without this, a model
    # that's 429ing all day silently shows "no calls made" in Settings.
    with _usage_lock:
        usage = get_usage()
        u = usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "requests": 0, "errors": 0})
        u["errors"] += 1
        u["last_error"] = message[:300]
        try:
            with open(_usage_path, "w", encoding="utf-8") as f:
                json.dump(usage, f)
        except OSError:
            pass


def get_usage() -> dict[str, dict]:
    try:
        with open(_usage_path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Back-compat alias — earlier code/routes referred to this as Claude-only
# before Gemini usage was tracked in the same file.
get_claude_usage = get_usage


def estimate_cost_usd(model: str, usage: dict) -> float:
    in_price, out_price = _PRICING.get(model, (0.0, 0.0))
    return (usage["input_tokens"] / 1_000_000) * in_price + (usage["output_tokens"] / 1_000_000) * out_price


# ── Ollama backend ────────────────────────────────────────────────────────────

def _chat_json_ollama(system: str, user: str, model: str, temperature: float, timeout: int) -> dict:
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
                # Reasoning models (qwen3.5, deepseek-r1, etc.) route their
                # output into message.thinking instead of message.content by
                # default — with a hidden reasoning budget eating num_predict,
                # content comes back empty. think:false forces direct output.
                "think": False,
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


# ── Claude backend ────────────────────────────────────────────────────────────

_claude_client = None
_claude_client_lock = threading.Lock()


def _get_claude_client():
    global _claude_client
    with _claude_client_lock:
        if _claude_client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ClaudeError("anthropic package not installed - run: pip install anthropic") from e
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                from .claude_config import load_claude_api_key
                api_key = load_claude_api_key()
            if not api_key:
                raise ClaudeError("No Claude API key configured (Settings -> AI DJ -> Claude)")
            _claude_client = anthropic.Anthropic(api_key=api_key)
    return _claude_client


def _chat_json_claude(system: str, user: str, model: str, effort: str, timeout: int) -> dict:
    client = _get_claude_client()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": effort,
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object"},
                },
            },
            timeout=timeout,
        )
    except Exception as e:
        _record_error(model, str(e))
        raise ClaudeError(f"Claude API request failed: {e}") from e

    _record_usage(model, response.usage.input_tokens, response.usage.output_tokens)

    if response.stop_reason == "refusal":
        _record_error(model, "safety refusal")
        raise ClaudeError("Claude declined the request (safety refusal)")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ClaudeError(f"Claude returned non-JSON output: {text[:200]!r}") from e


# ── Gemini backend ────────────────────────────────────────────────────────────

_gemini_client = None
_gemini_client_lock = threading.Lock()


def _get_gemini_client():
    global _gemini_client
    with _gemini_client_lock:
        if _gemini_client is None:
            try:
                from google import genai
            except ImportError as e:
                raise GeminiError("google-genai package not installed - run: pip install google-genai") from e
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                from .gemini_config import load_gemini_api_key
                api_key = load_gemini_api_key()
            if not api_key:
                raise GeminiError("No Gemini API key configured (Settings -> AI DJ -> Gemini)")
            _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _chat_json_gemini(system: str, user: str, model: str, timeout: int) -> dict:
    client = _get_gemini_client()
    try:
        from google.genai import types
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=timeout * 1000),  # ms
            ),
        )
    except Exception as e:
        _record_error(model, str(e))
        raise GeminiError(f"Gemini API request failed: {e}") from e

    usage = response.usage_metadata
    if usage is not None:
        _record_usage(model, usage.prompt_token_count or 0, usage.candidates_token_count or 0)

    text = response.text
    if not text:
        raise GeminiError("Gemini returned an empty response (possibly blocked by safety filters)")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise GeminiError(f"Gemini returned non-JSON output: {text[:200]!r}") from e


# ── Dispatch ──────────────────────────────────────────────────────────────────

# ── LLM call log ──────────────────────────────────────────────────────────────
# Rolling log of every prompt sent during tracklist creation, shown at the
# bottom of the Settings AI DJ card (same idea as the GarminDB sync log).
# Written wherever the call runs: Claude/Gemini mixes run on the Pi so their
# entries land next to the app; Ollama calls run on the remote service PC,
# so those entries stay on that machine.

_LLM_LOG_PATH = os.environ.get(
    "AI_DJ_LLM_LOG_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai-dj-llm-log.json"),
)
_LLM_LOG_MAX = 50
_PROMPT_LOG_CHARS = 4000


def get_llm_log() -> list[dict]:
    try:
        with open(_LLM_LOG_PATH, encoding="utf-8") as f:
            log = json.load(f)
        return log if isinstance(log, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _log_llm_call(entry: dict):
    with _usage_lock:
        try:
            with open(_LLM_LOG_PATH, encoding="utf-8") as f:
                log = json.load(f)
            if not isinstance(log, list):
                log = []
        except (FileNotFoundError, json.JSONDecodeError):
            log = []
        log.append(entry)
        try:
            with open(_LLM_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(log[-_LLM_LOG_MAX:], f)
        except OSError:
            pass  # logging is best-effort — never fail the mix over it


def chat_json(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: int = 300,
    effort: str = DEFAULT_CLAUDE_EFFORT,
) -> dict:
    """Send a chat request and return the parsed JSON reply.

    `model` selects the backend: a Claude model ID (see CLAUDE_MODELS) routes
    to the Claude API using `effort`; a Gemini model ID (see GEMINI_MODELS)
    routes to the Gemini API; anything else is treated as an Ollama model
    tag and uses `temperature`.
    """
    import datetime
    import time

    entry = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "system": system[:_PROMPT_LOG_CHARS] + ("…" if len(system) > _PROMPT_LOG_CHARS else ""),
        "prompt": user[:_PROMPT_LOG_CHARS] + ("…" if len(user) > _PROMPT_LOG_CHARS else ""),
    }
    start = time.monotonic()
    try:
        if is_claude_model(model):
            result = _chat_json_claude(system, user, model, effort, timeout)
        elif is_gemini_model(model):
            result = _chat_json_gemini(system, user, model, timeout)
        else:
            result = _chat_json_ollama(system, user, model, temperature, timeout)
    except Exception as e:
        entry.update(ok=False, error=str(e)[:300], durationMs=int((time.monotonic() - start) * 1000))
        _log_llm_call(entry)
        raise
    entry.update(ok=True, durationMs=int((time.monotonic() - start) * 1000))
    _log_llm_call(entry)
    return result
