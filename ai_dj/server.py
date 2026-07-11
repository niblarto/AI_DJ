"""HTTP service exposing the workout-mix builder to the Running app.

The Running app (on the Pi) POSTs a workout plus its library CSV; this
service returns an ordered, pace-matched setlist of Spotify URIs. Run it
wherever Ollama lives (the Windows PC), or anywhere with --no-llm since the
distance-chain fallback needs no GPU:

    python -m ai_dj.server [--port 8765] [--no-llm]

POST /mix
    {
      "title": "Steady into Tempo",
      "segments": ["1mi warm up ...", "1.5mi at 8:35/mi", ...],
      "csv": "<Exportify CSV text>",
      "easyPace": "9:15",              // optional
      "useLlm": true,                  // optional
      "cadenceBuckets": {"555": 171}   // optional: sec/mi pace bucket -> SPM,
                                       // from the caller's GarminDB (the
                                       // service host has no Garmin data)
    }
->  {
      "trackUris": ["spotify:track:...", ...],
      "totalSec": 2457,
      "timeline": [{"segment", "targetBpm", "tracks": [{uri, name, artist,
                    startsAt, tempo, camelot, energy}]}]
    }

GET /health -> {"ok": true, "llm": true|false, "claude": true|false}
GET /usage -> {"models": {"claude-sonnet-5": {"inputTokens", "outputTokens", "requests", "estimatedCostUsd"}, ...}}
POST /settings/claude-key {"apiKey": "sk-ant-..."} -> {"ok": true}
"""

import argparse
import io
import json
import queue
import sys
import threading

import pandas as pd
from flask import Flask, Response, jsonify, request

from bpm_matcher.camelot import to_camelot

from .llm import (
    CLAUDE_MODELS,
    DEFAULT_CLAUDE_EFFORT,
    DEFAULT_MODEL,
    estimate_cost_usd,
    get_claude_usage,
    get_llm_log,
    is_claude_model,
)
from .workout import DEFAULT_EASY_PACE, build_workout_playlist, max_projected_duration, parse_workout

app = Flask(__name__)
app.config["USE_LLM"] = True
app.config["MODEL"] = DEFAULT_MODEL


def _load_library(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    df = df.dropna(subset=["Tempo"]).drop_duplicates(subset=["Track URI"]).reset_index(drop=True)
    df["Camelot"] = [to_camelot(k, m) for k, m in zip(df["Key"], df["Mode"])]
    return df


def _parse_pace(text: str | None) -> float:
    if not text:
        return DEFAULT_EASY_PACE
    m, s = str(text).split(":")
    return int(m) * 60 + int(s)


@app.get("/health")
def health():
    llm_ok = False
    if app.config["USE_LLM"]:
        try:
            import requests as rq

            from .llm import OLLAMA_URL

            llm_ok = rq.get(f"{OLLAMA_URL}/api/tags", timeout=3).ok
        except Exception:
            pass

    claude_ok = False
    try:
        from .llm import _get_claude_client
        _get_claude_client()
        claude_ok = True
    except Exception:
        pass

    return jsonify({"ok": True, "llm": llm_ok, "claude": claude_ok, "claudeModels": CLAUDE_MODELS})


@app.post("/settings/claude-key")
def set_claude_key():
    """Saves the Claude API key sent by the Running app's Settings page.
    Also drops the cached client so the next request picks up the new key."""
    body = request.get_json(force=True) or {}
    api_key = (body.get("apiKey") or "").strip()
    if not api_key:
        return jsonify({"error": "apiKey required"}), 400

    from . import claude_config, llm
    claude_config.save_claude_api_key(api_key)
    with llm._claude_client_lock:
        llm._claude_client = None
    return jsonify({"ok": True})


@app.get("/llm-log")
def llm_log():
    """Prompts sent to the LLM on this host (Ollama mixes run here) — merged
    with the Pi's own log by the Running app's Settings page."""
    return jsonify({"entries": get_llm_log()})


@app.get("/usage")
def usage():
    """Claude API token usage per model since this service process started,
    for the Settings page's usage bars — see lib/ai-dj-config.ts / SettingsClient."""
    by_model = get_claude_usage()
    return jsonify({
        "models": {
            model: {
                "inputTokens": u["input_tokens"],
                "outputTokens": u["output_tokens"],
                "requests": u["requests"],
                "estimatedCostUsd": round(estimate_cost_usd(model, u), 4),
            }
            for model, u in by_model.items()
        },
    })


def _build_mix_payload(body: dict, progress=None) -> tuple[dict, int]:
    """Shared /mix and /mix/stream builder — returns (payload, http_status)."""
    segments_text = body.get("segments") or []
    csv_text = body.get("csv") or ""
    if not segments_text or not csv_text:
        return {"error": "segments and csv are required"}, 400

    try:
        library = _load_library(csv_text)
    except Exception as e:
        return {"error": f"Could not parse library CSV: {e}"}, 400

    segments = parse_workout(segments_text, easy_pace_sec=_parse_pace(body.get("easyPace")))
    if not segments:
        return {"error": "No runnable segments recognized in the workout"}, 400

    buckets = None
    raw_buckets = body.get("cadenceBuckets")
    if isinstance(raw_buckets, dict):
        try:
            buckets = {int(k): float(v) for k, v in raw_buckets.items()} or None
        except (TypeError, ValueError):
            buckets = None

    try:
        easy_bias = float(body.get("easyBias") or 0.0)
    except (TypeError, ValueError):
        easy_bias = 0.0

    feedback = body.get("trackFeedback")
    if not isinstance(feedback, list):
        feedback = None

    played = body.get("playedTracks")
    if not isinstance(played, list):
        played = None

    bpm_overrides = body.get("bpmOverrides")
    if not isinstance(bpm_overrides, dict):
        bpm_overrides = None

    avoid = body.get("avoidTracks")
    if not isinstance(avoid, list):
        avoid = None

    # Caller (Running app Settings) can pick a Claude model instead of the
    # server-startup Ollama default; effort only applies to Claude.
    model = body.get("model") or app.config["MODEL"]
    effort = (body.get("effort") or DEFAULT_CLAUDE_EFFORT) if is_claude_model(model) else None

    use_llm = app.config["USE_LLM"] and body.get("useLlm", True)
    try:
        playlist = build_workout_playlist(
            segments, library, model=model, use_llm=use_llm,
            cadence_buckets=buckets, easy_bias_sec=easy_bias, track_feedback=feedback,
            played_tracks=played, bpm_overrides=bpm_overrides,
            min_total_sec=max_projected_duration(segments_text), avoid_tracks=avoid,
            effort=effort, progress=progress,
        )
    except ValueError as e:
        return {"error": str(e)}, 422

    timeline = []
    for seg_label, group in playlist.groupby("Segment", sort=False):
        target_pace = group["Target Pace"].iloc[0] if "Target Pace" in group.columns else None
        target_bpm = group["Target BPM"].iloc[0]
        timeline.append(
            {
                "segment": seg_label,
                "targetBpm": float(target_bpm) if pd.notna(target_bpm) else None,
                "targetPaceSec": float(target_pace) if pd.notna(target_pace) else None,
                "tracks": [
                    {
                        "uri": row.get("Track URI"),
                        "name": row["Track Name"],
                        "artist": row["Artist Name(s)"],
                        "startsAt": row["Starts At"],
                        "durationSec": float(row["Duration (ms)"] / 1000),
                        "tempo": float(row["Tempo"]),
                        "camelot": row["Camelot"],
                        "energy": float(row["Energy"]),
                    }
                    for _, row in group.iterrows()
                ],
            }
        )

    llm_failures = playlist.attrs.get("llm_failures") or []
    return {
        "trackUris": [u for u in playlist["Track URI"] if isinstance(u, str)],
        "totalSec": float(playlist["Duration (ms)"].sum() / 1000),
        "timeline": timeline,
        # Segments where the model call failed (rate limit, quota, network)
        # and fell back to the deterministic distance-chain — the Running
        # app surfaces this as a warning instead of a silently lesser mix.
        "llmFailures": llm_failures,
    }, 200


@app.post("/mix")
def mix():
    payload, status = _build_mix_payload(request.get_json(force=True))
    return jsonify(payload), status


@app.post("/mix/stream")
def mix_stream():
    """SSE variant of /mix: streams per-segment progress while the mix builds.

    Events: {"type": "progress", "current", "total", "segment"} as each
    segment starts, then {"type": "done", ...mix} or {"type": "error", "error"}.
    The build runs in a worker thread; the generator drains its queue.
    """
    body = request.get_json(force=True)
    q: "queue.Queue[dict]" = queue.Queue()

    def worker():
        try:
            payload, status = _build_mix_payload(
                body,
                progress=lambda done, total, label: q.put(
                    {"type": "progress", "current": done, "total": total, "segment": label}
                ),
            )
            if status == 200:
                q.put({"type": "done", **payload})
            else:
                q.put({"type": "error", "error": payload.get("error", f"mix failed ({status})")})
        except Exception as e:  # never leave the stream hanging
            q.put({"type": "error", "error": str(e)})

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        # Padding comment flushes past client-side SSE buffers
        yield ": " + "x" * 1024 + "\n\n"
        while True:
            try:
                msg = q.get(timeout=15)
            except queue.Empty:
                # Heartbeat: a segment's LLM pick can run minutes with no
                # progress event — keep bytes flowing so proxies between the
                # Pi and this service never idle the connection out.
                yield ": hb\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                return

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def main():
    parser = argparse.ArgumentParser(description="AI DJ workout-mix service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-llm", action="store_true", help="Never call Ollama; distance-chain only.")
    args = parser.parse_args()

    app.config["USE_LLM"] = not args.no_llm
    app.config["MODEL"] = args.model
    print(f"AI DJ service on http://{args.host}:{args.port}  (LLM: {not args.no_llm})", file=sys.stderr)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
