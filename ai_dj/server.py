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
      "useLlm": true                   // optional
    }
->  {
      "trackUris": ["spotify:track:...", ...],
      "totalSec": 2457,
      "timeline": [{"segment", "targetBpm", "tracks": [{uri, name, artist,
                    startsAt, tempo, camelot, energy}]}]
    }

GET /health -> {"ok": true, "llm": true|false}
"""

import argparse
import io
import sys

import pandas as pd
from flask import Flask, jsonify, request

from bpm_matcher.camelot import to_camelot

from .llm import DEFAULT_MODEL
from .workout import DEFAULT_EASY_PACE, build_workout_playlist, parse_workout

app = Flask(__name__)
app.config["USE_LLM"] = True
app.config["MODEL"] = DEFAULT_MODEL


def _load_library(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    df = df.dropna(subset=["Tempo"]).reset_index(drop=True)
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
    return jsonify({"ok": True, "llm": llm_ok})


@app.post("/mix")
def mix():
    body = request.get_json(force=True)
    segments_text = body.get("segments") or []
    csv_text = body.get("csv") or ""
    if not segments_text or not csv_text:
        return jsonify({"error": "segments and csv are required"}), 400

    try:
        library = _load_library(csv_text)
    except Exception as e:
        return jsonify({"error": f"Could not parse library CSV: {e}"}), 400

    segments = parse_workout(segments_text, easy_pace_sec=_parse_pace(body.get("easyPace")))
    if not segments:
        return jsonify({"error": "No runnable segments recognized in the workout"}), 400

    use_llm = app.config["USE_LLM"] and body.get("useLlm", True)
    try:
        playlist = build_workout_playlist(
            segments, library, model=app.config["MODEL"], use_llm=use_llm
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    timeline = []
    for seg_label, group in playlist.groupby("Segment", sort=False):
        timeline.append(
            {
                "segment": seg_label,
                "targetBpm": float(group["Target BPM"].iloc[0]),
                "tracks": [
                    {
                        "uri": row.get("Track URI"),
                        "name": row["Track Name"],
                        "artist": row["Artist Name(s)"],
                        "startsAt": row["Starts At"],
                        "tempo": float(row["Tempo"]),
                        "camelot": row["Camelot"],
                        "energy": float(row["Energy"]),
                    }
                    for _, row in group.iterrows()
                ],
            }
        )

    return jsonify(
        {
            "trackUris": [u for u in playlist["Track URI"] if isinstance(u, str)],
            "totalSec": float(playlist["Duration (ms)"].sum() / 1000),
            "timeline": timeline,
        }
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
