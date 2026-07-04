"""Local web GUI for the AI DJ: a landing page that fronts the prompt->M3U pipeline.

Run it on the machine that has Ollama and the music library:

    python -m ai_dj.webapp [--port 8766]

Then open http://localhost:8766 - set your request, mix size (tracks or
minutes), where tracks come from (Exportify CSV + Mixxx DB / music folder)
and where the .m3u8 should be written.
"""

import argparse
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from bpm_matcher.features import load_playlist

from .llm import DEFAULT_MODEL, OLLAMA_URL, OllamaError
from .playlist import write_m3u
from .resolve import DEFAULT_MIXXXDB, resolve_locations
from .selector import build_setlist

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
DEFAULT_CSV = "E:/Code/AI_BPM/data/Running.csv"
DEFAULT_OUT_DIR = str(Path.home() / "Music" / "AI_DJ")

app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/defaults")
def defaults():
    llm_ok = False
    try:
        import requests as rq

        llm_ok = rq.get(f"{OLLAMA_URL}/api/tags", timeout=3).ok
    except Exception:
        pass
    return jsonify(
        {
            "csv": DEFAULT_CSV if os.path.isfile(DEFAULT_CSV) else "",
            "mixxxdb": DEFAULT_MIXXXDB if os.path.isfile(DEFAULT_MIXXXDB) else "",
            "outDir": DEFAULT_OUT_DIR,
            "model": DEFAULT_MODEL,
            "llm": llm_ok,
        }
    )


@app.get("/api/browse")
def browse():
    """List a directory for the path-picker modal.

    ?path=<dir> (empty = drive list) and ?ext=.csv,.sqlite to filter files
    (ext omitted = folders only).
    """
    path = (request.args.get("path") or "").strip()
    exts = tuple(e.lower() for e in request.args.get("ext", "").split(",") if e)

    if not path:
        drives = [d for d in os.listdrives()] if hasattr(os, "listdrives") else [
            f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{c}:\\")
        ]
        return jsonify({"path": "", "parent": None, "dirs": drives, "files": []})

    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return jsonify({"error": f"Not a folder: {path}"}), 400

    dirs, files = [], []
    try:
        for entry in sorted(os.scandir(path), key=lambda e: e.name.lower()):
            if entry.name.startswith((".", "$")):
                continue
            if entry.is_dir():
                dirs.append(entry.name)
            elif exts and entry.name.lower().endswith(exts):
                files.append(entry.name)
    except OSError as e:
        return jsonify({"error": str(e)}), 400

    parent = os.path.dirname(path)
    if parent == path:  # drive root - "up" goes to the drive list
        parent = ""
    return jsonify({"path": path, "parent": parent, "dirs": dirs, "files": files})


def _estimate_count(library, minutes: float) -> int:
    """How many library-average tracks fit in `minutes`."""
    mean_ms = float(library["Duration (ms)"].mean()) if "Duration (ms)" in library else 210_000.0
    return max(1, round(minutes * 60_000 / mean_ms))


@app.post("/api/generate")
def generate():
    body = request.get_json(force=True)
    prompt = (body.get("prompt") or "").strip()
    csv_path = (body.get("csv") or "").strip()
    if not prompt:
        return jsonify({"error": "Tell the DJ what you want first."}), 400
    if not csv_path or not os.path.isfile(csv_path):
        return jsonify({"error": f"Library CSV not found: {csv_path or '(empty)'}"}), 400

    try:
        library = load_playlist(csv_path)
    except Exception as e:
        return jsonify({"error": f"Could not load library CSV: {e}"}), 400

    n = body.get("tracks")
    minutes = body.get("minutes")
    if isinstance(minutes, (int, float)) and minutes > 0:
        n = _estimate_count(library, float(minutes))
        prompt = f"{prompt} The mix should run about {minutes:g} minutes."
    n = int(n) if isinstance(n, (int, float)) and n > 0 else None

    music_dir = (body.get("musicDir") or "").strip() or None
    mixxxdb = (body.get("mixxxdb") or "").strip() or DEFAULT_MIXXXDB
    order = body.get("order") or "none"

    try:
        setlist, reasoning = build_setlist(
            prompt,
            library,
            n=n,
            model=(body.get("model") or "").strip() or None,
            smooth=order == "smooth",
            arc=order == "arc",
        )
    except OllamaError as e:
        return jsonify({"error": str(e)}), 502
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    setlist = resolve_locations(setlist, music_dir=music_dir, mixxxdb=mixxxdb)

    wrote = None
    out_dir = (body.get("outDir") or "").strip()
    filename = (body.get("filename") or "").strip()
    if out_dir and filename:
        if not filename.lower().endswith((".m3u", ".m3u8")):
            filename += ".m3u8"
        out_path = os.path.join(out_dir, filename)
        try:
            os.makedirs(out_dir, exist_ok=True)
            resolved, missing = write_m3u(setlist, out_path)
        except OSError as e:
            return jsonify({"error": f"Could not write playlist: {e}"}), 400
        wrote = {"path": out_path, "resolved": resolved, "missing": missing}

    total_ms = float(setlist["Duration (ms)"].sum()) if "Duration (ms)" in setlist else 0.0
    tracks = [
        {
            "name": row["Track Name"],
            "artist": row["Artist Name(s)"],
            "bpm": float(row["Tempo"]),
            "camelot": row["Camelot"] or "?",
            "energy": float(row["Energy"]),
            "durationMs": float(row["Duration (ms)"]) if "Duration (ms)" in row else None,
            "location": row.get("Location"),
        }
        for _, row in setlist.iterrows()
    ]
    return jsonify(
        {"tracks": tracks, "reasoning": reasoning, "totalMin": total_ms / 60_000, "wrote": wrote}
    )


def main():
    parser = argparse.ArgumentParser(description="AI DJ web GUI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    print(f"AI DJ web GUI on http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
