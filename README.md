# AI DJ

Natural-language prompt → ordered DJ setlist → M3U playlist for Mixxx, all local.
Also the **workout-mix companion service for [PaceSync](https://github.com/niblarto/PaceSync)**:
it turns a Runna workout into a cadence-matched Spotify playlist over HTTP
(see [Service mode](#service-mode-pacesync-integration)).

A Windows take on the "local AI music DJ" stack, with two substitutions:
the audio-analysis layer (Essentia, which has no Windows wheels) is replaced by
[bpm_matcher](../AI_BPM) — Spotify-style features from an Exportify CSV, with
Camelot-wheel key handling and half/double-time BPM matching — and the LLM runs
in Ollama on the local GPU.

## Pipeline

```
Exportify CSV ──▶ bpm_matcher (Tempo/Camelot/Energy/Dance/Valence)
                        │
prompt ──▶ Qwen 2.5 7B: extract constraints (BPM range, energy, count)
                        │
                bpm_filter + energy filter ──▶ candidate pool (≤120)
                        │
           Qwen 2.5 7B: pick & order the setlist
                        │
        [--smooth] greedy reorder via bpm_matcher distance
                        │
     resolve to local files (mixxxdb.sqlite, then --music-dir scan)
                        │
                   playlist.m3u8 ──▶ Mixxx Auto DJ
```

## Setup

1. **Ollama** — install from https://ollama.com/download/windows, then:
   ```
   ollama pull qwen2.5:7b
   ```
2. **Python deps** — `pip install -r requirements.txt` (or reuse the AI_BPM
   venv, which already has everything except mutagen).
3. **bpm_matcher** — expected at `E:\Code\AI_BPM`; override with the
   `AI_BPM_PATH` environment variable.
4. **Mixxx** (optional but the point) — install from https://mixxx.org, scan
   your music library once so `%LOCALAPPDATA%\Mixxx\mixxxdb.sqlite` exists.

## Usage

```
python -m ai_dj "a 10-track high-energy house set around 125 BPM that builds up and peaks" `
    --csv E:\Code\AI_BPM\data\Running.csv --out set.m3u8 --smooth
```

Options:

| Flag | Meaning |
|---|---|
| `--csv PATH` | Exportify playlist CSV used as the track library (required) |
| `--out PATH` | Write an extended M3U; omit to just print the setlist |
| `-n N` | Force track count (default: inferred from prompt, else 15) |
| `--music-dir DIR` | Also scan this folder for local audio files |
| `--mixxxdb PATH` | Mixxx database (default: `%LOCALAPPDATA%\Mixxx\mixxxdb.sqlite`) |
| `--model NAME` | Ollama model (default `qwen2.5:7b`, or `AI_DJ_MODEL` env var) |
| `--smooth` | Reorder the picks by weighted BPM/key/feel distance for smoother transitions |

Tracks that can't be matched to a local file are written into the M3U as
`# MISSING:` comments with their Spotify links, so the file is always valid.

## Workout mode (Runna)

Turns a Runna workout into a playlist where each section's music matches your
cadence at that pace and lasts as long as the section:

```
python -m ai_dj.workout workouts\steady-into-tempo.txt --csv E:\Code\Running\Running.csv --out-csv mix.csv
```

- Segment lines are parsed as they appear on the Running app's cards
  ("1.5mi at 8:35/mi", "1mi warm up at a conversational pace…"); header lines
  ("Tempo • 4.5mi • 35m-45m") are ignored.
- Pace → BPM via your cadence: a linear fit to the app's observed data
  (172 spm @ 9:15/mi, +1 per ~40s/mi faster), or the exact Garmin bucket
  lookup with `--garmin-db path\to\garmin_activities.db`.
- Half/double-time matching means an 86 BPM track counts for a 172 cadence.
- Short walking rests (<2 min) merge into the previous section; section
  changes land on track boundaries, balanced to stay near the pace-change time.
- `--no-llm` skips Ollama entirely (deterministic BPM/key/feel chaining) —
  runs fine on a Raspberry Pi.

## Service mode (PaceSync integration)

`python -m ai_dj.server [--port 8765] [--no-llm]` exposes the workout mixer
over HTTP for the PaceSync running app:

- `POST /mix` `{title, segments, csv, easyPace?, useLlm?}` →
  `{trackUris, totalSec, timeline}`
- `GET /health` → `{ok, llm}`

The PaceSync side (Settings → 🎧 AI DJ → service URL + enable) adds an
**AI DJ Mix** button to each Runna workout card, which builds the mix and
creates/updates a Spotify playlist named `DD-MM-YY <Workout name>`.
If this PC hosts the service, allow inbound TCP 8765 through Windows Firewall.

To run it as a Windows service (auto-start, restart on failure), run
`install-service.ps1` in an elevated PowerShell — it installs
[NSSM](https://nssm.cc) via winget and registers `AIDJService`. Edit the
paths at the top of the script for your machine first.

## Playing the set in Mixxx

Library sidebar → Playlists → right-click → *Import Playlist* → pick the
`.m3u8`, then drag it onto **Auto DJ** and enable it. Mixxx handles beat sync
and crossfading. (Options → Preferences → Auto DJ to tune transition length.)

## Environment variables

- `OLLAMA_URL` — Ollama endpoint (default `http://localhost:11434`)
- `AI_DJ_MODEL` — default model name
- `AI_BPM_PATH` — bpm_matcher repo location
