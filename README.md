# AI DJ

Natural-language prompt → ordered DJ setlist → M3U playlist for Mixxx, all local.
Comes with a **web GUI** (see [Web GUI](#web-gui)) and a **library scanner** that
builds a track library straight from a music folder. Also the **workout-mix
companion service for [PaceSync](https://github.com/niblarto/PaceSync)**: it
turns a Runna workout into a cadence-matched Spotify playlist over HTTP
(see [Service mode](#service-mode-pacesync-integration)).

A Windows take on the "local AI music DJ" stack, with two substitutions:
the audio-analysis layer (Essentia, which has no Windows wheels) is replaced by
[bpm_matcher](../AI_BPM) — Spotify-style features from an Exportify CSV, with
Camelot-wheel key handling and half/double-time BPM matching — and the LLM runs
in Ollama on the local GPU.

## Pipeline

```
Exportify CSV or scanned folder ──▶ bpm_matcher (Tempo/Camelot/Energy/Dance/Valence)
                        │
prompt ──▶ LLM (Ollama): extract constraints (BPM range, energy, count)
                        │
                bpm_filter + energy filter ──▶ candidate pool (≤140)
                        │
           LLM (Ollama): pick & order the setlist
                        │
        [--smooth / --arc] reorder via bpm_matcher distance
                        │
     resolve to local files (mixxxdb.sqlite, then --music-dir scan)
                        │
                   playlist.m3u8 ──▶ Mixxx Auto DJ
```

This is the free-text/CLI pipeline. [Workout mode](#workout-mode-runna) (the
PaceSync integration) builds a candidate pool per workout section instead,
with no upper cap — see that section for how it differs.

## Setup

0. **Cloud LLM backends (optional)** — the setlist LLM call can also run
   against the Claude API or the Gemini API instead of a local Ollama model
   (useful for workout mode when the always-on-required Ollama PC isn't
   available; PaceSync's on-Pi bridge calls these directly). Set
   `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`, or save a key via
   `claude_config.save_claude_api_key()` / the equivalent Gemini config, or
   through the service's `POST /settings/claude-key`. `is_claude_model()` /
   `is_gemini_model()` in `ai_dj/llm.py` dispatch by model ID — see
   `CLAUDE_MODELS` / `GEMINI_MODELS` there for the current selectable set.
1. **Ollama** — install from https://ollama.com/download/windows, then:
   ```
   ollama pull qwen3.5:9b
   ```
   Default model is `qwen3.5:9b` (`DEFAULT_MODEL` in `ai_dj/llm.py`, override
   with `AI_DJ_MODEL` or `--model`). `qwen3.5:9b` is a reasoning model — the
   Ollama request sends `think: false` so its output goes straight to
   `message.content` instead of being consumed by hidden reasoning tokens
   (otherwise `format: json` + a bounded `num_predict` can come back with an
   empty `content` on a model that "thinks" before answering). On a 10GB GPU,
   `num_ctx` is capped at 9728 to stay fully on-GPU for larger models
   (measured against `mistral-nemo:12b`: 9728 holds at 100% GPU, 10240
   already spills ~7% to CPU) — the pipeline sets this by default regardless
   of which model you pull. `qwen2.5:14b` and `phi4:14b` were also tried as
   middle-ground options but neither fits on a 10GB card even at num_ctx=8192
   (23-35% spills to CPU); `gemma2:9b` fits fully on-GPU but its native
   context tops out at 8192, below what the candidate-pool prompt needs.
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
| `--model NAME` | Ollama model (default `qwen3.5:9b`, or `AI_DJ_MODEL` env var) |
| `--smooth` | Reorder the picks by weighted BPM/key/feel distance for smoother transitions |
| `--arc` | Reorder as a rising energy arc (mellow opener → peak-energy closer) |

Tracks that can't be matched to a local file are written into the M3U as
`# MISSING:` comments with their Spotify links, so the file is always valid.

## Web GUI

`python -m ai_dj.webapp [--port 8766]` (or the `ai-dj-web` script) serves a
landing page at http://localhost:8766 that fronts the whole pipeline:

- **Your request** — free-text prompt, plus mix size as a track count *or*
  minutes (minutes are converted via the library's average track length) and
  a track-ordering choice (DJ's order / smoothest transitions / rising
  energy arc).
- **Style toggles** — one-click chips appended to the prompt as style
  directives: tempo bands and shapes (Steady-BPM, Ramp-Up, Sprint…), key and
  harmony (Harmonic, Key-Locked, Minor/Major-Mode…), energy (Low/Medium/High,
  Energy-Wave, Peak-Mode…), mood (Dark-Club, Euphoric, Hypnotic…), selection
  (Hidden-Gems, Classics, Danceability…), structure (Story-Arc, Build-Up,
  Breakdown…) and texture (Minimal, Percussive, Vintage…). Numeric toggles
  become hard library filters; the rest steer the model's picks.
- **Track sources** — the library CSV (Exportify export or a scanned
  library), the Mixxx DB and an optional music folder for resolving local
  files, plus the folder scanner below.
- **Output** — where the `.m3u8` goes; leave the filename blank to preview
  the setlist without writing a file.

Every path field has a **Browse…** button backed by a server-side picker
(browsers can't read real filesystem paths). The result table shows BPM,
Camelot key, energy, length and whether each track resolved to a local file.

### Library scanner (music folder → library CSV)

*Track sources → Build a library from a music folder* scans a folder
recursively and writes an Exportify-shaped CSV that plugs straight in as the
library source. Per file: artist/title/duration from tags (mutagen, with an
`Artist - Title` filename fallback), then features from the best source that
answers:

1. **ISRC → ReccoBeats** — embedded ISRC tag, else Deezer search, else a
   retry with Last.fm's autocorrected spelling (`LASTFM_API_KEY`); the ISRC
   fetches full audio features (tempo, key, energy, danceability, valence).
2. **Mixxx / BPM tag** — Mixxx-analyzed BPM or a `TBPM` tag, with neutral
   feature values.
3. **Offline analysis** — librosa estimates BPM, musical key
   (Krumhansl-Schmuckler chroma correlation) and an RMS-loudness energy
   value. Needs `pip install librosa`; expect ~5–15 s per analyzed track.

The CSV keeps each file's path in a `Location` column (no Mixxx needed to
resolve those tracks), and re-scanning is incremental — already-scanned files
are skipped, so re-run after adding music.

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
- **Candidate pool per section**: unlike the free-text CLI/service pipeline
  above (capped at `MAX_CANDIDATES`, 140), workout mode's `_segment_pool()`
  in `ai_dj/workout.py` sends the model *every* track that survives that
  section's BPM/energy filter — there's no upper cap. `min_pool` (currently
  `20`, passed from `build_workout_playlist()`) only controls how far the
  BPM-tolerance/energy-window widens before the pool is accepted, i.e. the
  floor the loop stops at, not a sample size — if 80 tracks pass the filter
  at that tolerance step, all 80 go to the LLM.

## Service mode (PaceSync integration)

`python -m ai_dj.server [--port 8765] [--no-llm] [--model NAME]` exposes the
workout mixer over HTTP for the PaceSync running app:

- `POST /mix` `{title, segments, csv, easyPace?, useLlm?, model?, effort?,
  cadenceBuckets?, trackFeedback?, playedTracks?, bpmOverrides?,
  avoidTracks?}` → `{trackUris, totalSec, timeline, llmFailures}`
- `POST /mix/stream` — same body as `/mix`, but streams Server-Sent Events as
  each segment builds: `{"type":"progress","current","total","segment",
  "detail"?}` (detail carries the LLM interaction status for that segment —
  candidates sent, tracks returned, or a fallback notice), then
  `{"type":"done",...mix}` or `{"type":"error","error"}`. `detail` is
  populated from `build_workout_playlist`'s progress callback in
  `ai_dj/workout.py`.
- `GET /health` → `{ok, llm, claude, claudeModels}`
- `GET /usage` → per-model Claude/Gemini token usage and estimated cost since
  this process started (`ai_dj/llm.py`'s usage-tracking file)
- `GET /llm-log` → the last 50 LLM calls made on this host (prompt, model,
  ok/error, duration) — merged with PaceSync's own on-Pi log on the Settings
  page so Claude/Gemini calls (which run on the Pi) and Ollama calls (which
  run here) show up in one place, tagged by source.
- `POST /settings/claude-key` `{apiKey}` — saves a Claude API key the service
  should use, dropping the cached client so the next request picks it up.

`llmFailures` lists segments where the model call errored (rate limit, quota,
network) and fell back to the deterministic distance-chain, so a degraded
mix is surfaced instead of shipping silently.

The PaceSync side (Settings → 🎧 AI DJ → service URL + enable, or Claude/Gemini
API key + model) adds an **AI DJ Mix** button to each Runna workout card,
which builds the mix and creates/updates a Spotify playlist named
`DD-MM-YY <Workout name>`. Claude and Gemini mixes actually run on the Pi
itself via a small bridge script rather than calling out to this service, so
they don't depend on the service host being on — only Ollama-backed mixes
need this service reachable.

## Windows services

`install-service.ps1` (elevated PowerShell) installs [NSSM](https://nssm.cc)
via winget and registers both servers as auto-starting Windows services with
crash-restart and rotating logs:

| Service | What | Port | Logs |
|---|---|---|---|
| `AIDJService` | PaceSync workout-mix API | 8765 | `service.out/err.log` |
| `AIDJWebService` | Web GUI | 8766 | `service-web.out/err.log` |

The script also opens inbound firewall rules for both ports (the servers bind
`0.0.0.0`, so any device on the LAN can use the GUI — there's no auth, so
don't port-forward it to the internet) and passes the installing user's
profile into the service environment (services run as LocalSystem, whose
`%LOCALAPPDATA%` would otherwise miss your Mixxx DB). Edit the paths at the
top of the script for your machine first. Re-running it is idempotent.

## Streaming the set (Icecast)

`install-icecast.ps1` (elevated) installs [Icecast 2](https://icecast.org)
and registers it as the auto-starting `IcecastService` on port 8000 (firewall
opened), so Mixxx can live-broadcast and any device on the LAN can listen:

- Mixxx: *Preferences → Live Broadcasting* — Type `Icecast 2`, Host
  `localhost`, Port `8000`, Mount `/mixxx`, Login `source`, plus the source
  password you chose at install (`-SourcePassword`; stored in `.env.local`).
- Listen on any device at `http://<pc-ip>:8000/mixxx` (VLC, a browser, or a
  radio app). Status page: `http://<pc-ip>:8000/`.

## Playing the set in Mixxx

Library sidebar → Playlists → right-click → *Import Playlist* → pick the
`.m3u8`, then drag it onto **Auto DJ** and enable it. Mixxx handles beat sync
and crossfading. (Options → Preferences → Auto DJ to tune transition length.)

## Environment variables

- `OLLAMA_URL` — Ollama endpoint (default `http://localhost:11434`)
- `AI_DJ_MODEL` — default model name (default `qwen3.5:9b`)
- `AI_BPM_PATH` — bpm_matcher repo location
- `LASTFM_API_KEY` — enables the Last.fm autocorrect step in the library
  scanner ([get a key](https://www.last.fm/api/account/create))
- `ANTHROPIC_API_KEY` — Claude API key; alternative to saving one via
  `claude_config`/`POST /settings/claude-key`
- `GEMINI_API_KEY` — Gemini API key; alternative to saving one via
  `gemini_config`
- `AI_DJ_USAGE_FILE` — path to the Claude/Gemini token-usage tracking file
  (default: `ai-dj-usage.json` next to the `ai_dj/` package)
- `AI_DJ_LLM_LOG_FILE` — path to the rolling LLM-call log file that backs
  `GET /llm-log` (default: `ai-dj-llm-log.json` next to the `ai_dj/` package)

Secrets and machine-local values can live in a git-ignored `.env.local` at
the repo root (`KEY=value` lines, loaded on import; real environment
variables take precedence) — this is also how the Windows services see them.
