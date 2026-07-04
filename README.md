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
prompt ──▶ Qwen 2.5 7B: extract constraints (BPM range, energy, count)
                        │
                bpm_filter + energy filter ──▶ candidate pool (≤120)
                        │
           Qwen 2.5 7B: pick & order the setlist
                        │
        [--smooth / --arc] reorder via bpm_matcher distance
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

## Service mode (PaceSync integration)

`python -m ai_dj.server [--port 8765] [--no-llm]` exposes the workout mixer
over HTTP for the PaceSync running app:

- `POST /mix` `{title, segments, csv, easyPace?, useLlm?}` →
  `{trackUris, totalSec, timeline}`
- `GET /health` → `{ok, llm}`

The PaceSync side (Settings → 🎧 AI DJ → service URL + enable) adds an
**AI DJ Mix** button to each Runna workout card, which builds the mix and
creates/updates a Spotify playlist named `DD-MM-YY <Workout name>`.

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
- `AI_DJ_MODEL` — default model name
- `AI_BPM_PATH` — bpm_matcher repo location
- `LASTFM_API_KEY` — enables the Last.fm autocorrect step in the library
  scanner ([get a key](https://www.last.fm/api/account/create))

Secrets and machine-local values can live in a git-ignored `.env.local` at
the repo root (`KEY=value` lines, loaded on import; real environment
variables take precedence) — this is also how the Windows services see them.
