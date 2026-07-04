"""Build an AI DJ library CSV from a local music folder.

Walks the folder recursively, reads artist/title from tags (mutagen, with an
"Artist - Title" filename fallback), then resolves each track online:
Deezer search -> ISRC -> ReccoBeats audio features (tempo, key, energy,
danceability, valence) - the same pipeline bpm_matcher uses for suggestions.

Tracks the network can't resolve fall back to the Mixxx-analyzed BPM (or a
BPM tag) with neutral 0.5 feature values; files with no BPM from any source
are skipped, since BPM is the primary matching signal.

The output CSV is Exportify-shaped so `load_playlist` accepts it, plus a
Location column pointing at the local file. Re-scanning is incremental:
files already in the CSV are skipped.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd

from bpm_matcher.enrich import features_for_candidates
from bpm_matcher.sources import Candidate, deezer_search_track

from .resolve import AUDIO_EXTENSIONS, DEFAULT_MIXXXDB

NEUTRAL = 0.5  # feature value when the network has no answer


def _fallback_row(info: dict) -> dict:
    """Library row from local data only (Mixxx/tag BPM, neutral features)."""
    return {
        "Track Name": info["title"],
        "Artist Name(s)": info["artist"],
        "Tempo": info["bpm"],
        "Key": -1,
        "Mode": 0,
        "Energy": NEUTRAL,
        "Danceability": NEUTRAL,
        "Valence": NEUTRAL,
        "Duration (ms)": info["duration_ms"],
        "Location": info["path"],
        "Source": "scan:local",
    }


def _analyze_audio(path: str) -> dict | None:
    """Offline analysis via librosa: BPM, Krumhansl-Schmuckler key, RMS energy.

    Last resort for tracks no online source knows. Loads up to two minutes
    from 30s in (intros mislead both tempo and key estimates). Returns None
    when librosa is unavailable or the file can't be decoded.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None
    try:
        y, sr = librosa.load(path, mono=True, offset=30.0, duration=120.0)
        if y.size < sr * 5:  # very short file - retry from the top
            y, sr = librosa.load(path, mono=True, duration=120.0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        if not tempo:
            return None

        # Key: correlate averaged chroma against the 24 rotated K-S profiles.
        major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        best_key, best_mode, best_r = -1, 0, -2.0
        for pc in range(12):
            rolled = np.roll(chroma, -pc)
            for mode, profile in ((1, major), (0, minor)):
                r = float(np.corrcoef(rolled, profile)[0, 1])
                if r > best_r:
                    best_key, best_mode, best_r = pc, mode, r

        # Energy proxy: mean RMS in dBFS mapped onto ~0-1 (-35 dB quiet ambient,
        # -5 dB slammed club master). Rough, but orders tracks sensibly.
        rms_db = float(np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=y), ref=1.0)))
        energy = min(max((rms_db + 35.0) / 30.0, 0.0), 1.0)
        return {"bpm": tempo, "key": best_key, "mode": best_mode, "energy": round(energy, 3)}
    except Exception:
        return None


def _lastfm_correct(artist: str, title: str) -> tuple[str, str] | None:
    """Canonical (artist, title) via Last.fm track.getInfo autocorrection.

    Local tags are often messy ("feat." suffixes, label prefixes); Last.fm's
    autocorrect maps them to the canonical spelling, which gives Deezer a
    second chance to resolve. Needs LASTFM_API_KEY; returns None without it
    or when Last.fm doesn't know the track either.
    """
    from bpm_matcher.sources import LASTFM_API_URL, _get, lastfm_api_key

    key = lastfm_api_key()
    if not key:
        return None
    try:
        data = _get(
            LASTFM_API_URL,
            params={
                "method": "track.getInfo",
                "artist": artist,
                "track": title,
                "autocorrect": 1,
                "api_key": key,
                "format": "json",
            },
        )
    except Exception:
        return None
    track = data.get("track") or {}
    name = track.get("name")
    corrected_artist = (track.get("artist") or {}).get("name")
    if name and corrected_artist and (name, corrected_artist) != (title, artist):
        return corrected_artist, name
    return None


def _read_tags(path: Path) -> dict:
    """artist/title/duration/bpm from tags, filename fallback for identity."""
    import mutagen

    artist = title = isrc = None
    duration_ms = bpm = None
    try:
        tags = mutagen.File(path, easy=True)
        if tags is not None:
            artist = (tags.get("artist") or [None])[0]
            title = (tags.get("title") or [None])[0]
            isrc = (tags.get("isrc") or [None])[0]
            raw_bpm = (tags.get("bpm") or [None])[0]
            if raw_bpm:
                try:
                    bpm = float(raw_bpm)
                except ValueError:
                    pass
            if tags.info and getattr(tags.info, "length", 0):
                duration_ms = tags.info.length * 1000
    except Exception:
        pass
    if not (artist and title) and " - " in path.stem:
        artist, title = (s.strip() for s in path.stem.split(" - ", 1))
    return {"artist": artist, "title": title, "isrc": isrc, "duration_ms": duration_ms, "bpm": bpm}


def _mixxx_bpm_by_path(mixxxdb: str) -> dict:
    """Lowercased file path -> Mixxx-analyzed BPM."""
    if not mixxxdb or not os.path.isfile(mixxxdb):
        return {}
    with sqlite3.connect(f"file:{mixxxdb}?mode=ro", uri=True) as con:
        rows = con.execute(
            """SELECT t.location, l.bpm FROM library l
               JOIN track_locations t ON l.location = t.id
               WHERE l.bpm > 0 AND t.fs_deleted = 0 AND l.mixxx_deleted = 0"""
        ).fetchall()
    return {os.path.normpath(loc).lower(): float(bpm) for loc, bpm in rows}


def scan_folder(
    music_dir: str,
    out_csv: str,
    mixxxdb: str = DEFAULT_MIXXXDB,
    limit: int | None = None,
    progress=None,
) -> dict:
    """Scan `music_dir` into `out_csv`; returns stats. `progress(done, total, msg)`."""
    progress = progress or (lambda done, total, msg: None)

    existing = None
    known_paths: set[str] = set()
    if os.path.isfile(out_csv):
        existing = pd.read_csv(out_csv)
        if "Location" in existing.columns:
            known_paths = {
                os.path.normpath(str(p)).lower() for p in existing["Location"].dropna()
            }

    files = [
        p for p in sorted(Path(music_dir).rglob("*"))
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
        and os.path.normpath(str(p)).lower() not in known_paths
    ]
    if limit:
        files = files[:limit]
    total = len(files)
    progress(0, total, "reading tags")

    mixxx_bpm = _mixxx_bpm_by_path(mixxxdb)

    candidates: list[Candidate] = []
    by_isrc: dict[str, dict] = {}
    pending: list[dict] = []  # no ISRC anywhere - local data / analysis only
    fallbacks: list[dict] = []
    unresolved = 0
    seen_identity: set[tuple] = set()

    for i, path in enumerate(files, 1):
        meta = _read_tags(path)
        artist, title = meta["artist"], meta["title"]
        progress(i, total, f"resolving {artist or path.stem} - {title or ''}")
        if not (artist and title):
            unresolved += 1
            continue
        identity = (str(artist).lower(), str(title).lower())
        if identity in seen_identity:
            continue
        seen_identity.add(identity)

        # ISRC: embedded tag first, then Deezer search, then a retry with
        # Last.fm's autocorrected spelling (messy local tags miss otherwise).
        isrc = meta["isrc"]
        if not isrc:
            try:
                found = deezer_search_track(artist, title)
                isrc = found.isrc if found else None
            except Exception:
                pass
        if not isrc:
            corrected = _lastfm_correct(artist, title)
            if corrected:
                try:
                    found = deezer_search_track(*corrected)
                    isrc = found.isrc if found else None
                except Exception:
                    pass

        info = {
            "path": str(path),
            "duration_ms": meta["duration_ms"],
            "artist": artist,
            "title": title,
            "bpm": mixxx_bpm.get(os.path.normpath(str(path)).lower()) or meta["bpm"],
        }
        if isrc:
            candidates.append(Candidate(artist=artist, title=title, source="scan", isrc=isrc))
            by_isrc[isrc] = info
        else:
            pending.append(info)

    progress(total, total, "fetching audio features (ReccoBeats)")
    feats = features_for_candidates(candidates) if candidates else pd.DataFrame()

    rows = []
    enriched_isrcs = set(feats["ISRC"]) if not feats.empty else set()
    if not feats.empty:
        for _, row in feats.iterrows():
            info = by_isrc.get(row["ISRC"], {})
            rows.append(
                {
                    "Track Name": row["Track Name"],
                    "Artist Name(s)": row["Artist Name(s)"],
                    "Tempo": row["Tempo"],
                    "Key": row["Key"],
                    "Mode": row["Mode"],
                    "Energy": row["Energy"],
                    "Danceability": row["Danceability"],
                    "Valence": row["Valence"],
                    "Duration (ms)": info.get("duration_ms"),
                    "Spotify URL": row.get("Spotify URL"),
                    "Location": info.get("path"),
                    "Source": "scan:reccobeats",
                }
            )
    # Everything the network couldn't fully resolve: local BPM data if we have
    # it, otherwise offline librosa analysis (BPM + key + energy estimate).
    leftovers = pending + [info for isrc, info in by_isrc.items() if isrc not in enriched_isrcs]
    analyzed = 0
    for j, info in enumerate(leftovers, 1):
        if info["bpm"]:
            fallbacks.append(_fallback_row(info))
            continue
        progress(total, total, f"analyzing audio {j}/{len(leftovers)}: {info['artist']} - {info['title']}")
        analysis = _analyze_audio(info["path"])
        if analysis:
            row = _fallback_row(info)
            row.update(
                {
                    "Tempo": round(analysis["bpm"], 1),
                    "Key": analysis["key"],
                    "Mode": analysis["mode"],
                    "Energy": analysis["energy"],
                    "Source": "scan:analysis",
                }
            )
            fallbacks.append(row)
            analyzed += 1
        else:
            unresolved += 1

    rows.extend(fallbacks)
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
    if not combined.empty:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        combined.to_csv(out_csv, index=False)

    return {
        "scanned": total,
        "added": len(rows),
        "enriched": len(rows) - len(fallbacks),
        "analyzed": analyzed,
        "fallback": len(fallbacks) - analyzed,
        "skippedKnown": len(known_paths),
        "unresolved": unresolved,
        "libraryTotal": int(len(combined)),
        "out": os.path.abspath(out_csv),
    }
