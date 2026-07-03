"""Resolve (artist, title) rows to local audio files for the M3U.

Sources, in order of preference:
1. Mixxx's library database (%LOCALAPPDATA%/Mixxx/mixxxdb.sqlite) - already
   tagged and analyzed by the software that will play the set.
2. A music folder scan - tags via mutagen when installed, otherwise
   "Artist - Title" filename convention.

Tracks that resolve nowhere keep Location=None; the playlist writer lists
them as missing (with their Spotify URL when the CSV has one).
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd

from bpm_matcher.sources import normalized_key

AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".aiff"}

DEFAULT_MIXXXDB = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Mixxx", "mixxxdb.sqlite")


def _index_add(index: dict, artist: str, title: str, location: str):
    # Multi-artist strings ("A; B" or "A;B") get indexed under each artist.
    for a in str(artist).split(";"):
        index.setdefault(normalized_key(a, title), location)


def mixxx_index(db_path: str = DEFAULT_MIXXXDB) -> dict:
    """(artist, title) -> file path from Mixxx's library, if the DB exists."""
    if not os.path.isfile(db_path):
        return {}
    index: dict = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            """SELECT l.artist, l.title, t.location
               FROM library l JOIN track_locations t ON l.location = t.id
               WHERE t.fs_deleted = 0 AND l.mixxx_deleted = 0"""
        ).fetchall()
    for artist, title, location in rows:
        if artist and title and location:
            _index_add(index, artist, title, location)
    return index


def folder_index(music_dir: str) -> dict:
    """(artist, title) -> file path from a recursive folder scan."""
    try:
        import mutagen
    except ImportError:
        mutagen = None

    index: dict = {}
    for path in Path(music_dir).rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
            continue
        artist = title = None
        if mutagen is not None:
            try:
                tags = mutagen.File(path, easy=True)
                if tags:
                    artist = (tags.get("artist") or [None])[0]
                    title = (tags.get("title") or [None])[0]
            except Exception:
                pass
        if not (artist and title) and " - " in path.stem:
            artist, title = path.stem.split(" - ", 1)
        if artist and title:
            _index_add(index, artist, title, str(path))
    return index


def resolve_locations(
    setlist: pd.DataFrame,
    music_dir: str | None = None,
    mixxxdb: str = DEFAULT_MIXXXDB,
) -> pd.DataFrame:
    """Return the setlist with a Location column (None where unresolved)."""
    index = mixxx_index(mixxxdb)
    if music_dir:
        for k, v in folder_index(music_dir).items():
            index.setdefault(k, v)

    def lookup(artist, title):
        for a in str(artist).split(";"):
            loc = index.get(normalized_key(a, str(title)))
            if loc:
                return loc
        return None

    setlist = setlist.copy()
    setlist["Location"] = [
        lookup(a, t) for a, t in zip(setlist["Artist Name(s)"], setlist["Track Name"])
    ]
    return setlist
