"""AI DJ command line: prompt in, M3U out.

Example:
    python -m ai_dj "45 minutes of progressive house building up energy" \
        --csv E:/Code/AI_BPM/data/Running.csv --out set.m3u8
"""

import argparse
import sys

from bpm_matcher.features import load_playlist

from .llm import DEFAULT_MODEL
from .playlist import write_m3u
from .resolve import DEFAULT_MIXXXDB, resolve_locations
from .selector import build_setlist


def _format_track(row) -> str:
    loc = " [file found]" if isinstance(row.get("Location"), str) else ""
    camelot = row["Camelot"] if isinstance(row["Camelot"], str) else "?"
    return (
        f"{row['Track Name']} — {row['Artist Name(s)']} "
        f"({row['Tempo']:.0f} BPM, {camelot}, energy {row['Energy']:.2f}){loc}"
    )


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Turn a natural-language request into an ordered DJ setlist.")
    parser.add_argument("prompt", help="What kind of set you want, in plain language.")
    parser.add_argument("--csv", required=True, help="Exportify playlist CSV (the track library).")
    parser.add_argument("-n", type=int, default=None, help="Track count (default: from prompt, else 15).")
    parser.add_argument("--out", default=None, help="M3U path to write (omit to just print the setlist).")
    parser.add_argument("--music-dir", default=None, help="Folder to scan for local audio files.")
    parser.add_argument("--mixxxdb", default=DEFAULT_MIXXXDB, help="Path to mixxxdb.sqlite.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--smooth", action="store_true", help="Reorder picks for smoothest transitions.")
    parser.add_argument(
        "--unique-artists",
        action="store_true",
        help="Never pick two tracks by the same artist.",
    )
    parser.add_argument(
        "--arc",
        action="store_true",
        help="Reorder as a rising energy arc (mellow opener to peak-energy closer) instead of plain smoothing.",
    )
    args = parser.parse_args()

    library = load_playlist(args.csv)
    setlist, reasoning = build_setlist(
        args.prompt, library, n=args.n, model=args.model, smooth=args.smooth, arc=args.arc,
        unique_artists=args.unique_artists,
    )
    setlist = resolve_locations(setlist, music_dir=args.music_dir, mixxxdb=args.mixxxdb)

    print(f"\nSetlist ({len(setlist)} tracks):")
    for i, (_, row) in enumerate(setlist.iterrows(), 1):
        print(f"  {i:2d}. {_format_track(row)}")
    if reasoning:
        print(f"\nDJ notes: {reasoning}")

    if args.out:
        resolved, missing = write_m3u(setlist, args.out)
        print(f"\nWrote {args.out}: {resolved} playable, {missing} missing local files.")
        if missing:
            print("Missing tracks are listed as comments in the file (with Spotify links).")


if __name__ == "__main__":
    main()
