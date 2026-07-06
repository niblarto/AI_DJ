"""Runna workout -> pace-matched, duration-fitted playlist.

Parses segment lines as they appear in the Running app's Runna cards
("1.5mi at 8:35/mi", "1mi warm up at a conversational pace (no faster than
9:15/mi)"), converts each pace to a music BPM via the runner's cadence
(steps/min), then fills each segment's duration with BPM-matched tracks.

Cadence model: pace -> SPM from the Garmin activity_records buckets when a
garmin_activities.db is available (same query as the Running app's pace-spm
endpoint), else a linear fit to the observed data (172 spm @ 9:15/mi,
173 @ 8:35, 174 @ 7:55 - cadence barely moves with pace; stride does).

Music keeps playing through short walking rests (<2 min), so those merge
into the preceding segment. Section changes land on track boundaries: each
segment is filled until its time budget is met and the overshoot is carried
into the next segment's budget.
"""

import math
import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

from bpm_matcher.match import bpm_filter, cross_distance_matrix

from .selector import _log, choose_setlist

BPM_TOLERANCES = (3.0, 5.0, 8.0)
DEFAULT_EASY_PACE = 555  # 9:15/mi - conversational, per the Runna plan
# Fallback pad when the workout card has no projected-duration range: a
# little extra music so the playlist doesn't run out during pauses.
PLAYLIST_PAD_SEC = 300

# "1h50m - 2h10m" or "35m - 45m" in the card's summary line
_DUR_RANGE_RE = re.compile(r"(?:(\d+)\s*h\s*)?(\d+)\s*m(?:in)?s?\b", re.IGNORECASE)


def max_projected_duration(lines: list[str]) -> float | None:
    """Upper bound of the workout's projected duration, from the card's
    summary line ("Long Run • 13.1mi • 1h50m - 2h10m" -> 2h10m). This is the
    slowest projection, so a playlist that long can't run out mid-run even on
    a bad day. None when no summary line carries a duration."""
    best = None
    for line in lines:
        if "•" not in line:
            continue
        for m in _DUR_RANGE_RE.finditer(line):
            sec = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60
            if sec > 0:
                best = max(best or 0, sec)
    return best

# Energy envelope per segment kind (relaxed in steps when the pool runs dry).
ENERGY_BOUNDS = {
    "warmup": (0.45, 0.85),
    "work": (0.60, 1.00),
    "easy": (0.00, 0.55),
    "cooldown": (0.00, 0.60),
    "rest": (0.00, 0.50),
    # Strength sessions: no cadence to match (pace_sec is None so BPM is
    # unfiltered) — just up-tempo, high-energy motivation.
    "strength": (0.70, 1.00),
}

# Easy-effort kinds: get the calmer end of the energy envelope and the
# easy-pace bias. NOTE: BPM matching still outranks energy for these — the
# runner locks cadence to the music, so tempo stays tight for every kind and
# the energy window pads open instead when the pool runs dry.
CHILL_KINDS = {"easy", "cooldown", "rest"}

def _effective_run_tempo(tempo: float) -> float:
    # Below ~95 BPM a runner locks onto double-time; above it, the raw tempo.
    return tempo * 2 if tempo < 95 else tempo


def _kind_bpm_bounds(kind: str, overrides: dict | None) -> tuple[float | None, float | None]:
    """(min, max) BPM for a run type, from the Settings-page overrides.
    No override = no hard limits (automatic cadence matching only). Bounds
    compare against _effective_run_tempo, so half-time tracks count doubled."""
    o = (overrides or {}).get(kind)
    if isinstance(o, dict):
        try:
            lo = float(o["min"]) if o.get("min") else None
            hi = float(o["max"]) if o.get("max") else None
        except (TypeError, ValueError):
            lo = hi = None
        if lo is not None or hi is not None:
            return lo, hi
    return None, None

# Thumbs up/down from the app applies to paces within this window (sec/mi):
# a downvoted track is excluded from segments near that pace, an upvoted one
# is pulled to the front of the segment's setlist.
FEEDBACK_PACE_TOLERANCE = 10.0


@dataclass
class Segment:
    label: str
    kind: str  # warmup | work | easy | cooldown | rest
    duration_sec: float
    pace_sec: float | None  # seconds per mile
    bpm: float | None = None


# ── Parsing ──────────────────────────────────────────────────────────────────

_PACE_RE = re.compile(r"(\d+):(\d+)\s*/mi")
_DIST_RE = re.compile(r"([\d.]+)\s*mi\b")
_REST_RE = re.compile(r"(\d+)\s*(s|sec|secs|min|mins?)\b[^,]*\b(?:rest|walk)", re.IGNORECASE)


def _segment_kind(text: str) -> str:
    t = text.lower()
    if "strength" in t:
        return "strength"
    if "warm up" in t or "warmup" in t:
        return "warmup"
    if "cool down" in t or "cooldown" in t:
        return "cooldown"
    if "conversational" in t or "easy" in t or "recovery" in t:
        return "easy"
    return "work"


def _is_segment_line(line: str) -> bool:
    """True for actual workout steps; false for card header lines like
    "Tempo • 4.5mi • 35m - 45m" (distance/duration summary, no instruction)."""
    if "•" in line:
        return False
    t = line.lower()
    return bool(
        _PACE_RE.search(line)
        or _REST_RE.search(line)
        or " at " in t
        or "warm up" in t or "warmup" in t
        or "cool down" in t or "cooldown" in t
        or "conversational" in t
        or "strength" in t
    )


def parse_workout(lines: list[str], easy_pace_sec: float = DEFAULT_EASY_PACE) -> list[Segment]:
    """Parse Runna segment lines into Segments (rest parts split out)."""
    segments: list[Segment] = []
    for raw in lines:
        line = raw.strip()
        if not line or not _is_segment_line(line):
            continue

        rest_m = _REST_RE.search(line)
        run_part = line[: rest_m.start()].rstrip(", ") if rest_m else line

        pace_m = _PACE_RE.search(run_part)
        dist_m = _DIST_RE.search(run_part)
        pace = int(pace_m.group(1)) * 60 + int(pace_m.group(2)) if pace_m else easy_pace_sec
        if dist_m:
            duration = float(dist_m.group(1)) * pace
        else:
            dur_m = re.search(r"(\d+)\s*min", run_part)
            duration = int(dur_m.group(1)) * 60 if dur_m else 0

        if duration > 0:
            kind = _segment_kind(run_part)
            # A step with no stated pace runs at the conversational default —
            # treat it as easy effort, not a hard "work" interval.
            if kind == "work" and not pace_m:
                kind = "easy"
            # Strength has no pace: pace_sec None leaves seg.bpm unset, so
            # tracks match on energy alone (any BPM).
            segments.append(Segment(run_part, kind, duration, None if kind == "strength" else pace))

        if rest_m:
            value = int(rest_m.group(1))
            rest_sec = value * 60 if rest_m.group(2).startswith("min") else value
            # Give standalone rests (>=120s, so not folded into the prior
            # segment below) an easy/recovery pace so they still get
            # BPM-matched tracks instead of leaving seg.bpm unset.
            segments.append(Segment(rest_m.group(0), "rest", rest_sec, easy_pace_sec))

    # Music plays through short rests - fold them into the previous segment.
    merged: list[Segment] = []
    for seg in segments:
        if seg.kind == "rest" and seg.duration_sec < 120 and merged:
            merged[-1].duration_sec += seg.duration_sec
        else:
            merged.append(seg)
    return merged


# ── Pace -> BPM ──────────────────────────────────────────────────────────────

def garmin_cadence_buckets(db_path: str) -> dict[int, float]:
    """5-second pace bucket -> avg SPM, same query the Running app uses."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            """SELECT CAST(3600.0 / r.speed / 5 AS INTEGER) * 5 AS bucket,
                      AVG(r.cadence * 2) AS avg_spm
               FROM activity_records r
               JOIN activities a ON a.activity_id = r.activity_id
               WHERE LOWER(a.sport) LIKE '%running%'
                 AND r.speed > 0.3 AND r.speed IS NOT NULL
                 AND r.cadence IS NOT NULL AND r.cadence > 10
               GROUP BY bucket HAVING bucket BETWEEN 390 AND 600
               ORDER BY bucket"""
        ).fetchall()
    return {int(b): float(s) for b, s in rows}


def pace_to_bpm(pace_sec: float, cadence_buckets: dict[int, float] | None = None) -> float:
    if cadence_buckets:
        bucket = min(cadence_buckets, key=lambda b: abs(b - pace_sec))
        if abs(bucket - pace_sec) <= 15:
            return round(cadence_buckets[bucket])
    # Fit to measured FIT-file cadence: 171 spm @ 9:15/mi, ~1 spm per 30 s/mi
    # (cadence barely moves with pace; stride length does the work).
    return round(min(max(171 + (555 - pace_sec) / 30, 164.0), 180.0))


# ── Selection ────────────────────────────────────────────────────────────────

# One track per artist across the whole mix: compare on the first credited
# artist so "Foo", "Foo, Bar" and "Foo feat. Baz" all count as the same artist.
def _primary_artist(name) -> str:
    return re.split(r",|;|\bfeat\.?\b|\bft\.?\b|\bfeaturing\b|&", str(name), 1, flags=re.IGNORECASE)[0].strip().lower()


def _bpm_distance(tempo: float, bpm: float) -> float:
    """Distance to target allowing half/double-time matches, mirroring bpm_filter."""
    return min(abs(tempo - bpm), abs(tempo * 2 - bpm), abs(tempo / 2 - bpm))


def _segment_pool(
    library: pd.DataFrame, seg: Segment, used: set, min_pool: int, budget_sec: float,
    easy_bias_sec: float = 0.0, used_artists: set | None = None, played: set | None = None,
    bpm_bounds: tuple[float | None, float | None] = (None, None),
) -> pd.DataFrame:
    lo, hi = bpm_bounds
    if lo is not None or hi is not None:
        eff = library["Tempo"].map(lambda t: _effective_run_tempo(float(t)))
        mask = pd.Series(True, index=library.index)
        if lo is not None:
            mask &= eff >= lo
        if hi is not None:
            mask &= eff <= hi
        library = library[mask]
    e_lo, e_hi = ENERGY_BOUNDS[seg.kind]
    # Runner has been going faster than target on easy runs — calm the music
    # down further: ~0.05 off the energy ceiling per 10 s/mi of overshoot.
    if seg.kind in CHILL_KINDS and easy_bias_sec > 0:
        e_hi = max(0.35, e_hi - min(0.15, easy_bias_sec * 0.005))
    # BPM outranks energy for every kind: hold the tolerance tight and pad the
    # energy window open instead. Use the Run BPM limits in Settings to steer
    # tempo per run type (e.g. easy max 168).
    attempts = [(tol, pad) for tol in BPM_TOLERANCES for pad in (0.0, 0.1, 0.2, 0.45, 1.0)]
    # Last resorts: with one track per artist, a long segment can exhaust the
    # unique artists near the target BPM — better off-tempo music at the back
    # of the pool than silence mid-run.
    attempts += [(12.0, 1.0), (None, 1.0)]
    for tol, pad in attempts:
        pool = bpm_filter(library, seg.bpm, tolerance=tol) if seg.bpm and tol else library
        pool = pool[
            (pool["Energy"] >= max(e_lo - pad, 0))
            & (pool["Energy"] <= min(e_hi + pad, 1))
        ]
        pool = pool[~pool["Track URI"].isin(used)] if "Track URI" in pool.columns else pool
        if used_artists:
            pool = pool[~pool["Artist Name(s)"].map(_primary_artist).isin(used_artists)]
        # One track per artist, applied BEFORE the budget check below so the
        # relaxation loop keeps widening until enough unique-artist music
        # exists to fill the whole segment. Keep each artist's closest-to-BPM
        # track so the dedupe costs as little tempo accuracy as possible —
        # with already-played-at-this-pace tracks sorting behind unplayed
        # ones, so an artist's fresh track wins over their played one.
        if seg.bpm:
            dist = pool["Tempo"].map(lambda t: _bpm_distance(float(t), seg.bpm))
        else:
            # No tempo target (strength): keep each artist's highest-energy track
            dist = -pool["Energy"].astype(float)
        if played:
            dist = dist + pool["Track URI"].isin(played) * 1000.0
        pool = pool.loc[dist.sort_values(kind="stable").index]
        pool = pool.loc[~pool["Artist Name(s)"].map(_primary_artist).duplicated()]
        # The pool must be able to fill the whole segment — a 2h long run
        # needs far more than min_pool tracks.
        if len(pool) >= min_pool and pool["Duration (ms)"].sum() / 1000 >= budget_sec:
            return pool.reset_index(drop=True)
    return pool.reset_index(drop=True)


def _chain_order(pool: pd.DataFrame, anchor: pd.DataFrame | None) -> list[int]:
    """Pool indices ordered by greedy nearest-neighbour feel, starting from
    the track closest to the anchor (previous segment's last track)."""
    dist = cross_distance_matrix(pool, pool)
    if anchor is not None:
        start = int(cross_distance_matrix(pool, anchor).min(axis=1).argmin())
    else:
        start = 0
    order = [start]
    remaining = set(range(len(pool))) - {start}
    while remaining:
        nxt = min(remaining, key=lambda j: dist[order[-1], j])
        remaining.remove(nxt)
        order.append(nxt)
    return order


def _fit_duration(ordered: pd.DataFrame, budget_sec: float, overshoot: bool = False) -> pd.DataFrame:
    """Keep tracks in order until the budget is met, landing the section end
    as close to the budget as possible: the track that crosses the boundary
    is only kept when overshooting beats stopping short. With overshoot=True
    the crossing track is always kept, so the section never ends early —
    used for the final segment, whose budget is a minimum (workout + pad)."""
    picked, cum = [], 0.0
    for i, row in ordered.iterrows():
        dur = row["Duration (ms)"] / 1000
        if cum + dur >= budget_sec:
            if overshoot or not picked or (cum + dur - budget_sec) < (budget_sec - cum):
                picked.append(i)
            break
        picked.append(i)
        cum += dur
    return ordered.loc[picked]


def build_workout_playlist(
    segments: list[Segment],
    library: pd.DataFrame,
    model: str,
    use_llm: bool = True,
    cadence_buckets: dict[int, float] | None = None,
    easy_bias_sec: float = 0.0,
    track_feedback: list[dict] | None = None,
    played_tracks: list[dict] | None = None,
    bpm_overrides: dict | None = None,
    min_total_sec: float | None = None,
    progress=None,
) -> pd.DataFrame:
    """Fill every segment with BPM-matched tracks; returns rows with a
    Segment label and cumulative timing columns.

    track_feedback: [{"uri", "paceSec", "vote": "up"|"down"}] — downvoted
    tracks are dropped from segments whose pace is within
    FEEDBACK_PACE_TOLERANCE of the vote's pace; upvoted ones are moved to
    the front of that segment's setlist so they play more often.

    easy_bias_sec > 0 means recent easy runs came out that much faster than
    target (sec/mi): easy-type segments are then built as if their pace were
    that much slower (lower SPM) with a lower energy ceiling. Easy pace is a
    ceiling ("no faster than"), so the bias only ever slows the music down.

    progress: optional callable(done, total, segment_label) invoked as each
    segment starts building — lets callers stream a live progress bar (the
    per-segment LLM call is the slow part).
    """
    easy_bias_sec = min(max(easy_bias_sec, 0.0), 30.0)
    for seg in segments:
        if seg.pace_sec:
            pace = seg.pace_sec
            if seg.kind in CHILL_KINDS:
                pace += easy_bias_sec
            seg.bpm = pace_to_bpm(pace, cadence_buckets)
            # Clamp the target into the run type's bounds so BPM matching
            # aims inside the window instead of fighting the hard filter.
            lo, hi = _kind_bpm_bounds(seg.kind, bpm_overrides)
            if seg.bpm and hi is not None:
                seg.bpm = min(seg.bpm, hi)
            if seg.bpm and lo is not None:
                seg.bpm = max(seg.bpm, lo)

    # Cover the workout's slowest projected duration rather than appending
    # arbitrary padding tracks: stretch the final segment's budget only by
    # whatever the segment targets fall short of it. Without a projection,
    # fall back to the old fixed pad.
    total = sum(s.duration_sec for s in segments)
    if min_total_sec and min_total_sec > total:
        segments[-1].duration_sec += min_total_sec - total
    elif not min_total_sec:
        segments[-1].duration_sec += PLAYLIST_PAD_SEC

    parts: list[pd.DataFrame] = []
    used: set = set()
    used_artists: set = set()
    prev_tail: pd.DataFrame | None = None
    carry = 0.0

    def _feedback_uris(pace_sec, vote):
        if not track_feedback or not pace_sec:
            return set()
        return {
            f.get("uri") for f in track_feedback
            if f.get("vote") == vote and f.get("uri")
            and abs(float(f.get("paceSec") or 0) - pace_sec) <= FEEDBACK_PACE_TOLERANCE
        }

    # Tracks already played in a past run at (roughly) this pace: unvoted ones
    # rank below unplayed tracks, so mixes stay fresh unless the pool runs dry.
    def _played_uris(pace_sec):
        if not played_tracks or not pace_sec:
            return set()
        return {
            p.get("uri") for p in played_tracks
            if p.get("uri") and p.get("paceSec") is not None
            and abs(float(p["paceSec"]) - pace_sec) <= FEEDBACK_PACE_TOLERANCE
        }

    for seg_idx, seg in enumerate(segments):
        if progress:
            try:
                progress(seg_idx, len(segments), seg.label)
            except Exception:
                pass
        is_last = seg is segments[-1]
        budget = seg.duration_sec - carry
        # Previous overshoot already covers this segment (but the final
        # segment's budget is a hard minimum — only skip it when fully covered).
        if budget <= (0 if is_last else 30):
            carry = -budget
            continue

        downvoted = _feedback_uris(seg.pace_sec, "down")
        played = _played_uris(seg.pace_sec)
        lib_for_seg = library[~library["Track URI"].isin(downvoted)] if downvoted else library
        pool = _segment_pool(
            lib_for_seg, seg, used, min_pool=8, budget_sec=budget, easy_bias_sec=easy_bias_sec,
            used_artists=used_artists, played=played,
            bpm_bounds=_kind_bpm_bounds(seg.kind, bpm_overrides),
        )
        if pool.empty:
            _log(f"No tracks fit segment '{seg.label}' - skipping.")
            continue

        median_sec = pool["Duration (ms)"].median() / 1000
        n_est = min(math.ceil(budget / median_sec) + 1, len(pool))

        ordered = None
        if use_llm:
            cadence_line = f"Cadence target {seg.bpm:.0f} steps/min. " if seg.bpm else ""
            prompt = (
                f"Section of a run workout: {seg.label}. "
                f"{cadence_line}"
                + {
                    "warmup": "Easing in - upbeat but not full throttle.",
                    "work": "Hard effort - driving, motivating, relentless.",
                    "easy": "Conversational effort - chilled, laid-back, mellow; nothing aggressive or high-energy.",
                    "cooldown": "Winding down - relaxed and light.",
                    "rest": "Recovery - calm.",
                    "strength": "Strength training - up-tempo, high-energy, powerful and motivating; any BPM.",
                }[seg.kind]
            )
            try:
                ordered, _ = choose_setlist(prompt, pool, n_est, model)
            except Exception as e:
                _log(f"LLM selection failed for '{seg.label}' ({e}); using distance chain.")
        if ordered is None or ordered.empty:
            ordered = pool.iloc[_chain_order(pool, prev_tail)]

        # Top up from the rest of the pool if the picks don't cover the budget.
        # Compare by Track URI, not index — choose_setlist resets its result's
        # index, so index-based exclusion would re-add already-picked tracks
        # (duplicate index labels then make _fit_duration's .loc explode each
        # pick into multiple rows).
        if ordered["Duration (ms)"].sum() / 1000 < budget:
            leftover = pool[~pool["Track URI"].isin(ordered["Track URI"])].reset_index(drop=True)
            if not leftover.empty:
                extra = leftover.iloc[_chain_order(leftover, ordered.tail(1))]
                ordered = pd.concat([ordered, extra], ignore_index=True)

        # Played-but-unvoted tracks drop to the back of the ordering, so they
        # only make the cut when the unplayed pool can't fill the budget.
        boosted = _feedback_uris(seg.pace_sec, "up")
        demoted = played - boosted
        if demoted:
            is_played = ordered["Track URI"].isin(demoted)
            if is_played.any():
                ordered = pd.concat([ordered[~is_played], ordered[is_played]])

        # Upvoted-at-this-pace tracks lead the segment so they make the cut.
        if boosted:
            is_boost = ordered["Track URI"].isin(boosted)
            if is_boost.any():
                ordered = pd.concat([ordered[is_boost], ordered[~is_boost]])

        # One track per artist: the pool already excludes artists picked in
        # earlier segments, but the ordering itself can still carry several
        # tracks by one artist — keep only the first of each.
        ordered = ordered[~ordered["Artist Name(s)"].map(_primary_artist).duplicated()]

        chosen = _fit_duration(ordered, budget, overshoot=is_last).copy()
        chosen["Segment"] = seg.label
        chosen["Target BPM"] = seg.bpm
        chosen["Target Pace"] = seg.pace_sec  # sec/mi, for post-run pace review

        actual = chosen["Duration (ms)"].sum() / 1000
        carry = actual - budget
        used.update(chosen.get("Track URI", pd.Series(dtype=str)))
        used_artists.update(chosen["Artist Name(s)"].map(_primary_artist))
        prev_tail = chosen.tail(1)
        parts.append(chosen)
        _log(
            f"'{seg.label}': {len(chosen)} tracks, {actual/60:.1f} min "
            f"(target {seg.duration_sec/60:.1f}, carry {carry:+.0f}s)"
        )

    if not parts:
        raise ValueError("No segments could be filled from this library.")
    playlist = pd.concat(parts).reset_index(drop=True)
    ends = playlist["Duration (ms)"].cumsum() / 1000
    playlist["Starts At"] = (ends - playlist["Duration (ms)"] / 1000).map(_mmss)
    return playlist


def _mmss(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import os
    import sys

    from bpm_matcher.features import load_playlist

    from .llm import DEFAULT_MODEL
    from .playlist import write_m3u
    from .resolve import DEFAULT_MIXXXDB, resolve_locations

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Build a pace-matched playlist for a Runna workout.",
        epilog='Workout is a text file, or the segment lines themselves (newline/";" separated).',
    )
    parser.add_argument("workout", help="Workout file path, or the segment lines as text.")
    parser.add_argument("--csv", required=True, help="Exportify playlist CSV (the track library).")
    parser.add_argument("--out", default=None, help="Write an extended M3U here.")
    parser.add_argument("--out-csv", default=None, help="Write the setlist as an Exportify-style CSV (for the Running app).")
    parser.add_argument("--music-dir", default=None, help="Folder to scan for local audio files.")
    parser.add_argument("--mixxxdb", default=DEFAULT_MIXXXDB, help="Path to mixxxdb.sqlite.")
    parser.add_argument("--garmin-db", default=None, help="garmin_activities.db for exact pace->cadence lookup.")
    parser.add_argument("--easy-pace", default="9:15", help="Pace assumed for 'conversational' segments (default 9:15).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama; pick purely by BPM/key/feel distance.")
    args = parser.parse_args()

    if os.path.isfile(args.workout):
        with open(args.workout, encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        lines = [part for chunk in args.workout.splitlines() for part in chunk.split(";")]

    m, s = args.easy_pace.split(":")
    easy_pace = int(m) * 60 + int(s)

    segments = parse_workout(lines, easy_pace_sec=easy_pace)
    if not segments:
        print("No segments recognized in the workout text.")
        sys.exit(1)

    buckets = garmin_cadence_buckets(args.garmin_db) if args.garmin_db else None
    library = load_playlist(args.csv)
    playlist = build_workout_playlist(
        segments, library, model=args.model, use_llm=not args.no_llm, cadence_buckets=buckets
    )
    playlist = resolve_locations(playlist, music_dir=args.music_dir, mixxxdb=args.mixxxdb)

    total = playlist["Duration (ms)"].sum() / 1000
    print(f"\nWorkout playlist — {len(playlist)} tracks, {_mmss(total)} total:")
    for seg_label, group in playlist.groupby("Segment", sort=False):
        bpm = group["Target BPM"].iloc[0]
        print(f"\n  ▶ {seg_label}  (target {bpm:.0f} BPM)")
        for _, row in group.iterrows():
            print(
                f"    {row['Starts At']:>6}  {row['Track Name']} — {row['Artist Name(s)']} "
                f"({row['Tempo']:.0f} BPM, {row['Camelot'] or '?'}, energy {row['Energy']:.2f})"
            )

    if args.out:
        resolved, missing = write_m3u(playlist, args.out)
        print(f"\nWrote {args.out}: {resolved} playable, {missing} missing local files.")
    if args.out_csv:
        original_cols = [c for c in library.columns if c in playlist.columns and c != "Camelot"]
        playlist[original_cols].to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv} (Exportify-style, loadable by the Running app).")


if __name__ == "__main__":
    main()
