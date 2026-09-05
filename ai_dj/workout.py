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

from .selector import _log, choose_flow_order, choose_setlist, MAX_CANDIDATES

BPM_TOLERANCES = (3.0, 5.0, 8.0)
DEFAULT_EASY_PACE = 555  # 9:15/mi - conversational, per the Runna plan
# A rest explicitly described as "walking" (vs. a jogging/easy-pace rest) is
# never faster than a 20:00/mi walk, regardless of the configured easy pace.
WALKING_REST_MIN_PACE_SEC = 1200
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

# BPM tolerance widens further for warmup/easy/cooldown/rest before landing
# on a pool - these are paces where a wider tempo spread doesn't hurt (a
# runner isn't locked to the beat as tightly at an easy effort), so a bigger
# pool of decent-fit music beats a small pool of exact-fit music. "work" (the
# hard-effort segments - tempo, intervals, time trials, where pace really
# counts) stays on the tight default tolerance, so the LLM only ever sees
# close-to-target candidates. The Run BPM limits ceiling (Settings) still
# applies on top of either — this only controls how far the *search* widens
# before accepting a pool, not the hard min/max bound.
WIDE_TOLERANCE_KINDS = {"warmup", "easy", "cooldown", "rest"}
BPM_TOLERANCES_WIDE = (3.0, 5.0, 8.0, 12.0)

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


def _kind_bpm_sweet(kind: str, overrides: dict | None) -> float | None:
    """Preferred BPM within a run type's range, from the Settings-page sweet
    spot. Unlike _kind_bpm_bounds, this never excludes a track - it only
    sways which of the still-eligible candidates _segment_pool ranks first.
    A separate function rather than widening _kind_bpm_bounds to a 3-tuple:
    both of that function's call sites unpack exactly 2 values, and bounds
    vs. preference are different enough concepts to keep visibly distinct.
    None = no preference set for this kind (0-is-falsy is deliberate, same
    convention as min/max above - a literal 0 BPM isn't meaningful)."""
    o = (overrides or {}).get(kind)
    if isinstance(o, dict):
        try:
            return float(o["sweet"]) if o.get("sweet") else None
        except (TypeError, ValueError):
            return None
    return None

# Thumbs up/down from the app applies to paces within this window (sec/mi):
# a downvoted track is excluded from segments near that pace, an upvoted one
# is pulled to the front of the segment's setlist.
FEEDBACK_PACE_TOLERANCE = 10.0

# Play-count weight: 0-1000, +100 per confirmed-mix play, capped at 1000.
# This is a hard tier, not a graded nudge - every 0-play track sorts ahead of
# every 1-play track regardless of BPM fit, every 1-play track ahead of every
# 2-play track, and so on, with BPM distance only breaking ties *within* the
# same play-count tier. So the candidate pool (and therefore what MAX_CANDIDATES
# truncates down to before the LLM ever sees it) is drawn from the
# least-played tracks first, only reaching further-played tracks once the
# lower tiers can't fill the segment's duration budget on their own.
PLAY_COUNT_WEIGHT = 100
PLAY_COUNT_WEIGHT_CAP = 1000


@dataclass
class Segment:
    label: str
    kind: str  # warmup | work | easy | cooldown | rest
    duration_sec: float
    pace_sec: float | None  # seconds per mile
    bpm: float | None = None
    # Preferred BPM within this kind's range (Settings' sweet spot) - a
    # separate field from `bpm` (the pace-derived cadence target) since the
    # two mean different things: `bpm` also drives the clamp, the LLM
    # prompt's "Cadence target" line, the Target BPM output column, and
    # _bpm_smooth_order's playback arc - none of which this should touch.
    sweet_bpm: float | None = None
    # Seconds of a short (<120s) rest folded into duration_sec by the merge
    # pass below, plus that rest's own pace_sec - purely informational,
    # unused by BPM matching/track-fill (which correctly just wants a single
    # time budget). Exists so a distance-from-duration consumer (see
    # scripts/parse_workout_segments.py) can split the folded rest back out
    # instead of wrongly converting its time to miles at the work portion's
    # much faster pace.
    folded_rest_sec: float = 0.0
    folded_rest_pace_sec: float | None = None


# ── Parsing ──────────────────────────────────────────────────────────────────

_PACE_RE = re.compile(r"(\d+):(\d+)\s*/mi")
_DIST_RE = re.compile(r"([\d.]+)\s*mi\b")
_REST_RE = re.compile(r"(\d+)\s*(s|sec|secs|min|mins?)\b[^,]*\b(?:rest|walk)", re.IGNORECASE)
# A "no faster than X/mi" (or "or slower") clause is a ceiling on an
# easy/warmup/cooldown step, not its target pace - matching it as the
# target reads a conversational warm up as a fast one. Strip it before
# looking for the real target pace, which (if present at all) always comes
# from the "at X/mi" part earlier in the line.
_PACE_CEILING_RE = re.compile(r"\(?\s*(?:no faster than|or slower)\b.*?/mi\)?", re.IGNORECASE)
# "5 reps of:" / "3 x of:" / "4 sets of:" / "Repeat the following 3x:"
# header introducing a block of lines to repeat N times - the block ends at
# the next line that isn't part of it (blank, or itself another header/
# segment at the top level). Runna renders the block's lines with no special
# indentation of their own, so the only signal is this header line and the
# segment lines that follow until the next non-continuation line.
_REPS_RE = re.compile(
    r"^\s*(?:(\d+)\s*(?:reps?|sets?|x)\s+of|repeat\s+the\s+following\s+(\d+)\s*x)\s*:?\s*$",
    re.IGNORECASE,
)
# A multi-line repeat body is fenced by a "----------"-style dashed rule
# above and below it (Runna's "Repeat the following 3x:" phrasing, as
# opposed to the older single-line "N reps of:" body) - 3+ dashes, nothing
# else on the line.
_DASH_RULE_RE = re.compile(r"^-{3,}$")


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
    "Tempo • 4.5mi • 35m - 45m" (distance/duration summary, no instruction).

    "•" is overloaded in Runna's text: the card header uses it as a
    "label • label • label" separator between several short fields, while a
    reps-block body line uses it only as a single LEADING list marker
    ("• 0.62mi at 7:25/mi ..."). Reject the former (2+ bullets, or a bullet
    with no real instruction after it) but not the latter.
    """
    if line.count("•") >= 2:
        return False
    stripped = line.lstrip("•").strip()
    if "•" in stripped:
        return False
    t = stripped.lower()
    return bool(
        _PACE_RE.search(stripped)
        or _REST_RE.search(stripped)
        or " at " in t
        or "warm up" in t or "warmup" in t
        or "cool down" in t or "cooldown" in t
        or "conversational" in t
        or "strength" in t
    )


def _expand_repeat_blocks(lines: list[str]) -> list[str]:
    """Expand a repeat-block header ("5 reps of:" / "Repeat the following
    3x:") into N literal copies of its body, so the rest of the parser sees
    a flat list exactly as if the block had been written out longhand.

    Two body shapes are observed from Runna:
    - A single compound line right after the header (e.g. "0.62mi at
      7:25/mi ..., 90s walking rest") - the older "N reps/sets/x of:" style.
    - A "----------"-fenced block of one or more lines after a "Repeat the
      following Nx:" header (e.g. a work interval followed by its own
      recovery interval as two separate lines) - each rep repeats the WHOLE
      fenced group together, not just its first line.

    Runna's plain-text export gives no indentation or other structural
    marker for a block's body, so the body can't be told apart from a
    following top-level line (e.g. a cooldown) by line count alone - only
    the fence (or, without one, the next single non-blank line) marks where
    the body ends.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = _REPS_RE.match(line)
        if not m:
            out.append(lines[i])
            i += 1
            continue
        count = int(m.group(1) or m.group(2))
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j < len(lines) and _DASH_RULE_RE.match(lines[j].strip()):
            # Fenced multi-line body: collect every segment line up to the
            # closing dash rule (or, failing that, the next blank/non-
            # segment line - a missing closing fence shouldn't swallow the
            # rest of the workout).
            k = j + 1
            body_lines: list[str] = []
            while k < len(lines):
                stripped = lines[k].strip()
                if _DASH_RULE_RE.match(stripped):
                    k += 1
                    break
                if not stripped or not _is_segment_line(stripped):
                    break
                body_lines.append(stripped.lstrip("•").strip())
                k += 1
            if body_lines:
                # Every rep's rest is short enough to fold into the work
                # before it (see the merge pass in parse_workout), so the
                # chart ends up showing one long flat segment with no visible
                # sign the reps or rests exist at all - tag the first line of
                # each rep with "Nx ..." up front so it's clear this block
                # was picked up and merged, not missed.
                for rep in range(max(count, 0)):
                    for idx, body in enumerate(body_lines):
                        if idx == 0 and count > 1:
                            out.append(f"{count}x {body}")
                        else:
                            out.append(body)
                i = k
                continue
            # Fence with no recognizable segment lines inside it - fall
            # through to treat the header itself as unmatched below.
            i = j
            continue

        if j < len(lines) and _is_segment_line(lines[j].strip()):
            body = lines[j].strip().lstrip("•").strip()
            tagged = f"{count}x {body}" if count > 1 else body
            for _ in range(max(count, 0)):
                out.append(tagged)
            i = j + 1
        else:
            i = j
    return out


def parse_workout(lines: list[str], easy_pace_sec: float = DEFAULT_EASY_PACE) -> list[Segment]:
    """Parse Runna segment lines into Segments (rest parts split out)."""
    segments: list[Segment] = []
    for raw in _expand_repeat_blocks(lines):
        line = raw.strip().lstrip("•").strip()
        if not line or not _is_segment_line(line):
            continue

        rest_m = _REST_RE.search(line)
        run_part = line[: rest_m.start()].rstrip(", ") if rest_m else line

        ceiling_m = _PACE_CEILING_RE.search(run_part)
        pace_run_part = _PACE_CEILING_RE.sub("", run_part)
        pace_m = _PACE_RE.search(pace_run_part)
        dist_m = _DIST_RE.search(run_part)
        if pace_m:
            pace = int(pace_m.group(1)) * 60 + int(pace_m.group(2))
        elif ceiling_m:
            # No separate "at X:XX/mi" target — the ceiling clause ("no
            # faster than X:XX/mi") is the only pace given for this segment,
            # so it IS the target, not just a bound to check the target
            # against. Search the ceiling text itself, not pace_run_part
            # (which just had it stripped out).
            ceiling_pace_m = _PACE_RE.search(ceiling_m.group(0))
            pace = (int(ceiling_pace_m.group(1)) * 60 + int(ceiling_pace_m.group(2))) if ceiling_pace_m else easy_pace_sec
        else:
            pace = easy_pace_sec
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
            # BPM-matched tracks instead of leaving seg.bpm unset. A rest
            # explicitly called out as "walking" is slower still - floor it
            # at 20:00/mi rather than the (jogging-speed) easy pace.
            rest_pace = easy_pace_sec
            if "walk" in rest_m.group(0).lower():
                rest_pace = max(rest_pace, WALKING_REST_MIN_PACE_SEC)
            segments.append(Segment(rest_m.group(0), "rest", rest_sec, rest_pace))

    # Music plays through short rests - fold them into the previous segment's
    # track-fill pass (a standalone segment this short would force an early,
    # cut-off track change just for the rest). The label still records the
    # fold so downstream UI (e.g. the activity timeline strip) can show the
    # rest as its own block even though it shares tracks with what preceded it.
    merged: list[Segment] = []
    for seg in segments:
        if seg.kind == "rest" and seg.duration_sec < 120 and merged:
            merged[-1].duration_sec += seg.duration_sec
            merged[-1].label = f"{merged[-1].label} + {seg.label}"
            merged[-1].folded_rest_sec += seg.duration_sec
            merged[-1].folded_rest_pace_sec = seg.pace_sec
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
    """Distance to target allowing half/double-time matches, mirroring bpm_filter.

    Always symmetric, even for "work" segments (see bpm_filter's
    no_upper_limit) - an on-target track should still rank ahead of an
    unnecessarily fast one; only bpm_filter's exclusion cutoff is asymmetric,
    not this ranking distance.
    """
    return min(abs(tempo - bpm), abs(tempo * 2 - bpm), abs(tempo / 2 - bpm))


def _segment_pool(
    library: pd.DataFrame, seg: Segment, used: set, min_pool: int, budget_sec: float,
    easy_bias_sec: float = 0.0, used_artists: set | None = None, played: set | None = None,
    bpm_bounds: tuple[float | None, float | None] = (None, None), avoid: set | None = None,
    play_counts: dict[str, int] | None = None, boosted: set | None = None,
    sweet_bpm: float | None = None,
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
    # tempo per run type (e.g. easy max 168). Hard-effort ("work") segments
    # keep the tight default tolerance since pace really counts there;
    # warmup/easy/cooldown/rest can widen further — see WIDE_TOLERANCE_KINDS.
    tolerances = BPM_TOLERANCES_WIDE if seg.kind in WIDE_TOLERANCE_KINDS else BPM_TOLERANCES
    attempts = [(tol, pad) for tol in tolerances for pad in (0.0, 0.1, 0.2, 0.45, 1.0)]
    # Last resorts: with one track per artist, a long segment can exhaust the
    # unique artists near the target BPM — better off-tempo music at the back
    # of the pool than silence mid-run.
    attempts += [(12.0, 1.0), (None, 1.0)]

    # Hard-effort segments: running faster than the planned pace is fine
    # (even good), so a track above target BPM is never filtered out or
    # penalized for it - only a track slower than target is a real mismatch.
    no_upper_limit = seg.kind == "work"

    # Play-count tiers gate which tracks are even eligible, not just how
    # they're ranked: for a given play-count tier, only tracks with that many
    # (or fewer) confirmed plays are offered; only once the lowest tier
    # (0 plays) can't fill the segment's duration budget at ANY BPM/energy
    # tolerance does the pool expand to the next tier (0-1 plays), then
    # 0-1-2, and so on up to PLAY_COUNT_WEIGHT_CAP // PLAY_COUNT_WEIGHT plays.
    # Tier selection is judged purely on the duration budget, never on
    # min_pool's breadth floor (WIDE_TOLERANCE_KINDS wants up to
    # MAX_CANDIDATES candidates for LLM variety, which has nothing to do with
    # how much music the segment's actual runtime needs) - otherwise a short
    # segment would drag in played tracks solely to hit that breadth target.
    # Once a tier fills the budget, BPM/energy tolerance keeps widening
    # *within that same tier* to gather up to min_pool candidates for variety,
    # but never spills into a higher play-count tier to do so. A thumbs-upped
    # track is always eligible in the tier-0 pool regardless of its real play
    # count, so it's never excluded by this gating. play_counts=None (or
    # empty) skips tiering entirely and behaves as before (BPM/energy
    # attempts only, judged against min_pool as it always was).
    max_tier = PLAY_COUNT_WEIGHT_CAP // PLAY_COUNT_WEIGHT
    play_tiers = list(range(max_tier + 1)) if play_counts else [None]

    def _pool_for(tol, pad, max_plays):
        pool = bpm_filter(library, seg.bpm, tolerance=tol, no_upper_limit=no_upper_limit) if seg.bpm and tol else library
        pool = pool[
            (pool["Energy"] >= max(e_lo - pad, 0))
            & (pool["Energy"] <= min(e_hi + pad, 1))
        ]
        pool = pool[~pool["Track URI"].isin(used)] if "Track URI" in pool.columns else pool
        if used_artists:
            pool = pool[~pool["Artist Name(s)"].map(_primary_artist).isin(used_artists)]

        if max_plays is not None:
            eligible = pool["Track URI"].map(lambda u: min(play_counts.get(u, 0), max_tier)) <= max_plays
            if boosted:
                eligible |= pool["Track URI"].isin(boosted)
            pool = pool[eligible]

        # One track per artist, applied BEFORE the budget check below so the
        # relaxation loop keeps widening until enough unique-artist music
        # exists to fill the whole segment. Keep each artist's closest-to-BPM
        # track so the dedupe costs as little tempo accuracy as possible —
        # with already-played-at-this-pace tracks sorting behind unplayed
        # ones, so an artist's fresh track wins over their played one.
        # Sweet spot (Settings) is a ranking preference, not a filter - it
        # never excludes a track (the bpm_bounds pre-filter above and the
        # bpm_filter tolerance-tier call above both stay anchored on
        # seg.bpm, the pace-derived cadence target, unaffected by this).
        # When set, it substitutes for seg.bpm as the ranking anchor rather
        # than blending with it, so a track exactly at the sweet spot always
        # outranks one merely closer to the cadence target - matching "tracks
        # with this BPM should always be selected first". Falls back to
        # today's exact behavior (dist anchored on seg.bpm) when unset.
        anchor = sweet_bpm or seg.bpm
        if anchor:
            dist = pool["Tempo"].map(lambda t: _bpm_distance(float(t), anchor))
        else:
            # No tempo target (strength): keep each artist's highest-energy track
            dist = -pool["Energy"].astype(float)
        if play_counts:
            # Within a play-count tier, dist (BPM/energy fit) still breaks
            # ties — see PLAY_COUNT_WEIGHT above. A thumbs-upped track
            # always gets weight 0 regardless of how many times it's
            # played, so it never loses a spot in the pool (and therefore
            # the LLM's candidate list) to its own play count.
            weight = pool["Track URI"].map(lambda u: min(play_counts.get(u, 0), max_tier) * PLAY_COUNT_WEIGHT)
            weight = weight.clip(upper=PLAY_COUNT_WEIGHT_CAP)
            if boosted:
                weight = weight.where(~pool["Track URI"].isin(boosted), 0)
            pool = pool.loc[pd.DataFrame({"weight": weight, "dist": dist}).sort_values(["weight", "dist"], kind="stable").index]
        else:
            pool = pool.loc[dist.sort_values(kind="stable").index]
        pool = pool.loc[~pool["Artist Name(s)"].map(_primary_artist).duplicated()]
        # Fresh tracks always precede played/avoided ones as two separate
        # blocks (not a combined distance score) - with a large pool, a
        # blended "+1000 penalty" can still let a played/avoided track land
        # ahead of some fresh ones by index once the fresh block runs past
        # MAX_CANDIDATES, since only the pool's *row order* determines what
        # choose_setlist's head(MAX_CANDIDATES) slice actually shows the LLM.
        # A hard split guarantees every fresh track outranks every
        # played/avoided one regardless of pool size.
        if played:
            is_played = pool["Track URI"].isin(played)
            pool = pd.concat([pool[~is_played], pool[is_played]])
        return pool

    def _fills_budget(pool, tol, max_plays):
        # Without play-count tiering, preserve the original contract exactly:
        # a pool isn't accepted until it clears min_pool too, not just the
        # duration budget.
        if max_plays is None and len(pool) < min_pool:
            return False
        if pool["Duration (ms)"].sum() / 1000 < budget_sec:
            return False
        # Remix: a tight pool can be nothing but the avoided tracks, and
        # demoting them changes nothing — keep widening until fresh music
        # alone covers the budget (the tol=None last resort still returns
        # whatever exists, so genuinely dry pools fall back gracefully).
        if avoid and tol is not None:
            fresh_sec = pool.loc[~pool["Track URI"].isin(avoid), "Duration (ms)"].sum() / 1000
            if fresh_sec < budget_sec:
                return False
        return True

    last_pool = None
    for max_plays in play_tiers:
        tier_fill = None  # (tol, pad) of the loosest attempt that fills the budget in this tier
        for tol, pad in attempts:
            pool = _pool_for(tol, pad, max_plays)
            last_pool = pool
            if _fills_budget(pool, tol, max_plays):
                tier_fill = (tol, pad)
                break
        if tier_fill is None:
            # This tier can't fill the segment even at the widest BPM/energy
            # tolerance — move to the next play-count tier rather than
            # settling for a short pool.
            continue
        # Tier found. Keep widening BPM/energy tolerance *within this same
        # tier* for breadth (min_pool), but never drop below the fill point
        # found above and never cross into a higher play-count tier.
        best = pool
        for tol, pad in attempts[attempts.index(tier_fill):]:
            pool = _pool_for(tol, pad, max_plays)
            if not _fills_budget(pool, tol, max_plays):
                continue
            best = pool
            if len(pool) >= min_pool:
                break
        if play_counts:
            counts = best["Track URI"].map(lambda u: min(play_counts.get(u, 0), max_tier))
            breakdown = ", ".join(f"{n}x{c}" for c, n in counts.value_counts().sort_index().items())
            _log(
                f"'{seg.label}': pool settled at play-count tier <= {max_plays} "
                f"({len(best)} tracks, {breakdown})"
            )
        return best.reset_index(drop=True)

    # No play-count tier filled the budget at any tolerance — fall back to
    # the loosest attempt's raw pool (matches the pre-tiering behavior for a
    # genuinely dry library).
    return (last_pool if last_pool is not None else library).reset_index(drop=True)


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


def _bpm_smooth_order(chosen: pd.DataFrame, target_bpm: float | None) -> pd.DataFrame:
    """Reorders a segment's already-selected tracks slowest to fastest by
    effective BPM, so playback steadily climbs across the segment with the
    smallest possible jump between every consecutive pair. No-op for
    target_bpm=None (strength segments have no tempo target)."""
    if len(chosen) <= 1 or target_bpm is None:
        return chosen
    eff = chosen["Tempo"].map(lambda t: _effective_run_tempo(float(t)))
    return chosen.loc[eff.sort_values(kind="stable").index]


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
    play_counts: dict[str, int] | None = None,
    bpm_overrides: dict | None = None,
    min_total_sec: float | None = None,
    avoid_tracks: list[str] | None = None,
    effort: str | None = None,
    progress=None,
    on_llm=None,
) -> pd.DataFrame:
    """Fill every segment with BPM-matched tracks; returns rows with a
    Segment label and cumulative timing columns.

    track_feedback: [{"uri", "paceSec", "vote": "up"|"down"}] — downvoted
    tracks are dropped from segments whose pace is within
    FEEDBACK_PACE_TOLERANCE of the vote's pace; upvoted ones are moved to
    the front of that segment's setlist so they play more often.

    avoid_tracks: URIs from the mix being rebuilt ("Remix") — demoted the
    same way as already-played tracks, at any pace, so a remix comes out
    mostly different but the pool can still fall back on them if it runs dry.

    play_counts: {uri: confirmed-mix-appearance-count} across all history,
    not just this pace band. Used as a graded tiebreaker within the BPM-
    sorted candidate pool — least-played tracks sort earlier, so they're
    more likely to survive the MAX_CANDIDATES cut and be shown to the LLM.
    This is separate from played_tracks' pace-specific hard demotion: a
    track can be low-priority here (played often) while still being
    eligible, unlike played_tracks which pushes same-pace repeats to the
    very back of the per-artist dedup.

    easy_bias_sec > 0 means recent easy runs came out that much faster than
    target (sec/mi): easy-type segments are then built as if their pace were
    that much slower (lower SPM) with a lower energy ceiling. Easy pace is a
    ceiling ("no faster than"), so the bias only ever slows the music down.

    progress: optional callable(done, total, segment_label) invoked as each
    segment starts building — lets callers stream a live progress bar (the
    per-segment LLM call is the slow part).
    """
    # Tracks without a duration can't be time-budgeted — NaN would poison the
    # cumulative sums ("cannot convert float NaN to integer"). Drop them.
    if "Duration (ms)" in library.columns:
        library = library[library["Duration (ms)"].notna()]

    easy_bias_sec = min(max(easy_bias_sec, 0.0), 30.0)
    for seg in segments:
        # Outside the pace_sec guard below (unlike the clamp) so it's set
        # uniformly - naturally None for strength since bpm_overrides never
        # has a "strength" key from the Settings side.
        seg.sweet_bpm = _kind_bpm_sweet(seg.kind, bpm_overrides)
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

    def _progress(seg_idx: int, label: str, detail: str | None = None, candidate_uris: list[str] | None = None):
        if not progress:
            return
        try:
            progress(seg_idx, len(segments), label, detail, candidate_uris)
        except TypeError:
            try:
                progress(seg_idx, len(segments), label, detail)
            except TypeError:
                progress(seg_idx, len(segments), label)  # older 3-arg callers
        except Exception:
            pass

    llm_failures: list[str] = []
    for seg_idx, seg in enumerate(segments):
        _progress(seg_idx, seg.label)
        is_last = seg is segments[-1]
        budget = seg.duration_sec - carry
        # Previous overshoot already covers this segment (but the final
        # segment's budget is a hard minimum — only skip it when fully covered).
        if budget <= (0 if is_last else 30):
            carry = -budget
            continue

        downvoted = _feedback_uris(seg.pace_sec, "down")
        boosted = _feedback_uris(seg.pace_sec, "up")
        avoid = set(avoid_tracks or [])
        played = _played_uris(seg.pace_sec) | avoid
        lib_for_seg = library[~library["Track URI"].isin(downvoted)] if downvoted else library
        # _segment_pool's tolerance-widening loop stops the INSTANT it clears
        # min_pool, even if far more close-BPM tracks exist just one tier
        # out — with the old flat min_pool=20, a large library easily has
        # 20+ tracks within the tightest (3 BPM) tolerance, so the loop
        # returned right there and never considered the rest. For the
        # "wide tolerance" kinds (easy/cooldown/rest/warmup — WIDE_TOLERANCE_
        # KINDS above), pull the floor up to MAX_CANDIDATES itself so the
        # search keeps widening until either it's seen every reasonably
        # close track or run out of headroom, giving the LLM (and the
        # deterministic fallback) the full available choice to sequence a
        # tight, smoothly-transitioning segment from — not just the bare
        # minimum needed to fill the time budget. "work" stays at 20: pace
        # accuracy matters more than candidate breadth there, and its
        # tolerance is already the tight, non-widening default.
        seg_min_pool = MAX_CANDIDATES if seg.kind in WIDE_TOLERANCE_KINDS else 20
        pool = _segment_pool(
            lib_for_seg, seg, used, min_pool=seg_min_pool, budget_sec=budget, easy_bias_sec=easy_bias_sec,
            used_artists=used_artists, played=played, play_counts=play_counts, boosted=boosted or None,
            bpm_bounds=_kind_bpm_bounds(seg.kind, bpm_overrides), avoid=avoid or None,
            sweet_bpm=seg.sweet_bpm,
        )
        if pool.empty:
            _log(f"No tracks fit segment '{seg.label}' - skipping.")
            continue

        median_sec = pool["Duration (ms)"].median() / 1000
        n_est = min(math.ceil(budget / median_sec) + 1, len(pool))
        # The LLM sees the whole settled-tier pool (every 0-play track that
        # passed BPM/energy filtering, or the next tier up only if 0-play
        # tracks alone can't fill the segment - see _segment_pool), not just
        # enough to hit n_est: capping the candidate list tighter than that
        # once made it just as easy for the model to reach for a played track
        # a few rows down as an unplayed one, since a small list gives it too
        # little room to express a BPM/mood preference without leaving the
        # 0-play tier. choose_setlist still hard-caps at MAX_CANDIDATES on
        # its own for model-quality reasons (see its docstring).
        llm_pool = pool

        ordered = None
        if use_llm:
            cadence_line = f"Cadence target {seg.bpm:.0f} steps/min. " if seg.bpm else ""
            sweet_line = (
                f"Sweet spot BPM {seg.sweet_bpm:.0f} - strongly prefer tracks at or very near this "
                "BPM (candidates are listed closest-to-sweet-spot first; favour tracks near the top "
                "of the list over ones further down) unless another factor makes a close-BPM track a "
                "clearly worse fit. "
                if seg.sweet_bpm else ""
            )
            prompt = (
                f"Section of a run workout: {seg.label}. "
                f"{cadence_line}{sweet_line}"
                + {
                    "warmup": "Easing in - upbeat but not full throttle.",
                    "work": "Hard effort - driving, motivating, relentless. Faster than the cadence target is fine, even better - only slower tracks are a mismatch here.",
                    "easy": "Conversational effort - chilled, laid-back, mellow; nothing aggressive or high-energy.",
                    "cooldown": "Winding down - relaxed and light.",
                    "rest": "Recovery - calm.",
                    "strength": "Strength training - up-tempo, high-energy, powerful and motivating; any BPM.",
                }[seg.kind]
            )
            target_bpm_str = f"{seg.bpm:.0f}" if seg.bpm else "none"
            # llm_pool (capped to candidate_cap above) is what's actually
            # sent - report that count, plus the full pool size when it's
            # bigger, so the progress line doesn't overstate how many the LLM
            # actually sees.
            sent_count = len(llm_pool)
            pool_desc = f"{sent_count} candidates" if sent_count == len(pool) else f"{sent_count} of {len(pool)} candidates"
            sweet_str = f", sweet {seg.sweet_bpm:.0f}" if seg.sweet_bpm else ""
            # Play-count breakdown of what's actually sent to the LLM (not
            # the wider `pool`) - "0x9, 1x4" reads as 9 never-played tracks
            # and 4 one-play tracks made the candidate list, so a mix that
            # still picks played tracks can be told apart from one that had
            # no fresher choice to offer.
            plays_str = ""
            if play_counts:
                sent_counts = llm_pool["Track URI"].map(lambda u: play_counts.get(u, 0))
                plays_str = ", plays " + ", ".join(f"{c}x{n}" for c, n in sent_counts.value_counts().sort_index().items())
            _log(
                f"'{seg.label}': sending {pool_desc} to {model} "
                f"(target {target_bpm_str} BPM{sweet_str}, pool range "
                f"{pool['Tempo'].min():.0f}-{pool['Tempo'].max():.0f} BPM, count target {n_est}{plays_str})"
            )
            candidate_uris = llm_pool["Track URI"].tolist() if "Track URI" in llm_pool.columns else None
            _progress(seg_idx, seg.label, f"Sending {pool_desc} to {model}…", candidate_uris)
            try:
                ordered, _ = choose_setlist(prompt, llm_pool, n_est, model, effort=effort, on_llm=on_llm)
                picked_bpm = ordered["Tempo"].tolist() if not ordered.empty else []
                picked_plays_str = ""
                if play_counts and not ordered.empty:
                    picked_plays_str = f", plays: {ordered['Track URI'].map(lambda u: play_counts.get(u, 0)).tolist()}"
                _log(f"'{seg.label}': {model} returned {len(ordered)} tracks, BPMs: {picked_bpm}{picked_plays_str}")
                _progress(seg_idx, seg.label, f"{model} returned {len(ordered)} tracks")
            except Exception as e:
                _log(f"LLM selection failed for '{seg.label}' ({e}); using distance chain.")
                llm_failures.append(f"{seg.label}: {e}")
                _progress(seg_idx, seg.label, f"{model} call failed — falling back to BPM matching")
        if ordered is None or ordered.empty:
            ordered = pool.iloc[_chain_order(pool, prev_tail)]

        # Always extend the ordering with the rest of the pool: the LLM's
        # setlist is sized to the budget, so without the leftovers the
        # played/avoided demotion below has no fresh alternatives to promote
        # and demoted tracks still make the cut. Extras past the budget are
        # never reached by _fit_duration, so undemoted mixes are unchanged.
        # Compare by Track URI, not index — choose_setlist resets its result's
        # index, so index-based exclusion would re-add already-picked tracks
        # (duplicate index labels then make _fit_duration's .loc explode each
        # pick into multiple rows).
        leftover = pool[~pool["Track URI"].isin(ordered["Track URI"])].reset_index(drop=True)
        if not leftover.empty:
            extra = leftover.iloc[_chain_order(leftover, ordered.tail(1))]
            ordered = pd.concat([ordered, extra], ignore_index=True)

        # Played-but-unvoted tracks drop to the back of the ordering, so they
        # only make the cut when the unplayed pool can't fill the budget.
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
        # Playback order within the segment: track *selection* above (LLM
        # picks / deterministic fallback, played/boosted demotion, artist
        # dedup, budget fit) is untouched — this only resequences the
        # already-chosen set into a greedy nearest-effective-BPM chain, so
        # every consecutive pair's BPM jump is as small as the available
        # tracks allow (opens on whichever track sits closest to the
        # segment's target). Replaces an earlier version that force-sorted
        # every segment slowest-to-fastest regardless of what was selected
        # (produced an artificial rising staircase) — this instead minimizes
        # jumps in whichever direction, without imposing a fixed direction.
        chosen = _bpm_smooth_order(chosen, seg.bpm)
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
    # Segments where the LLM call failed and fell back to the deterministic
    # distance-chain — surfaced by server.py/ai_dj_bridge.py as a warning so
    # a rate-limited/quota-exhausted model doesn't silently degrade the mix.
    playlist.attrs["llm_failures"] = llm_failures
    return playlist


def _mmss(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def build_flow_mix(
    title: str,
    library: pd.DataFrame,
    track_uris: list[str],
    model: str,
    use_llm: bool = True,
    track_feedback: list[dict] | None = None,
    play_counts: dict[str, int] | None = None,
    effort: str | None = None,
    progress=None,
    budget_sec: float | None = None,
) -> pd.DataFrame:
    """Sequence a fixed, caller-chosen set of tracks (e.g. every track in a
    selected HR zone) for smooth transitions, instead of picking tracks to
    fit workout segments. BPM/energy/key act only as a local
    smoothness guide between neighbouring tracks - never a target to hit.

    Without budget_sec, every track in `track_uris` that exists in the
    library appears exactly once in the result (nothing is filtered out by
    pace/BPM fit). With budget_sec (e.g. the next scheduled run's estimated
    duration), the flow-ordered list is trimmed to that length the same way
    a workout segment is (_fit_duration) - the track that crosses the
    boundary is only kept when overshooting beats stopping short.

    track_feedback: [{"uri", "vote": "up"|"down"}] - downvoted tracks are
    dropped outright (no pace scoping, unlike the segment-based mixer,
    since a flow mix has no per-track pace target); upvoted tracks are
    boosted to sort earlier as a tiebreak.

    play_counts: {uri: confirmed-mix-appearance-count} - least-played tracks
    sort earlier (see PLAY_COUNT_WEIGHT); ties within the same play count are
    broken by the model/chain order rather than being excluded outright.
    """
    pool = library[library["Track URI"].isin(track_uris)].reset_index(drop=True)
    if "Duration (ms)" in pool.columns:
        pool = pool[pool["Duration (ms)"].notna()]
    if pool.empty:
        raise ValueError("None of the selected tracks were found in the library.")

    downvoted = {f.get("uri") for f in (track_feedback or []) if f.get("vote") == "down" and f.get("uri")}
    upvoted = {f.get("uri") for f in (track_feedback or []) if f.get("vote") == "up" and f.get("uri")}
    if downvoted:
        pool = pool[~pool["Track URI"].isin(downvoted)].reset_index(drop=True)
    if pool.empty:
        raise ValueError("Every selected track was thumbs-downed.")

    if play_counts:
        # A thumbs-upped track always gets weight 0 regardless of play count.
        weight = pool["Track URI"].map(lambda u: min(play_counts.get(u, 0), 10) * PLAY_COUNT_WEIGHT).clip(upper=PLAY_COUNT_WEIGHT_CAP)
        if upvoted:
            weight = weight.where(~pool["Track URI"].isin(upvoted), 0)
        pool = pool.loc[weight.sort_values(kind="stable").index].reset_index(drop=True)

    if progress:
        try:
            progress(0, 1, title, f"Sending {min(len(pool), MAX_CANDIDATES)} tracks to {model}…" if use_llm else None)
        except Exception:
            pass

    ordered = None
    llm_failures: list[str] = []
    if use_llm:
        try:
            ordered, _ = choose_flow_order(pool, model, effort=effort)
        except Exception as e:
            _log(f"Flow-mix LLM ordering failed ({e}); using distance chain.")
            llm_failures.append(f"{title}: {e}")
    if ordered is None or ordered.empty:
        ordered = pool.iloc[_chain_order(pool, None)].reset_index(drop=True)

    if upvoted:
        is_boost = ordered["Track URI"].isin(upvoted)
        if is_boost.any():
            ordered = pd.concat([ordered[is_boost], ordered[~is_boost]]).reset_index(drop=True)

    if budget_sec is not None and ordered["Duration (ms)"].sum() / 1000 > budget_sec:
        ordered = _fit_duration(ordered, budget_sec).reset_index(drop=True)

    if progress:
        try:
            progress(1, 1, title, f"{model} returned {len(ordered)} tracks" if use_llm else None)
        except Exception:
            pass

    ends = ordered["Duration (ms)"].cumsum() / 1000
    ordered["Starts At"] = (ends - ordered["Duration (ms)"] / 1000).map(_mmss)
    ordered["Segment"] = title
    ordered["Target BPM"] = pd.NA
    ordered["Target Pace"] = pd.NA
    ordered.attrs["llm_failures"] = llm_failures
    return ordered


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
    block_id = (playlist["Segment"] != playlist["Segment"].shift()).cumsum()
    for _, group in playlist.groupby(block_id, sort=False):
        seg_label = group["Segment"].iloc[0]
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
