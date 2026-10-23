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


# Widest a strict-BPM step-out search will try before giving up and
# returning the library unfiltered (which then likely leaves the segment
# empty downstream, same as "no candidates at all" always has) - generous
# enough to find literally any populated BPM in a real library without
# risking a runaway loop on a pathological one.
_STRICT_BPM_MAX_STEP = 40

def _exact_bpm_pool(
    library: pd.DataFrame, target_bpm: float, bpm_bounds: tuple[float | None, float | None] = (None, None),
) -> pd.DataFrame:
    """Every mix segment with a BPM target now gets EXACTLY one whole-number
    BPM for its entire candidate pool, not a tolerance band — restricts
    `library` to tracks whose raw tempo, doubled, or halved rounds to
    target_bpm (round-to-nearest), same half/double-time symmetry the
    literal-BPM/AI-Remix filter already used (a half/double-time track
    genuinely plays at the target cadence once halved/doubled, so it's a
    real match, not just a close one). If nothing rounds to the exact
    target, steps outward by whole BPM (target+1, target-1, target+2,
    target-2, ...) until a populated BPM is found - every segment still
    gets music, but always from a single exact BPM value, never a widened
    band. Returns the (possibly empty, if the whole step-out range was dry)
    filtered library; the caller logs which BPM actually got used when it
    differs from the original target.

    bpm_bounds (a kind's Settings min/max, compared against _effective_run_
    tempo same as the hard filter _segment_pool applies on top of this)
    narrows which stepped-out BPM values are even considered a candidate -
    without this, the step-out search could lock onto a populated BPM that
    _segment_pool's OWN bpm_bounds filter then immediately wipes back out
    again downstream, silently emptying the segment even though a real,
    in-range match existed just one or two steps further out.
    """
    target_rounded = round(target_bpm)
    lo, hi = bpm_bounds

    def _in_bounds(bpm: int) -> bool:
        eff = _effective_run_tempo(float(bpm))
        if lo is not None and eff < lo:
            return False
        if hi is not None and eff > hi:
            return False
        return True

    def _at(bpm: int) -> pd.DataFrame:
        match = library["Tempo"].map(
            lambda t: bpm in (round(float(t)), round(float(t) * 2), round(float(t) / 2))
        )
        return library[match]

    if _in_bounds(target_rounded):
        pool = _at(target_rounded)
        if not pool.empty:
            return pool
    for step in range(1, _STRICT_BPM_MAX_STEP + 1):
        for bpm in (target_rounded + step, target_rounded - step):
            if not _in_bounds(bpm):
                continue
            pool = _at(bpm)
            if not pool.empty:
                _log(f"No in-range tracks at exactly {target_rounded} BPM — widened to {bpm} BPM instead")
                return pool
    return library.iloc[0:0]


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
    # The pace-derived cadence target BEFORE _kind_bpm_bounds' Settings
    # ceiling/floor clamp (build_workout_playlist sets both `bpm` and this
    # together - see its own comment). `bpm` stays the CLAMPED value for
    # everything it always drove (the LLM prompt's "Cadence target" line,
    # the Target BPM output column, _bpm_smooth_order's playback arc, and
    # _segment_pool's bpm_bounds/ranking anchor) - this field exists solely
    # for the strict exact-BPM pool lock (_exact_bpm_pool), which must use
    # the REAL pace target rather than the clamp: a segment whose pace
    # genuinely calls for 185 BPM but gets clamped to a 170 "work" ceiling
    # otherwise locks onto exactly 170, same as every OTHER segment the
    # clamp also caps at 170 - flattening every fast segment onto one
    # identical BPM instead of each locking to its own real cadence (the
    # Settings ceiling/floor still applies as a hard boundary via
    # bpm_bounds, same as before - it just no longer collapses the LOCK
    # target itself). None whenever `bpm` is (strength, or clamping never
    # ran e.g. strict_pace_tolerance/literal_bpm).
    raw_bpm: float | None = None
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
    # True for a segment built from an explicit "Nmin at Xbpm" line (the
    # Running app's AI Remix) rather than a pace — `bpm` is the user's own
    # literal target, so build_workout_playlist must never apply the
    # Settings sweet-spot preference or run-type bounds on top of it (both
    # are pace-derived-target concepts; a directly-typed BPM has nothing to
    # be swayed or clamped toward).
    literal_bpm: bool = False
    # Identifies which "Repeat the following Nx:" block (if any) this
    # segment came from, and which occurrence (0-based) within it - set by
    # _expand_repeat_blocks, consumed by the merge pass in parse_workout
    # that combines a too-short repeat body's segments (see that pass's own
    # doc comment for why: a body of several short alternating-pace
    # segments, e.g. "0.25mi at 7:15/mi, 0.25mi at 8:15/mi" repeated 3x,
    # otherwise gets only ONE real track pick for the first occurrence,
    # which overshoots so far that its carry silently skips every other
    # occurrence's own segment entirely). None for a segment that wasn't
    # part of any repeat block.
    repeat_group: int | None = None
    rep_index: int | None = None
    # A timed all-out effort with no stated pace ("60s running hard uphill",
    # strides, sprints) — build_workout_playlist targets the TOP of the
    # Settings "work" BPM range (or 180) instead of a pace-derived cadence.
    max_effort: bool = False


# ── Parsing ──────────────────────────────────────────────────────────────────

_PACE_RE = re.compile(r"(\d+):(\d+)\s*/mi")
# Literal-BPM segment ("595s at 173bpm") — the Running app's Dashboard
# chart's multi-select "AI Remix…" writes this instead of a pace, when the
# user picked a target BPM directly rather than a run pace. Distinct from
# _PACE_RE (which always means sec/mi) so a normal Runna workout line can
# never accidentally match this.
_BPM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*bpm\b", re.IGNORECASE)
# Whole-seconds duration for a literal-BPM segment only — "Nmin" (used by
# every other segment kind) can't express the sub-minute precision this
# app's ±15s budget tolerance needs.
_SEG_SEC_RE = re.compile(r"(\d+)\s*s\b", re.IGNORECASE)
_DIST_RE = re.compile(r"([\d.]+)\s*mi\b")
# "60s running hard uphill" / "20 sec strides" — a timed step.
_TIMED_SEC_RE = re.compile(r"(\d+)\s*(?:s|sec|secs|seconds)\b", re.IGNORECASE)
# Effort words meaning "as hard as you can" when no pace is given.
_MAX_EFFORT_RE = re.compile(r"\b(?:hard|uphill|hills?|sprints?|strides?|max(?:imum)?|all[\s-]?out|fast)\b", re.IGNORECASE)
# Nominal pace for a max-effort step — only keeps the pace-based merge/
# ordering logic working; its BPM target comes from max_effort instead.
MAX_EFFORT_PACE_SEC = 390.0
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
        or _BPM_RE.search(stripped)
        or _REST_RE.search(stripped)
        or " at " in t
        or "warm up" in t or "warmup" in t
        or "cool down" in t or "cooldown" in t
        or "conversational" in t
        or "strength" in t
    )


_repeat_group_counter_holder = {"next": 0}


def _next_repeat_group_id() -> int:
    _repeat_group_counter_holder["next"] += 1
    return _repeat_group_counter_holder["next"]


def _expand_repeat_blocks(lines: list[str]) -> list[tuple[str, int | None, int | None]]:
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

    Returns (text, repeat_group, rep_index) triples instead of plain text -
    repeat_group is a fresh id per repeat-block instance (None for a line
    outside any block), rep_index is which occurrence (0-based) within that
    block the line belongs to. parse_workout copies these onto each Segment
    it builds, and a later merge pass in that function uses them to combine
    a too-short repeat body's segments into one real track-fillable segment
    per occurrence (see that pass's own doc comment).
    """
    out: list[tuple[str, int | None, int | None]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = _REPS_RE.match(line)
        if not m:
            out.append((lines[i], None, None))
            i += 1
            continue
        count = int(m.group(1) or m.group(2))
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j < len(lines) and _DASH_RULE_RE.match(lines[j].strip()):
            # Fenced multi-line body: collect every segment line up to the
            # closing dash rule (or, failing that, the next non-blank,
            # non-segment line - a missing closing fence shouldn't swallow
            # the rest of the workout). A BLANK line inside the fence is
            # just Runna's normal line spacing between body lines (every
            # real export has one between "0.10mi at 7:05/mi" and "0.50mi
            # at a comfortable pace") and must be skipped, not treated as
            # the end of the body - treating it as a terminator (an earlier
            # version of this) meant a fenced body of more than one line
            # only ever collected its FIRST line before hitting the blank
            # separator and bailing, so the "Nx" repeat never actually
            # multiplied anything: the whole directive silently fell
            # through as if it were a single one-off pair, exactly matching
            # a live report of a "3x 0.10mi + 0.50mi" interval block
            # rendering as just one un-repeated instance of each.
            k = j + 1
            body_lines: list[str] = []
            while k < len(lines):
                stripped = lines[k].strip()
                if _DASH_RULE_RE.match(stripped):
                    k += 1
                    break
                if not stripped:
                    k += 1
                    continue
                if not _is_segment_line(stripped):
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
                group_id = _next_repeat_group_id() if count > 1 else None
                for rep in range(max(count, 0)):
                    for idx, body in enumerate(body_lines):
                        if idx == 0 and count > 1:
                            out.append((f"{count}x {body}", group_id, rep))
                        else:
                            out.append((body, group_id, rep))
                i = k
                continue
            # Fence with no recognizable segment lines inside it - fall
            # through to treat the header itself as unmatched below.
            i = j
            continue

        # Several "•" lines straight after the header (no "-----" fence) are
        # ALL the rep's body — e.g. "2 reps of: • 60s hard uphill… •
        # 30s hard uphill…" — not just the first one (the second used to be
        # read as a separate one-off step after the block).
        bullets: list[str] = []
        k = j
        while k < len(lines):
            st = lines[k].strip()
            if not st:
                k += 1
                continue
            if not st.startswith("•") or not _is_segment_line(st):
                break
            bullets.append(st.lstrip("•").strip())
            k += 1
        if len(bullets) >= 2:
            group_id = _next_repeat_group_id() if count > 1 else None
            for rep in range(max(count, 0)):
                for idx, body in enumerate(bullets):
                    out.append((f"{count}x {body}" if idx == 0 and count > 1 else body, group_id, rep))
            i = k
            continue

        if j < len(lines) and _is_segment_line(lines[j].strip()):
            body = lines[j].strip().lstrip("•").strip()
            tagged = f"{count}x {body}" if count > 1 else body
            for _ in range(max(count, 0)):
                # No repeat_group here - this is the single-compound-line
                # body style (one segment per rep, not an alternating pair),
                # so there's nothing for the merge pass below to combine.
                out.append((tagged, None, None))
            i = j + 1
        else:
            i = j
    return out


def parse_workout(lines: list[str], easy_pace_sec: float = DEFAULT_EASY_PACE) -> list[Segment]:
    """Parse Runna segment lines into Segments (rest parts split out)."""
    segments: list[Segment] = []
    for raw, repeat_group, rep_index in _expand_repeat_blocks(lines):
        line = raw.strip().lstrip("•").strip()
        if not line or not _is_segment_line(line):
            continue

        rest_m = _REST_RE.search(line)
        run_part = line[: rest_m.start()].rstrip(", ") if rest_m else line

        bpm_m = _BPM_RE.search(run_part)
        if bpm_m:
            # Literal-BPM segment: no pace at all, so build_workout_playlist's
            # `if seg.pace_sec:` guard leaves seg.bpm exactly as set here —
            # no pace_to_bpm conversion, no kind-based clamping. Duration is
            # an explicit whole-seconds count ("595s at 173bpm") — the
            # Running app's AI Remix writes this, never "Nmin": this app's
            # standard ±15s budget tolerance needs second-level precision, and
            # rounding to the nearest whole MINUTE (an earlier version of
            # this) could throw the target off by up to 30s before the fit
            # search even started (confirmed live: a 9:55 selection sent as
            # "10min" hunted for combinations that summed to 600s, not 595s).
            dur_m = _SEG_SEC_RE.search(run_part)
            duration = int(dur_m.group(1)) if dur_m else 0
            if duration > 0:
                # "work": the tight, symmetric BPM_TOLERANCES band — a
                # literal target BPM is exactly the "hard effort, pace
                # really counts" case that tolerance band exists for.
                literal_bpm_val = float(bpm_m.group(1))
                segments.append(Segment(
                    run_part, "work", duration, None, bpm=literal_bpm_val, raw_bpm=literal_bpm_val, literal_bpm=True,
                    repeat_group=repeat_group, rep_index=rep_index,
                ))
            continue

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
        max_effort = False
        if dist_m:
            duration = float(dist_m.group(1)) * pace
        else:
            dur_m = re.search(r"(\d+)\s*min", run_part)
            duration = int(dur_m.group(1)) * 60 if dur_m else 0
            # Timed effort with no distance — "60s running hard uphill",
            # "20s strides". Hard/uphill/sprint wording with no stated pace
            # = maximum effort.
            if duration == 0:
                sec_m = _TIMED_SEC_RE.search(run_part)
                if sec_m:
                    duration = int(sec_m.group(1))
                    if not pace_m and _MAX_EFFORT_RE.search(run_part):
                        max_effort = True
                        pace = MAX_EFFORT_PACE_SEC

        if duration > 0:
            kind = "work" if max_effort else _segment_kind(run_part)
            # A step with no stated pace runs at the conversational default —
            # treat it as easy effort, not a hard "work" interval.
            if kind == "work" and not pace_m and not max_effort:
                kind = "easy"
            # Strength has no pace: pace_sec None leaves seg.bpm unset, so
            # tracks match on energy alone (any BPM).
            segments.append(Segment(
                run_part, kind, duration, None if kind == "strength" else pace,
                repeat_group=repeat_group, rep_index=rep_index, max_effort=max_effort,
            ))

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
            if max_effort and repeat_group is not None:
                # Inside a repeat block of short all-out efforts the rests and
                # the "easy jog back down" (not timed by Runna — estimated at
                # the same time as the effort: the user's own estimate for
                # 2x [60s up + 30s walk, 30s up + 20s walk] was 7:40, which
                # this gives exactly) belong to the rep: the whole block is then
                # merged below into one maximum-effort segment.
                if re.search(r"jog\s+back", line, re.IGNORECASE):
                    rest_sec += round(duration * 1.0)
                segments.append(Segment(rest_m.group(0), "work", rest_sec, MAX_EFFORT_PACE_SEC,
                                        repeat_group=repeat_group, rep_index=rep_index, max_effort=True))
            else:
                segments.append(Segment(rest_m.group(0), "rest", rest_sec, rest_pace))

    # A repeat block whose body is SEVERAL short, alternating-pace segments
    # (e.g. "Repeat the following 3x: 0.25mi at 7:15/mi, 0.25mi at 8:15/mi")
    # otherwise gets only ONE real track pick for the first occurrence of
    # the first pace - that track overshoots the tiny (~109s) budget so far
    # (a real song runs minutes, not under two) that the leftover carry
    # silently skips every OTHER occurrence's segment outright, so the
    # chart/mix only ever shows ~1 strip per unique pace in the block
    # instead of all 3 reps (confirmed live, exact report: a real 3x
    # [0.25mi@7:15/mi, 0.25mi@8:15/mi] workout produced only one 7:15
    # track covering 5.4min against a combined 1.8min target before carry
    # ate the next two segments). Fixed per the user's own specified rule:
    #   1. Merge each occurrence's own segments into ONE segment at the
    #      HARDEST (lowest pace_sec = fastest) pace in the group - the
    #      combined DISTANCE of that occurrence's segments, run entirely at
    #      that uplifted pace, giving a real duration budget per rep.
    #   2. If even that per-occurrence duration is still too short to
    #      reliably get a real track pick (< VERY_SHORT_SEGMENT_SEC), merge
    #      ACROSS every occurrence in the group too, into one single big
    #      segment covering the whole block's combined distance at the same
    #      uplifted pace.
    # A standalone rest segment (e.g. the "90s walking rest" after this
    # block) is never part of a repeat_group, so it's untouched here and
    # still gets folded/handled by the passes below exactly as before.
    by_group: dict[int, list[Segment]] = {}
    group_order: list[int] = []
    for seg in segments:
        if seg.repeat_group is None:
            continue
        if seg.repeat_group not in by_group:
            by_group[seg.repeat_group] = []
            group_order.append(seg.repeat_group)
        by_group[seg.repeat_group].append(seg)

    replacement: dict[int, list[Segment]] = {}
    for gid in group_order:
        group_segs = by_group[gid]
        # Only pace-bearing, non-strength segments participate - a repeat
        # block's body is always work/easy/etc with a real pace in
        # practice, but guard against a stray segment with no pace_sec
        # (e.g. strength) rather than dividing by it.
        if any(s.pace_sec is None or s.pace_sec <= 0 for s in group_segs):
            continue
        uplifted_pace = min(s.pace_sec for s in group_segs)  # fastest = hardest = lowest sec/mi

        by_rep: dict[int, list[Segment]] = {}
        rep_order: list[int] = []
        for s in group_segs:
            r = s.rep_index if s.rep_index is not None else 0
            if r not in by_rep:
                by_rep[r] = []
                rep_order.append(r)
            by_rep[r].append(s)

        def _merged_segment(segs: list[Segment], pace: float, label: str) -> Segment:
            # Duration is the REAL combined time the runner actually spends
            # on this ground - each segment's own original duration (at its
            # own original pace), summed as-is. Only the BPM TARGET uses the
            # uplifted pace (every segment in the merged group gets music
            # matched to the group's hardest pace) - the time budget itself
            # must stay the real-world total, not get recomputed shorter by
            # re-deriving it from distance at the faster uplifted pace
            # (confirmed against the user's own worked example: 0.25mi@
            # 7:15/mi + 0.25mi@8:15/mi is a real ~3:53 on the ground, not
            # the ~3:38 distance-at-uplifted-pace would give).
            duration = sum(s.duration_sec for s in segs)
            return Segment(label, "work", duration, pace, repeat_group=gid,
                           max_effort=any(s.max_effort for s in segs))

        # One shared, rep-independent label built from a single occurrence's
        # own segments (e.g. "0.25mi at 7:15/mi + 0.25mi at 8:15/mi") - used
        # for every per-rep merged segment, and reused (with a "Nx" prefix)
        # if the whole-block fallback below is needed too.
        rep_label = " + ".join(s.label.split("x ", 1)[-1] for s in by_rep[rep_order[0]])
        per_rep_merged = [_merged_segment(by_rep[r], uplifted_pace, rep_label) for r in rep_order]

        if per_rep_merged and min(s.duration_sec for s in per_rep_merged) < VERY_SHORT_SEGMENT_SEC:
            # Still too short per-occurrence even after merging each rep's
            # own pair - merge every occurrence in the block together too.
            count = len(rep_order)
            whole_label = f"{count}x {rep_label}" if count > 1 else rep_label
            replacement[gid] = [_merged_segment(group_segs, uplifted_pace, whole_label)]
        else:
            replacement[gid] = per_rep_merged

    if replacement:
        rebuilt: list[Segment] = []
        seen_groups: set[int] = set()
        for seg in segments:
            if seg.repeat_group is None:
                rebuilt.append(seg)
                continue
            if seg.repeat_group in seen_groups:
                continue  # already emitted this group's replacement segment(s)
            seen_groups.add(seg.repeat_group)
            rebuilt.extend(replacement.get(seg.repeat_group, [seg]))
        segments = rebuilt

    # Music plays through short rests - fold them into an adjacent segment's
    # track-fill pass (a standalone segment this short would force an early,
    # cut-off track change just for the rest). The label still records the
    # fold so downstream UI (e.g. the activity timeline strip) can show the
    # rest as its own block even though it shares tracks with what it folded
    # into.
    #
    # Direction: folds FORWARD into the next segment when that next segment
    # is an easier kind (easy/cooldown/rest) - a rest is itself a recovery
    # moment, so it belongs with whatever slower effort follows it, not
    # tacked onto the tail of a hard interval that just finished (confirmed
    # live: a 90s walking rest after the last rep of a fast interval block,
    # immediately followed by a conversational-pace cooldown, was folding
    # BACKWARD into the interval - extending its already-hard-pace segment's
    # duration rather than becoming part of the cooldown it's actually
    # transitioning into). Only falls back to the OLD backward-fold (into
    # whatever preceded it) when there's no next segment, or the next
    # segment isn't easier (e.g. a rest sitting between two work intervals,
    # where there's no genuinely "easier" neighbor to prefer).
    merged: list[Segment] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.kind == "rest" and seg.duration_sec < 120:
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            if nxt is not None and nxt.kind in ("easy", "cooldown", "rest"):
                nxt.duration_sec += seg.duration_sec
                nxt.label = f"{seg.label} + {nxt.label}"
                nxt.folded_rest_sec += seg.duration_sec
                nxt.folded_rest_pace_sec = seg.pace_sec
                i += 1
                continue
            if merged:
                merged[-1].duration_sec += seg.duration_sec
                merged[-1].label = f"{merged[-1].label} + {seg.label}"
                merged[-1].folded_rest_sec += seg.duration_sec
                merged[-1].folded_rest_pace_sec = seg.pace_sec
                i += 1
                continue
        merged.append(seg)
        i += 1

    # A "work" segment shorter than VERY_SHORT_SEGMENT_SEC (e.g. a 0.10mi
    # sprint rep, ~40s) can't be covered start-to-finish by any real track —
    # nothing runs that short. Whatever track gets picked at the work BPM
    # necessarily plays on past the segment boundary regardless (see
    # _fit_duration_precise's own top-of-pool pick for a segment this
    # short), so giving the very next recovery-type segment (easy/cooldown/
    # rest) its OWN separate, slower-BPM track pick right at that boundary
    # produces an abrupt, unnatural cut mere seconds into the work track —
    # confirmed live: a real "3x 0.10mi @ 7:05/mi, 0.50mi comfortable"
    # interval still got two distinctly separate tracks per rep, one per
    # segment, because the work segment's overshoot (a couple minutes at
    # most) rarely eats the WHOLE following recovery segment's own budget,
    # so build_workout_playlist's normal carry-shrinks-the-next-budget
    # mechanism alone doesn't cover this case. Folding the recovery
    # segment's time directly into the short work segment here — BEFORE
    # any track-fill runs — means the whole combined window is built as
    # one segment at the work segment's own (harder) target the same way
    # a folded short rest already is, so there's simply no separate,
    # slower-BPM segment left to trigger a second track pick at all.
    fully_merged: list[Segment] = []
    i = 0
    while i < len(merged):
        seg = merged[i]
        if seg.kind == "work" and seg.duration_sec < VERY_SHORT_SEGMENT_SEC and i + 1 < len(merged) and merged[i + 1].kind in ("easy", "cooldown", "rest"):
            nxt = merged[i + 1]
            seg.duration_sec += nxt.duration_sec
            seg.label = f"{seg.label} + {nxt.label}"
            fully_merged.append(seg)
            i += 2
            continue
        fully_merged.append(seg)
        i += 1
    return fully_merged


# ── Pace -> BPM ──────────────────────────────────────────────────────────────

def garmin_cadence_buckets(db_path: str) -> dict[int, float]:
    """5-second pace bucket -> avg SPM, same query the Running app uses.

    UNFILTERED by effort/segment kind - averages cadence across every type
    of run at a given pace, from an easy jog to a hard interval that merely
    passed through that pace band transiently (e.g. mid-warmup or a brief
    recovery). Confirmed live with real data: at ~8:50/mi this returns
    ~172-173 SPM (dominated by the much larger volume of faster-workout
    samples that pass through that pace band), but genuine easy-run samples
    at that same pace cluster closer to ~168 SPM - cadence at race/work
    effort is measurably higher than cadence at the same-ish pace during an
    easy effort, so blending them pushes an easy segment's BPM target too
    high, making the mixer pick faster music than the effort calls for and
    the runner drift faster than planned (exactly this symptom, reported and
    confirmed against two real runs: an easy 8:50/mi run measuring ~168 SPM
    at that pace band, and a race at 8:00-8:30/mi measuring ~173-175 SPM).
    Kept as the fallback for callers with no segment-kind context (or no
    saved mix history to classify against) - see garmin_cadence_buckets_by_
    kind below for the fix.
    """
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


# Mirrors app/api/garmin/run-pacing/route.ts's isEasySegment() exactly - a
# segment's saved label (free text, e.g. "1mi warm up at a conversational
# pace") is classified "easy" (warmup/conversational/cooldown/recovery/rest/
# walk - allowed to run at any cadence without penalty) or "work" (an actual
# interval/race effort). Kept in sync by hand since one side is TS and the
# other Python - if this list changes, update that function too.
def _is_easy_segment_label(label: str) -> bool:
    t = label.lower()
    return any(kw in t for kw in (
        "warm up", "warmup", "cool down", "cooldown",
        "conversational", "easy", "recovery", "rest", "walk",
    ))

# A kind's own bucket needs at least this many samples before it's trusted
# over the unfiltered fallback - below this, one run's quirks (a single
# brief easy stretch, say) could swing the whole bucket. Mirrors Settings ->
# Pace Analysis's "Measured target pace -> outcome" chart's own
# MIN_RELIABLE_SAMPLES (app/api/settings/bpm-pace-chart/route.ts) - this
# function and that chart are now the SAME pipeline (pace-consistency
# filter + sample-weighted rolling smoothing), kept in sync by hand across
# the TS/Python boundary, so the mixer's actual BPM target for a pace and
# what the chart displays for that same pace should agree.
_KIND_BUCKET_MIN_SAMPLES = 5
# A track's play window only counts as one real, steady-state cadence
# sample if the pace stayed roughly constant throughout it - a window that
# spans a pace CHANGE (e.g. a fast interval running into a walking rest
# folded into the same segment, or a transition between two segments) is
# neither a clean "work" sample nor a clean "easy" one, and blending it in
# skews the cadence average toward nonsense (confirmed live: a "4x 0.5mi at
# 8:20/mi... + 90s walking rest" label matched the "rest"/"walk" keywords
# and got called easy, but its track window spanned the fast interval AND
# the walking rest, dragging that bucket's average cadence down). Checked
# via coefficient of variation (stdev / mean) on the window's own per-
# sample instantaneous pace - unitless, so the same threshold works at any
# speed, same rule and same threshold as the TS chart's own MAX_PACE_CV.
_MAX_PACE_CV = 0.08
# ±2 buckets (±10s/mi), sample-count-weighted rolling average - smooths out
# a single bucket's own small-sample noise while still tracking real trend
# shifts. Same ROLLING_WINDOW_BUCKETS the TS chart uses.
_ROLLING_WINDOW_BUCKETS = 2
_BUCKET_WIDTH_SEC = 5


def garmin_cadence_buckets_by_kind(db_path: str, app_root: str) -> dict[str, dict[int, float]]:
    """Like garmin_cadence_buckets, but split into "easy" and "work" cadence
    profiles by cross-referencing GarminDB's activity_records against this
    app's own saved "Today's Run" mix history (pacesync.db) - GarminDB
    itself has no concept of a Runna workout segment, so kind classification
    can only come from the saved tracklist's own segment labels.

    THE SAME PIPELINE as Settings -> Pace Analysis's "Measured target pace
    -> outcome" chart (app/api/settings/bpm-pace-chart/route.ts): bucketed
    by each track's own TARGET pace (not the measured actual), a track's
    play window is discarded if its pace wasn't steady throughout (pace-
    consistency filter, _MAX_PACE_CV), and each bucket's raw average is
    smoothed via a sample-count-weighted rolling average over its
    neighboring buckets (_ROLLING_WINDOW_BUCKETS). Kept in sync BY HAND
    across the TS/Python boundary - if one changes, the other must too, or
    the mixer's actual BPM target for a given pace will quietly diverge from
    what the chart displays for that same pace again.

    Returns {"easy": {bucket: avg_spm}, "work": {bucket: avg_spm}} - a kind
    with too few RAW samples in a given bucket (< _KIND_BUCKET_MIN_SAMPLES)
    simply has no entry for that bucket at all (even post-smoothing), so
    callers should fall back to garmin_cadence_buckets' unfiltered result
    for any bucket missing here. Best-effort: returns {} on any failure
    (missing pacesync.db, no confirmed history yet, etc.) so callers always
    have the unfiltered function to fall back to.
    """
    import os

    pacesync_path = os.path.join(app_root, "pacesync.db")
    if not os.path.exists(pacesync_path):
        return {}

    try:
        with sqlite3.connect(f"file:{pacesync_path}?mode=ro", uri=True) as sync_con:
            entries = sync_con.execute(
                "SELECT date, tracks_json FROM todays_run_history WHERE approved IS NULL OR approved != 0"
            ).fetchall()
    except Exception:
        return {}
    if not entries:
        return {}

    import json as _json

    samples: dict[str, list[tuple[int, float]]] = {"easy": [], "work": []}
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as garmin_con:
            garmin_con.row_factory = sqlite3.Row
            for date, tracks_json in entries:
                activity = garmin_con.execute(
                    """SELECT activity_id, start_time FROM activities
                       WHERE LOWER(sport) LIKE '%running%' AND DATE(start_time) = ?
                       ORDER BY distance DESC LIMIT 1""",
                    (date,),
                ).fetchone()
                if not activity:
                    continue
                records = garmin_con.execute(
                    """SELECT timestamp, speed, cadence FROM activity_records
                       WHERE activity_id = ? AND speed IS NOT NULL ORDER BY timestamp""",
                    (activity["activity_id"],),
                ).fetchall()
                if not records:
                    continue

                import datetime

                start_dt = datetime.datetime.fromisoformat(activity["start_time"].replace(" ", "T"))
                record_samples = []
                for r in records:
                    try:
                        t = (datetime.datetime.fromisoformat(r["timestamp"].replace(" ", "T")) - start_dt).total_seconds()
                    except ValueError:
                        continue
                    if t < 0:
                        continue
                    record_samples.append((t, r["speed"], r["cadence"]))
                if not record_samples:
                    continue
                run_end_sec = record_samples[-1][0]

                try:
                    tracks = _json.loads(tracks_json)
                except (ValueError, TypeError):
                    continue
                for track in tracks:
                    tempo = track.get("tempo")
                    starts_at = track.get("startsAtSec")
                    duration = track.get("durationSec")
                    segment_label = track.get("segment") or ""
                    target_pace_sec = track.get("targetPaceSec")
                    if tempo is None or tempo <= 0 or starts_at is None or not duration or target_pace_sec is None:
                        continue
                    if starts_at >= run_end_sec - 15:
                        continue
                    window_end = min(starts_at + duration, run_end_sec)
                    in_window = [
                        (t, mph, cadence) for t, mph, cadence in record_samples
                        if starts_at <= t < window_end and mph is not None and mph > 0.5
                    ]
                    if len(in_window) < 5:
                        continue

                    # Discard any window whose pace wasn't steady throughout
                    # - see _MAX_PACE_CV's own doc comment above. Same rule
                    # as the TS chart's own pace-consistency filter.
                    inst_paces = [3600 / mph for _, mph, _ in in_window]
                    mean_pace = sum(inst_paces) / len(inst_paces)
                    variance = sum((p - mean_pace) ** 2 for p in inst_paces) / len(inst_paces)
                    pace_cv = (variance ** 0.5) / mean_pace if mean_pace else 0
                    if pace_cv > _MAX_PACE_CV:
                        continue

                    cadence_samples = [c for _, _, c in in_window if c is not None and c > 10]
                    if len(cadence_samples) < 5:
                        continue
                    avg_spm = sum(c * 2 for c in cadence_samples) / len(cadence_samples)
                    kind = "easy" if _is_easy_segment_label(segment_label) else "work"
                    # Bucketed by the segment's own TARGET pace (not the
                    # measured actual) - same key the TS chart's own
                    # pace_to_bpm lookup is keyed by (a target pace is known
                    # in advance; pace_to_bpm's whole job is "given a target
                    # pace, what cadence should I aim for", so the lookup
                    # table must be indexed by target, not outcome).
                    bucket = round(target_pace_sec / _BUCKET_WIDTH_SEC) * _BUCKET_WIDTH_SEC
                    samples[kind].append((bucket, avg_spm))
    except Exception:
        return {}

    raw: dict[str, dict[int, float]] = {}
    counts: dict[str, dict[int, int]] = {}
    for kind, pairs in samples.items():
        by_bucket: dict[int, list[float]] = {}
        for bucket, spm in pairs:
            by_bucket.setdefault(bucket, []).append(spm)
        raw[kind] = {bucket: sum(spms) / len(spms) for bucket, spms in by_bucket.items()}
        counts[kind] = {bucket: len(spms) for bucket, spms in by_bucket.items()}

    # Sample-count-weighted rolling average over ±_ROLLING_WINDOW_BUCKETS
    # neighbors, same smoothing the TS chart applies - a 1-sample bucket
    # barely nudges the smoothed value while a 40-sample one dominates it,
    # so a near-empty bucket can't distort its neighbors as much as a
    # well-sampled one. Only a bucket that clears _KIND_BUCKET_MIN_SAMPLES
    # on ITS OWN RAW count is published at all (same floor as before) - the
    # smoothing softens noise within an already-trusted bucket, it isn't
    # what makes a sparse bucket trustworthy in the first place.
    result: dict[str, dict[int, float]] = {}
    for kind in raw:
        result[kind] = {}
        for bucket, count in counts[kind].items():
            if count < _KIND_BUCKET_MIN_SAMPLES:
                continue
            weighted_sum, weight_total = 0.0, 0
            for d in range(-_ROLLING_WINDOW_BUCKETS, _ROLLING_WINDOW_BUCKETS + 1):
                neighbor_bucket = bucket + d * _BUCKET_WIDTH_SEC
                neighbor_spm = raw[kind].get(neighbor_bucket)
                neighbor_count = counts[kind].get(neighbor_bucket, 0)
                if neighbor_spm is None or neighbor_count == 0:
                    continue
                weighted_sum += neighbor_spm * neighbor_count
                weight_total += neighbor_count
            result[kind][bucket] = weighted_sum / weight_total if weight_total > 0 else raw[kind][bucket]
    return result


def pace_to_bpm(
    pace_sec: float, cadence_buckets: dict[int, float] | None = None,
    kind_buckets: dict[str, dict[int, float]] | None = None, kind: str | None = None,
) -> float:
    # Kind-specific cadence (garmin_cadence_buckets_by_kind) wins when
    # available for this kind/pace - unfiltered cadence_buckets blends easy
    # and hard-effort samples together, which measurably skews toward the
    # higher cadence hard efforts carry even when they only pass THROUGH an
    # easy-ish pace band transiently (confirmed live: ~168 SPM for a genuine
    # easy 8:50/mi run vs. ~172-173 SPM in the unfiltered all-efforts
    # aggregate at the same pace). Falls back to the unfiltered buckets
    # (then the formula) for any kind/bucket the kind-specific data doesn't
    # cover yet - a sparse kind_buckets is expected early on, before enough
    # confirmed runs of that kind exist.
    if kind_buckets and kind:
        bucket_map = kind_buckets.get(kind)
        if bucket_map:
            bucket = min(bucket_map, key=lambda b: abs(b - pace_sec))
            if abs(bucket - pace_sec) <= 15:
                return round(bucket_map[bucket])
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


# A workout segment this short (or shorter) gets exactly one track, picked
# for closeness to the budget alone — there's no room to search combinations
# at this scale, and a single well-matched track reads more naturally than a
# multi-track search forcing an exact sum out of a 2-3 minute window.
SHORT_SEGMENT_SEC = 180.0
# A segment shorter than THIS gets its track picked by POOL PREFERENCE
# (ordered's own top-ranked BPM/energy/play-count fit) instead of by
# closeness to the tiny budget — closest-duration actively picks the
# SHORTEST available track for a window this small (confirmed live: a
# 0.10mi @ ~42s work interval got a track picked to fit ~42s, cutting off
# right where the following recovery segment started a whole new,
# separately-picked track — jarring, and defeating the point of a hard
# interval this brief in the first place, since no track can cleanly start
# AND finish inside a 42s window anyway). A real song naturally runs
# several minutes; letting it run long and carry (build_workout_playlist's
# `carry` bookkeeping) into the following recovery segment — shrinking or
# skipping that segment's own budget once carry eats into it — means the
# fast track's own BPM keeps playing through the recovery instead of an
# abrupt cut to a new, slower track mere seconds in.
VERY_SHORT_SEGMENT_SEC = 150.0
# Bounded search effort for a >SHORT_SEGMENT_SEC segment — kept small since
# ordered can run to MAX_CANDIDATES (150) tracks and this runs once per
# workout segment; a full subset-sum search isn't worth it when this gets
# within a few seconds of exact almost every time in practice.
_FIT_DROP_ATTEMPTS = 6  # how many of the greedy prefix's own tracks to try dropping, one at a time
_FIT_SWAP_LOOKAHEAD = 6  # how many tracks past the greedy cut to try swapping in for the crossing track


def _fit_duration_precise(ordered: pd.DataFrame, budget_sec: float, overshoot: bool = False) -> pd.DataFrame:
    """Like _fit_duration, but searches nearby combinations of `ordered`
    (never reordering by preference — only choosing a different SUBSET of
    the same ranked pool) for whichever gets closest to budget_sec, instead
    of accepting whatever the first greedy pass lands on. Used for workout
    segments, where the AI DJ mix's own track boundaries are meant to line
    up with the workout's planned segment changes as closely as possible;
    the plain greedy _fit_duration remains what flow mixes (no segment
    structure to line up against) use.

    budget_sec <= SHORT_SEGMENT_SEC: picks the single closest-duration track
    from `ordered` instead of searching sums — see SHORT_SEGMENT_SEC.

    Search strategy for a longer segment: start from the greedy prefix
    (same rule _fit_duration uses — the crossing track only kept when
    overshoot beats stopping short), then try two families of nearby
    alternatives that stay close to the pool's own preference order: (a)
    swap the crossing track for one of the next few tracks in rank order,
    which may fit tighter, and (b) drop one already-picked track (starting
    from the lowest-ranked of the picked set) and see if the resulting
    shortfall is better filled by the next unpicked track. Keeps whichever
    candidate combination lands closest to budget_sec; ties keep the
    earliest (highest-ranked) combination found."""
    if ordered.empty:
        return ordered

    if budget_sec <= VERY_SHORT_SEGMENT_SEC:
        # Take the pool's own top pick (already ranked by BPM/energy/play-
        # count fit — see _segment_pool) and let it run its natural length,
        # rather than hunting for whichever track happens to be closest to
        # this tiny budget — see VERY_SHORT_SEGMENT_SEC's own comment.
        return ordered.iloc[[0]]

    if budget_sec <= SHORT_SEGMENT_SEC:
        durations = ordered["Duration (ms)"] / 1000
        best_idx = (durations - budget_sec).abs().idxmin()
        return ordered.loc[[best_idx]]

    durations = (ordered["Duration (ms)"] / 1000).tolist()
    order_idx = list(ordered.index)
    n = len(order_idx)

    def total(picked_positions: list[int]) -> float:
        return sum(durations[p] for p in picked_positions)

    # Greedy prefix, as positions into order_idx (not labels) so dropping/
    # swapping by position is straightforward.
    greedy: list[int] = []
    cum = 0.0
    crossing_pos: int | None = None
    for pos in range(n):
        dur = durations[pos]
        if cum + dur >= budget_sec:
            crossing_pos = pos
            if overshoot or not greedy or (cum + dur - budget_sec) < (budget_sec - cum):
                greedy.append(pos)
            break
        greedy.append(pos)
        cum += dur

    candidates: list[list[int]] = [greedy]

    # (a) Swap the crossing track for one of the next few candidates in rank
    # order — a later, shorter/longer track might close the gap tighter than
    # whatever happened to be next in the ranking.
    if not overshoot and crossing_pos is not None:
        base = [p for p in greedy if p != crossing_pos]
        base_total = total(base)
        for lookahead_pos in range(crossing_pos + 1, min(crossing_pos + 1 + _FIT_SWAP_LOOKAHEAD, n)):
            candidates.append(base + [lookahead_pos])

    # (b) Drop one already-picked track (lowest-ranked first, since that's
    # the least preferred to lose) and try filling the resulting shortfall
    # with the next unpicked track instead.
    if not overshoot:
        picked_set = set(greedy)
        unpicked = [p for p in range(n) if p not in picked_set]
        for drop_pos in reversed(greedy[-_FIT_DROP_ATTEMPTS:]):
            reduced = [p for p in greedy if p != drop_pos]
            reduced_total = total(reduced)
            if reduced_total >= budget_sec:
                candidates.append(reduced)
                continue
            for fill_pos in unpicked[:_FIT_SWAP_LOOKAHEAD]:
                if fill_pos in reduced:
                    continue
                candidates.append(reduced + [fill_pos])

    def miss(picked_positions: list[int]) -> float:
        return abs(total(picked_positions) - budget_sec)

    best = min(candidates, key=miss)
    return ordered.iloc[sorted(best)]


_REPEAT_PREFIX_RE = re.compile(r"^\d+x\s+", re.IGNORECASE)


def _step_key(text: str) -> str:
    """A workout line (or one part of a segment label) reduced to its run
    part - bullet, "Nx " repeat prefix and trailing rest clause removed,
    lower-cased, whitespace collapsed - so a source line and the segment(s)
    parse_workout built from it compare equal."""
    t = str(text).strip().lstrip("•").strip()
    t = _REPEAT_PREFIX_RE.sub("", t)
    m = _REST_RE.search(t)
    if m:
        t = t[: m.start()].rstrip(", ")
    return " ".join(t.lower().split())


def _resolve_line_candidates(segments: list[Segment], entries: list | None) -> list | None:
    """Advanced AI DJ Mix sends its per-step candidate lists keyed by the
    workout LINE ({"line", "uris"}) rather than by position, because
    parse_workout doesn't map lines to segments 1:1 (a "3 reps of" block
    becomes 3 segments, short rests fold into neighbours, header lines are
    skipped). Converts them to the positional per-segment list the builder
    uses: a segment takes the list of the line it came from (any part of a
    merged "A + B" label, first match wins); no match = unrestricted.
    Positional lists (Pace Pro) pass through unchanged."""
    if not entries or not any(isinstance(e, dict) for e in entries):
        return entries
    keyed: dict[str, list[str]] = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("uris"), list):
            key = _step_key(e.get("line", ""))
            if key and key not in keyed:
                keyed[key] = e["uris"]
    out: list[list[str] | None] = []
    for seg in segments:
        parts = [_step_key(part) for part in str(seg.label).split(" + ")]
        out.append(next((keyed[k] for k in parts if k in keyed), None))
    return out


def build_workout_playlist(
    segments: list[Segment],
    library: pd.DataFrame,
    model: str,
    use_llm: bool = True,
    cadence_buckets: dict[int, float] | None = None,
    kind_cadence_buckets: dict[str, dict[int, float]] | None = None,
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
    strict_pace_tolerance: bool = False,
    segment_candidate_uris: list[list[str] | None] | None = None,
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

    strict_pace_tolerance: hard-excludes any candidate whose effective BPM
    falls more than BPM_TOLERANCES[0] (the tightest, "work"-segment band)
    below each segment's own cadence-derived target — no upper limit above
    target, matching lib/pace-analysis.ts's classifyPaceFit() exactly, so a
    track this rule accepts is one Pace Analysis would also call "fits" for
    that segment's pace. Overrides bpm_overrides' per-kind bounds entirely
    (not just tightens them) for every segment, not only "work" ones —
    intended for Pace Pro mixes, where every split is itself a pace target
    and there's no warmup/easy/cooldown structure to treat more loosely.
    Segments with no pace (seg.bpm is None, e.g. strength) are unaffected.

    segment_candidate_uris: one entry per segment (same order/length as
    `segments`; None or a missing trailing entry means "no restriction" for
    that segment). When given, that segment's pool is hard-restricted to
    ONLY those URIs, on top of (not instead of) every other filter (energy,
    play-count tiering, artist dedup, downvotes, etc.) — this is stronger
    than strict_pace_tolerance, which still lets the BPM-tolerance-widening
    loop range over the whole library; this caps the library itself before
    that loop ever runs. Intended for Pace Pro's "use Pace Analysis's own
    fit list as the candidates" mode: the caller has already computed
    exactly which confirmed-play tracks fit this segment's pace (see
    lib/pace-analysis.ts) and wants the LLM to choose only from those, not
    from every library track that happens to match the BPM window.
    """
    segment_candidate_uris = _resolve_line_candidates(segments, segment_candidate_uris)

    # Tracks without a duration can't be time-budgeted — NaN would poison the
    # cumulative sums ("cannot convert float NaN to integer"). Drop them.
    if "Duration (ms)" in library.columns:
        library = library[library["Duration (ms)"].notna()]

    easy_bias_sec = min(max(easy_bias_sec, 0.0), 30.0)
    for seg in segments:
        # Outside the pace_sec guard below (unlike the clamp) so it's set
        # uniformly - naturally None for strength since bpm_overrides never
        # has a "strength" key from the Settings side. Skipped for a literal
        # BPM segment - see Segment.literal_bpm's own doc comment.
        seg.sweet_bpm = None if seg.literal_bpm else _kind_bpm_sweet(seg.kind, bpm_overrides)
        if seg.pace_sec:
            # BPM lookup always uses the segment's LITERAL target pace, never
            # adjusted by easy_bias_sec (the "recent easy runs came out
            # faster than target" auto-ease signal) - that adjustment used
            # to shift the effective pace here before the cadence-bucket
            # lookup, which made the mix's actual BPM target silently diverge
            # from what Settings -> Pace Analysis's chart shows for the same
            # target pace (confirmed live: an 8:55/mi segment with ANY
            # nonzero bias landed in the chart's 170 BPM bucket instead of
            # the chart's own displayed 167, since the shifted pace crossed
            # into a different 5s bucket). easy_bias_sec still softens the
            # ENERGY window in _segment_pool below (calmer music, same BPM) -
            # only the BPM target itself is now exempt.
            pace = seg.pace_sec
            # WIDE_TOLERANCE_KINDS (warmup/easy/cooldown/rest) maps to the
            # "easy" cadence profile, "work" to the "work" one - same binary
            # split _is_easy_segment_label/isEasySegment's own classification
            # uses, chosen over 5 separate per-kind buckets since the
            # confirmed-run dataset is small enough that finer slicing would
            # mostly just produce empty buckets.
            seg.bpm = pace_to_bpm(
                pace, cadence_buckets, kind_buckets=kind_cadence_buckets,
                kind="easy" if seg.kind in WIDE_TOLERANCE_KINDS else "work",
            )
            seg.raw_bpm = seg.bpm
            # Clamp the target into the run type's bounds so BPM matching
            # aims inside the window instead of fighting the hard filter.
            # Skipped under strict_pace_tolerance: that mode bypasses the
            # kind-based bounds system entirely (see its call site below),
            # so clamping seg.bpm into them here first would still leak
            # through and diverge from Pace Analysis's raw cadence target.
            if not strict_pace_tolerance:
                lo, hi = _kind_bpm_bounds(seg.kind, bpm_overrides)
                if seg.bpm and hi is not None:
                    seg.bpm = min(seg.bpm, hi)
                if seg.bpm and lo is not None:
                    seg.bpm = max(seg.bpm, lo)
        if seg.max_effort:
            # All-out effort: aim at the top of the Settings "work" range
            # (180 if none is set), not a pace-derived cadence.
            _lo, _hi = _kind_bpm_bounds("work", bpm_overrides)
            seg.bpm = seg.raw_bpm = float(_hi) if _hi else 180.0
            seg.sweet_bpm = None

    # Cover the workout's slowest projected duration rather than appending
    # arbitrary padding tracks: stretch the final segment's budget only by
    # whatever the segment targets fall short of it. Without a projection,
    # fall back to the old fixed pad. Never applied when the final segment
    # is a literal-BPM one (AI Remix) - that segment's whole point is to
    # fit EXACTLY the given budget, not "don't run out of music" slack
    # (confirmed live: a 595s AI Remix request came back as a 895s fill -
    # PLAYLIST_PAD_SEC's unconditional +300s was being added on top of the
    # exact budget the caller asked for, same class of bug the is_last
    # override above already fixed for the fit-search's overshoot mode -
    # this is the OTHER place "final segment" assumptions leaked in).
    total = sum(s.duration_sec for s in segments)
    if not segments[-1].literal_bpm:
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
        # A literal-BPM segment (AI Remix) is never "the last segment of a
        # real run that must not go silent" — it's a fixed-length swap-in for
        # exactly the tracks it's replacing, so it always wants the tight
        # swap/drop search below, never the greedy-overshoot-only behavior
        # `overshoot=True` gives every workout's genuinely final segment
        # (confirmed live: a 595s AI Remix selection came back 16:58 long —
        # _fit_duration_precise's greedy prefix crossed the budget on its
        # 3rd track and, with overshoot always true, kept it unconditionally
        # instead of ever checking whether stopping at 2 tracks, or swapping
        # the 3rd for a shorter one, would land closer).
        is_last = seg is segments[-1] and not seg.literal_bpm
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

        has_explicit_candidates = bool(
            segment_candidate_uris and seg_idx < len(segment_candidate_uris) and segment_candidate_uris[seg_idx]
        )
        # Advanced AI DJ Mix's per-segment BPM range: the candidate list IS
        # the user's BPM choice, so aim the segment at it — target = median
        # effective tempo of the listed tracks (shown as the segment's
        # target BPM, and what _segment_pool ranks by) — and drop the
        # Settings run-type bounds below (confirmed live: a 166 BPM range on
        # a "work" segment with a 170-180 Settings range was clamped empty,
        # fell back to the full library and came out at 174). Pace Pro's
        # strict mode keeps its own pace-derived target.
        user_bpm_range = has_explicit_candidates and not strict_pace_tolerance and seg.bpm is not None
        if user_bpm_range:
            listed = library[library["Track URI"].isin(set(segment_candidate_uris[seg_idx]))]
            if not listed.empty:
                eff = listed["Tempo"].astype(float).map(_effective_run_tempo)
                seg.bpm = seg.raw_bpm = float(round(eff.median()))
                seg.sweet_bpm = None

        # seg_bpm_bounds computed here (moved ahead of its original spot
        # below `pool`) so the strict exact-BPM lock just below can respect
        # the same kind ceiling/floor its step-out search widens within —
        # see _exact_bpm_pool's own bpm_bounds param doc.
        if strict_pace_tolerance and seg.bpm:
            # Hard floor at target - tightest tolerance, no ceiling - same
            # rule lib/pace-analysis.ts's classifyPaceFit() judges "fits"
            # by. Replaces (not tightens) the kind-based bpm_overrides bounds.
            seg_bpm_bounds = (seg.bpm - BPM_TOLERANCES[0], None)
        elif seg.literal_bpm:
            # A literal, directly-typed BPM (AI Remix) has nothing to do
            # with the "work" kind's general Settings range (e.g. a
            # configured 170-180 "work" window) - that range is for
            # pace-derived cadence targets, not a specific number the user
            # just typed in. Clamping to it here was silently excluding
            # real matching tracks before the exact-BPM filter below ever
            # got to see them (confirmed: 55 real 168 BPM library tracks,
            # all excluded because "work" was configured for a different
            # range), making a perfectly satisfiable request come back
            # empty. No kind-based bounds at all - the exact-match filter
            # below is the only constraint that should apply.
            seg_bpm_bounds = (None, None)
        elif user_bpm_range:
            seg_bpm_bounds = (None, None)
        else:
            seg_bpm_bounds = _kind_bpm_bounds(seg.kind, bpm_overrides)

        # Every segment with a BPM target (every kind except strength, which
        # has none) is locked to ONE exact whole-number BPM for its entire
        # candidate pool before anything else runs — see _exact_bpm_pool's
        # own doc comment. This replaces (not stacks with) _segment_pool's
        # own tolerance-widening ladder for BPM specifically; energy/play-
        # count/artist-dedup widening inside _segment_pool is unaffected,
        # it just now operates on an already BPM-locked library.
        #
        # Locks to raw_bpm (the pace-derived target BEFORE the Settings
        # ceiling/floor clamp), not the clamped seg.bpm — otherwise every
        # segment whose real pace exceeds a kind's configured ceiling
        # collapses onto that exact same clamped number (confirmed live: a
        # "work" ceiling of 170 made literally every fast segment lock to
        # 170 regardless of its actual pace). The Settings ceiling/floor
        # still applies as a hard boundary (passed into _exact_bpm_pool's
        # own step-out search, AND still enforced again by _segment_pool's
        # bpm_bounds filter below) — it just no longer flattens the LOCK
        # target itself onto one shared clamped value.
        #
        # SKIPPED entirely when this segment has its own explicit candidate
        # list (segment_candidate_uris — Advanced AI DJ Mix's BPM-range
        # picker, or Pace Pro's pace-analysis "fits" list): that list is
        # itself the user's (or Pace Analysis's) own deliberate BPM choice,
        # and locking lib_for_seg to the PACE-derived target first — before
        # the candidate list even gets applied below — fought it outright
        # (confirmed live: Advanced Mix asked for 168-168 BPM on an 8:55/mi
        # easy segment, but the pace-derived lock had already collapsed
        # lib_for_seg to 170 BPM; intersecting 168-only candidates against a
        # 170-only library came back empty, which silently fell through to
        # "widen to the full library" — except that fallback reused the
        # already-170-locked lib_for_seg too, so the user's own 168 BPM
        # choice was never actually reachable, and the mix came out at 170
        # regardless of what was typed in). The explicit candidate list is
        # applied against the UNLOCKED library instead — see seg_allowed_uris
        # below — and is the only BPM constraint for this segment short of
        # the kind's own Settings ceiling/floor (seg_bpm_bounds, still
        # enforced via _segment_pool's bpm_bounds either way).
        lock_bpm = seg.raw_bpm or seg.bpm
        if lock_bpm and not has_explicit_candidates:
            lib_for_seg = _exact_bpm_pool(lib_for_seg, lock_bpm, bpm_bounds=seg_bpm_bounds)
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
        #
        # Moot once lib_for_seg was locked to one exact BPM above — guarded
        # by the SAME has_explicit_candidates check the lock itself used, so
        # a segment whose lock was skipped (explicit candidate list) still
        # gets the normal kind-based breadth floor, not the locked-pool-only
        # floor of 1. _segment_pool's own BPM tolerance ladder can't widen a
        # pool that's already a single BPM's worth of tracks, so demanding a
        # breadth floor from it (MAX_CANDIDATES, or even the old flat 20)
        # would fail segments whose exact-BPM pool has genuinely fewer
        # tracks than that but still easily covers the duration budget on
        # its own — exactly
        # backwards for a "strictly one BPM" ask, where the point is to
        # accept whatever that one BPM actually has, not demand artificial
        # breadth from it (confirmed: this only actually bites with play-
        # count tiering OFF — see _fills_budget's own `if max_plays is
        # None: len(pool) < min_pool` gate, which tiering normally skips
        # entirely). The duration-budget check in _fills_budget is still the
        # real gate either way.
        seg_min_pool = 1 if (seg.bpm and not has_explicit_candidates) else (MAX_CANDIDATES if seg.kind in WIDE_TOLERANCE_KINDS else 20)

        seg_allowed_uris = segment_candidate_uris[seg_idx] if segment_candidate_uris and seg_idx < len(segment_candidate_uris) else None
        pool = pd.DataFrame()
        if seg_allowed_uris is not None:
            # Preferred pool: restricted to exactly the caller-supplied URIs
            # (Pace Pro: whatever lib/pace-analysis.ts already classified as
            # "fits" this segment's pace from confirmed play history) - tried
            # first so the LLM only ever sees music already proven to work at
            # this pace. Only falls through to the full library below if this
            # can't fill the segment's own time budget on its own.
            restricted_lib = lib_for_seg[lib_for_seg["Track URI"].isin(set(seg_allowed_uris))]
            candidate_pool = _segment_pool(
                restricted_lib, seg, used, min_pool=seg_min_pool, budget_sec=budget, easy_bias_sec=easy_bias_sec,
                used_artists=used_artists, played=played, play_counts=play_counts, boosted=boosted or None,
                bpm_bounds=seg_bpm_bounds, avoid=avoid or None,
                sweet_bpm=seg.sweet_bpm,
            )
            if not candidate_pool.empty and candidate_pool["Duration (ms)"].sum() / 1000 >= budget:
                pool = candidate_pool
            elif not candidate_pool.empty:
                _log(f"'{seg.label}': restricted candidate list too small for the segment budget, widening to the full library")
        if pool.empty:
            pool = _segment_pool(
                lib_for_seg, seg, used, min_pool=seg_min_pool, budget_sec=budget, easy_bias_sec=easy_bias_sec,
                used_artists=used_artists, played=played, play_counts=play_counts, boosted=boosted or None,
                bpm_bounds=seg_bpm_bounds, avoid=avoid or None,
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
            # choose_setlist truncates its own view of the pool to
            # MAX_CANDIDATES (see its docstring - past ~150-200 candidates
            # local models start losing track of the numeric BPM data and
            # quality collapses) regardless of how large llm_pool is, so
            # mirror that same slice here for the log/progress line - this
            # used to just report len(llm_pool), which claimed the full
            # (potentially 300+) pool was "sent" when choose_setlist had
            # already silently capped what the model actually saw.
            visible_pool = llm_pool.head(MAX_CANDIDATES) if len(llm_pool) > MAX_CANDIDATES else llm_pool
            sent_count = len(visible_pool)
            pool_desc = f"{sent_count} of {len(llm_pool)} candidates" if sent_count < len(llm_pool) else f"{sent_count} candidates"
            sweet_str = f", sweet {seg.sweet_bpm:.0f}" if seg.sweet_bpm else ""
            # Play-count breakdown of what's actually sent to the LLM (not
            # the wider `pool`) - "0x9, 1x4" reads as 9 never-played tracks
            # and 4 one-play tracks made the candidate list, so a mix that
            # still picks played tracks can be told apart from one that had
            # no fresher choice to offer.
            plays_str = ""
            if play_counts:
                sent_counts = visible_pool["Track URI"].map(lambda u: play_counts.get(u, 0))
                plays_str = ", plays " + ", ".join(f"{c}x{n}" for c, n in sent_counts.value_counts().sort_index().items())
            _log(
                f"'{seg.label}': sending {pool_desc} to {model} "
                f"(target {target_bpm_str} BPM{sweet_str}, pool range "
                f"{pool['Tempo'].min():.0f}-{pool['Tempo'].max():.0f} BPM, count target {n_est}{plays_str})"
            )
            candidate_uris = visible_pool["Track URI"].tolist() if "Track URI" in visible_pool.columns else None
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

        chosen = _fit_duration_precise(ordered, budget, overshoot=is_last).copy()
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
