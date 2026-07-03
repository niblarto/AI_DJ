"""Prompt -> setlist selection.

Two LLM calls, mirroring the classic local-DJ pipeline:
1. Extract hard constraints (BPM range, energy range, track count) from the
   user's prompt so we can pre-filter the library with bpm_matcher.
2. Hand the filtered candidate pool to the model to pick and order tracks.

An optional post-pass reorders the model's picks with bpm_matcher's weighted
distance (greedy nearest-neighbour chain) for smoother transitions.
"""

import sys

import numpy as np
import pandas as pd

from bpm_matcher.match import bpm_filter, cross_distance_matrix

from .llm import chat_json

MAX_CANDIDATES = 120

_CONSTRAINTS_SYSTEM = """\
You extract DJ-set constraints from a request. Reply with ONLY a JSON object:
{
  "bpm_min": number or null,
  "bpm_max": number or null,
  "energy_min": number 0-1 or null,
  "energy_max": number 0-1 or null,
  "count": integer or null
}
Use null when the request doesn't imply a value. Genre/mood words alone do not
imply BPM unless the genre strongly does (e.g. drum & bass ~170-180, house
~120-128, downtempo/chill <100). "count" is how many tracks were asked for."""

_SETLIST_SYSTEM = """\
You are a DJ building a setlist. You get a request and a numbered list of
candidate tracks with BPM, Camelot key, energy, danceability and valence
(0-1). Pick tracks that fit the request and order them as a coherent set:
smooth BPM progression, harmonically compatible keys where possible
(same/adjacent Camelot numbers), and an energy arc that suits the request.
Reply with ONLY a JSON object holding the track NUMBERS in play order, e.g.:
{"setlist": [17, 4, 62, 31], "reasoning": "one short paragraph"}
Do not repeat the track details in your reply - numbers only. Use each track
number at most once. Pick exactly the requested amount."""


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def extract_constraints(prompt: str, model: str) -> dict:
    raw = chat_json(_CONSTRAINTS_SYSTEM, prompt, model=model)

    def num(key, lo, hi):
        v = raw.get(key)
        if isinstance(v, (int, float)) and lo <= v <= hi:
            return float(v)
        return None

    count = raw.get("count")
    return {
        "bpm_min": num("bpm_min", 30, 300),
        "bpm_max": num("bpm_max", 30, 300),
        "energy_min": num("energy_min", 0, 1),
        "energy_max": num("energy_max", 0, 1),
        "count": int(count) if isinstance(count, (int, float)) and count > 0 else None,
    }


def filter_candidates(
    df: pd.DataFrame,
    constraints: dict,
    count: int,
    max_candidates: int = MAX_CANDIDATES,
) -> pd.DataFrame:
    """Narrow the library to a pool the model can handle.

    Models tend to extract over-literal constraints ("around 125 BPM" ->
    exactly 125, "high energy" -> >= 0.9), so narrow BPM targets get a
    floor tolerance and both filters are progressively relaxed until the
    pool is comfortably larger than the requested set.
    """
    bpm_lo, bpm_hi = constraints["bpm_min"], constraints["bpm_max"]
    if bpm_lo or bpm_hi:
        bpm_lo = bpm_lo or max(30.0, bpm_hi - 20)
        bpm_hi = bpm_hi or bpm_lo + 20
    target = (bpm_lo + bpm_hi) / 2 if bpm_lo else None
    base_tol = max((bpm_hi - bpm_lo) / 2, 6.0) if bpm_lo else None

    e_lo = constraints["energy_min"] or 0.0
    e_hi = constraints["energy_max"] or 1.0

    min_pool = min(max(3 * count, 30), len(df))
    pool = df
    for bpm_pad, energy_pad in ((0, 0.0), (4, 0.1), (8, 0.2), (None, None)):
        if bpm_pad is None:  # last resort: BPM filter only, energy dropped
            pool = bpm_filter(df, target, tolerance=base_tol + 12) if target else df
        else:
            pool = bpm_filter(df, target, tolerance=base_tol + bpm_pad) if target else df
            pool = pool[
                (pool["Energy"] >= max(e_lo - energy_pad, 0))
                & (pool["Energy"] <= min(e_hi + energy_pad, 1))
            ]
        if len(pool) >= min_pool:
            break

    if len(pool) > max_candidates:
        if "bpm_diff" in pool.columns:
            pool = pool.nsmallest(max_candidates, "bpm_diff")
        else:
            pool = pool.sample(max_candidates, random_state=0)
    return pool.reset_index(drop=True)


def _format_pool(pool: pd.DataFrame) -> str:
    lines = []
    for i, row in pool.iterrows():
        lines.append(
            f"{i}. {row['Track Name']} — {row['Artist Name(s)']} | "
            f"{row['Tempo']:.0f} BPM | {row['Camelot'] or '?'} | "
            f"energy {row['Energy']:.2f} | dance {row['Danceability']:.2f} | "
            f"mood {row['Valence']:.2f}"
        )
    return "\n".join(lines)


def _norm_title(s: str) -> str:
    s = str(s).lower().strip()
    # Model replies often quote "Title — Artist"; keep the title part, and
    # strip version suffixes the same way bpm_matcher does.
    for sep in (" — ", " - ", " ("):
        if sep in s:
            s = s.split(sep)[0].strip()
    return s


def _parse_picks(raw: dict, pool: pd.DataFrame) -> list[int]:
    """Extract pool indices from the model's JSON, however it shaped it.

    Handles the intended {"setlist": [ints]} as well as common 7B failure
    modes: a differently-named key, or a list of track names/objects instead
    of numbers (mapped back to the pool by normalized title).
    """
    items = None
    for key in ("setlist", "tracks", "track", "playlist", "order"):
        if isinstance(raw.get(key), list):
            items = raw[key]
            break
    if items is None:
        items = next((v for v in raw.values() if isinstance(v, list)), [])

    title_to_idx: dict[str, int] = {}
    for i, t in enumerate(pool["Track Name"]):
        title_to_idx.setdefault(_norm_title(t), i)

    picks, seen = [], set()
    for item in items:
        idx = None
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            idx = int(item)
        elif isinstance(item, str):
            idx = title_to_idx.get(_norm_title(item))
        elif isinstance(item, dict):
            name = item.get("name") or item.get("title") or item.get("track")
            number = item.get("number", item.get("index"))
            if isinstance(number, (int, float)):
                idx = int(number)
            elif name:
                idx = title_to_idx.get(_norm_title(name))
        if idx is not None and 0 <= idx < len(pool) and idx not in seen:
            seen.add(idx)
            picks.append(idx)
    return picks


def choose_setlist(prompt: str, pool: pd.DataFrame, count: int, model: str) -> tuple[pd.DataFrame, str]:
    """Ask the model to pick and order `count` tracks from the pool."""
    user = (
        f"Request: {prompt}\n"
        f"Pick exactly {count} tracks.\n\n"
        f"Candidates:\n{_format_pool(pool)}"
    )
    raw = chat_json(_SETLIST_SYSTEM, user, model=model, temperature=0.4)
    picks = _parse_picks(raw, pool)

    if not picks:
        _log("Model reply had no usable tracks; retrying once...")
        retry = user + '\n\nIMPORTANT: reply as {"setlist": [numbers], "reasoning": "..."} - candidate NUMBERS only.'
        raw = chat_json(_SETLIST_SYSTEM, retry, model=model, temperature=0.2)
        picks = _parse_picks(raw, pool)
    if not picks:
        raise ValueError(f"Model returned no usable track picks: {raw}")
    if len(picks) < count:
        _log(f"Model picked {len(picks)}/{count} usable tracks; continuing with those.")

    setlist = pool.iloc[picks[:count]].reset_index(drop=True)
    return setlist, str(raw.get("reasoning", ""))


def smooth_order(setlist: pd.DataFrame) -> pd.DataFrame:
    """Reorder by greedy nearest-neighbour chain over bpm_matcher's weighted
    distance, keeping the model's opening track."""
    if len(setlist) < 3:
        return setlist
    dist = cross_distance_matrix(setlist, setlist)
    np.fill_diagonal(dist, np.inf)

    order = [0]
    remaining = set(range(1, len(setlist)))
    while remaining:
        nxt = min(remaining, key=lambda j: dist[order[-1], j])
        remaining.remove(nxt)
        order.append(nxt)
    return setlist.iloc[order].reset_index(drop=True)


def build_setlist(
    prompt: str,
    library: pd.DataFrame,
    n: int | None = None,
    model: str = None,
    smooth: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Full pipeline: constraints -> filter -> LLM selection -> optional smoothing."""
    from .llm import DEFAULT_MODEL

    model = model or DEFAULT_MODEL
    constraints = extract_constraints(prompt, model)
    _log(f"Constraints: {constraints}")

    count = n or constraints["count"] or 15
    pool = filter_candidates(library, constraints, count)
    if pool.empty:
        raise ValueError("No tracks in the library match the extracted constraints.")
    _log(f"Candidate pool: {len(pool)} tracks (library: {len(library)})")

    count = min(count, len(pool))
    setlist, reasoning = choose_setlist(prompt, pool, count, model)

    if smooth:
        setlist = smooth_order(setlist)
    return setlist, reasoning
