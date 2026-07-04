"""M3U output for Mixxx (and anything else that reads extended M3U)."""

import pandas as pd


def write_m3u(setlist: pd.DataFrame, out_path: str) -> tuple[int, int]:
    """Write an extended M3U; returns (resolved, missing) counts.

    Unresolved tracks are emitted as comments so the file stays valid and the
    gaps stay visible when opened in a text editor.
    """
    resolved = missing = 0
    lines = ["#EXTM3U"]
    for _, row in setlist.iterrows():
        seconds = int(row["Duration (ms)"] / 1000) if "Duration (ms)" in row and pd.notna(row.get("Duration (ms)")) else -1
        title = f"{row['Artist Name(s)']} - {row['Track Name']}"
        location = row.get("Location")
        # NaN (pandas' None) is truthy - only a real string is a usable path.
        if isinstance(location, str) and location:
            lines.append(f"#EXTINF:{seconds},{title}")
            lines.append(str(location))
            resolved += 1
        else:
            url = row.get("Spotify URL") or _uri_to_url(row.get("Track URI"))
            lines.append(f"# MISSING: {title}" + (f" | {url}" if url else ""))
            missing += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return resolved, missing


def _uri_to_url(uri) -> str | None:
    if isinstance(uri, str) and uri.startswith("spotify:track:"):
        return f"https://open.spotify.com/track/{uri.rsplit(':', 1)[1]}"
    return None
