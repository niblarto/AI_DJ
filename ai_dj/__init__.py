"""AI DJ: natural-language prompt -> ordered setlist -> M3U for Mixxx.

Feature layer is bpm_matcher (Exportify CSV + Camelot/BPM matching) instead of
the Essentia analysis a typical local-DJ stack uses; the LLM layer is a local
Ollama model (Qwen 2.5 7B by default).
"""

import os
import sys

# Secrets and machine-local settings live in a git-ignored .env.local at the
# repo root (KEY=value lines). Real environment variables take precedence.
_ENV_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.local")
if os.path.isfile(_ENV_LOCAL):
    with open(_ENV_LOCAL, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# bpm_matcher lives in a sibling repo, not on PyPI. Allow an env override for
# other machines; default to its location on this one.
_BPM_REPO = os.environ.get("AI_BPM_PATH", r"E:\Code\AI_BPM")

try:
    import bpm_matcher  # noqa: F401
except ImportError:
    if os.path.isdir(_BPM_REPO):
        sys.path.insert(0, _BPM_REPO)
    import bpm_matcher  # noqa: F401
