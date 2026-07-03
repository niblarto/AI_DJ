"""AI DJ: natural-language prompt -> ordered setlist -> M3U for Mixxx.

Feature layer is bpm_matcher (Exportify CSV + Camelot/BPM matching) instead of
the Essentia analysis a typical local-DJ stack uses; the LLM layer is a local
Ollama model (Qwen 2.5 7B by default).
"""

import os
import sys

# bpm_matcher lives in a sibling repo, not on PyPI. Allow an env override for
# other machines; default to its location on this one.
_BPM_REPO = os.environ.get("AI_BPM_PATH", r"E:\Code\AI_BPM")

try:
    import bpm_matcher  # noqa: F401
except ImportError:
    if os.path.isdir(_BPM_REPO):
        sys.path.insert(0, _BPM_REPO)
    import bpm_matcher  # noqa: F401
