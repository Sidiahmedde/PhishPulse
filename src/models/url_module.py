"""url_module: adapter for super_module integration.

Interface:
    from url_module import url_module
    score = url_module(email["urls"])  # float in [0, 1]
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_URL_DIR = _ROOT / "url_model"

if str(_URL_DIR) not in sys.path:
    sys.path.insert(0, str(_URL_DIR))


def url_module(url_text: str) -> float:
    """Return phishing probability in [0, 1] for the first URL-like input."""
    if not url_text or not str(url_text).strip():
        return 0.5

    try:
        from predict_url_model import predict_url_module
        result = predict_url_module(str(url_text))
        return float(result["phishing_probability"])
    except Exception:
        return 0.5
