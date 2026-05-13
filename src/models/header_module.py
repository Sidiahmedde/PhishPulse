"""header_module: adapter for super_module integration.

Interface:
    from header_module import header_module
    score = header_module(email["headers"])  # float in [0, 1]

email["headers"] should be the raw header block string (everything before
the blank line separating headers from body in a .eml file).
"""

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _ROOT / "dataset/artifacts/header/header_model.joblib"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_model = None


def header_module(raw_headers: str) -> float:
    """Return phishing probability in [0, 1] from raw email header text."""
    global _model

    if not raw_headers or not str(raw_headers).strip():
        return 0.5

    if _model is None:
        try:
            from src.fusion.header_model import HeaderModel
            _model = HeaderModel.load(str(_MODEL_PATH))
        except Exception as exc:
            logger.warning("header_module: model unavailable (%s) – returning 0.5", exc)
            return 0.5

    try:
        from src.fusion.header_model import extract_features_from_eml_bytes
        raw = str(raw_headers).encode("utf-8", errors="replace")
        features = extract_features_from_eml_bytes(raw)
        X = np.array([features], dtype=float)
        return float(_model.predict_proba(X)[0])
    except Exception as exc:
        logger.warning("header_module: predict_proba failed (%s) – returning 0.5", exc)
        return 0.5
