"""sender_module: adapter for super_module integration.

Interface:
    from sender_module import sender_module
    score = sender_module(email["sender"])  # float in [0, 1]
"""

from __future__ import annotations

from pathlib import Path

import joblib

_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE / "sender_model_super_v2.pkl"
_VECTORIZER_PATH = _HERE / "sender_vectorizer_super_v2.pkl"

_model = None
_vectorizer = None


def sender_module(sender_text: str) -> float:
    """Return phishing probability in [0, 1] for a sender string."""
    global _model, _vectorizer

    if not sender_text or not str(sender_text).strip():
        return 0.5

    if _model is None or _vectorizer is None:
        try:
            _model = joblib.load(_MODEL_PATH)
            _vectorizer = joblib.load(_VECTORIZER_PATH)
        except Exception:
            return 0.5

    try:
        X = _vectorizer.transform([str(sender_text)])
        return float(_model.predict_proba(X)[0][1])
    except Exception:
        return 0.5
