"""body_module: adapter for super_module integration.

Interface:
    from body_module import body_module
    score = body_module(email["body"])  # float in [0, 1]
"""

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _ROOT / "dataset/artifacts/body/body_model.joblib"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_model = None


def body_module(body_text: str) -> float:
    """Return phishing probability in [0, 1] for an email body string."""
    global _model

    if not body_text or not str(body_text).strip():
        return 0.5

    if _model is None:
        try:
            from src.fusion.body_model import BodyModel
            _model = BodyModel.load(str(_MODEL_PATH))
        except Exception as exc:
            logger.warning("body_module: model unavailable (%s) – returning 0.5", exc)
            return 0.5

    try:
        return float(_model.predict_proba([str(body_text)])[0])
    except Exception as exc:
        logger.warning("body_module: predict_proba failed (%s) – returning 0.5", exc)
        return 0.5
