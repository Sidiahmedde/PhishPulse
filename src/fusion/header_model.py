"""HeaderModel: structured header features + Logistic Regression for phishing detection.

Feature order (FEATURE_NAMES):
  0.  multiple_from              – more than one From: header present (multiple-From attack)
  1.  from_encoded               – From value contains RFC 2047 encoded-word (=?...?=)
  2.  from_return_path_match     – From domain matches Return-Path domain (1=match)
  3.  mail_from_empty            – Return-Path is '<>' or absent
  4.  has_reply_to_diff_domain   – Reply-To present and its domain differs from From domain
  5.  from_mid_domain_match      – Message-ID domain matches From domain (1=match)
  6.  x_mailer_present           – X-Mailer header present (bulk-sender fingerprint)
  7.  received_hop_count         – number of Received: headers
  8.  duplicate_header_count     – total duplicate header occurrences
  9.  has_x_spam                 – X-Spam-Status header present
  10. sender_is_freemail         – From domain is a major free-mail provider
  11. from_display_name_has_domain – display name part of From contains a '@' or domain
  12. subject_has_encoding       – Subject contains RFC 2047 encoded-word
"""

import logging
import re
from dataclasses import dataclass
from email import message_from_bytes, message_from_string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex constants
# ---------------------------------------------------------------------------

_DOMAIN_RE = re.compile(r"@([\w.\-]+)", re.IGNORECASE)
_ENCODED_WORD_RE = re.compile(r"=\?[^?]+\?[BbQq]\?[^?]*\?=")
_FREEMAIL_DOMAINS = frozenset(
    [
        "gmail.com", "yahoo.com", "yahoo.co.uk", "hotmail.com", "hotmail.co.uk",
        "outlook.com", "live.com", "aol.com", "icloud.com", "protonmail.com",
        "mail.com", "yandex.com", "gmx.com", "gmx.net", "zoho.com",
    ]
)
_DISPLAY_DOMAIN_RE = re.compile(r"[\w.\-]+\.\w{2,}", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "multiple_from",
    "from_encoded",
    "from_return_path_match",
    "mail_from_empty",
    "has_reply_to_diff_domain",
    "from_mid_domain_match",
    "x_mailer_present",
    "received_hop_count",
    "duplicate_header_count",
    "has_x_spam",
    "sender_is_freemail",
    "from_display_name_has_domain",
    "subject_has_encoding",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_domain(addr: str) -> str:
    """Return the lowercase domain from an email address string, or ''."""
    if not addr:
        return ""
    match = _DOMAIN_RE.search(addr)
    return match.group(1).lower() if match else ""


def _extract_display_name(addr: str) -> str:
    """Return the display-name portion of an RFC 5322 address, or ''."""
    addr = addr.strip()
    if addr.startswith('"'):
        end = addr.find('"', 1)
        return addr[1:end] if end != -1 else ""
    if "<" in addr:
        return addr[: addr.index("<")].strip()
    return ""


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_header_features(
    *,
    from_field: str = "",
    return_path: str = "",
    reply_to: str = "",
    message_id: str = "",
    subject: str = "",
    all_header_keys: Optional[List[str]] = None,
    received_count: int = 0,
    x_mailer: str = "",
    x_spam_status: str = "",
) -> List[float]:
    """Extract numeric header features from parsed email header values.

    All arguments are optional strings; missing headers should be passed as
    empty strings (``""``).

    Feature order matches :data:`FEATURE_NAMES`.

    Parameters
    ----------
    from_field : str
        Raw value of the From: header (may contain display name).
    return_path : str
        Raw value of the Return-Path: header.
    reply_to : str
        Raw value of the Reply-To: header.
    message_id : str
        Raw value of the Message-ID: header.
    subject : str
        Raw value of the Subject: header.
    all_header_keys : list of str, optional
        All header field names (duplicates included) from the parsed message.
        Used to detect multiple From: headers and duplicate headers.
    received_count : int
        Number of Received: headers in the message.
    x_mailer : str
        Raw value of the X-Mailer: header (empty string if absent).
    x_spam_status : str
        Raw value of the X-Spam-Status: header (empty string if absent).

    Returns
    -------
    list of float
        Feature vector of length ``len(FEATURE_NAMES)``.
    """
    keys = all_header_keys or []

    from_domain = _extract_domain(from_field)
    rp_domain = _extract_domain(return_path)
    rt_domain = _extract_domain(reply_to)
    mid_domain = _extract_domain(message_id)
    display_name = _extract_display_name(from_field)

    # 0. multiple_from
    multiple_from = float(keys.count("From") > 1)

    # 1. from_encoded
    from_encoded = float(bool(_ENCODED_WORD_RE.search(from_field)))

    # 2. from_return_path_match
    if from_domain and rp_domain:
        from_return_path_match = float(from_domain == rp_domain)
    else:
        from_return_path_match = 0.0  # unknown / missing = suspicious default

    # 3. mail_from_empty — only flag when Return-Path is explicitly present but empty.
    # An absent header (empty string passed by caller) is treated as unknown → 0.
    if return_path == "":
        mail_from_empty = 0.0  # unknown / header not present
    else:
        rp_stripped = return_path.strip().strip("<>").strip()
        mail_from_empty = float(rp_stripped == "")

    # 4. has_reply_to_diff_domain
    if reply_to and rt_domain:
        has_reply_to_diff_domain = float(rt_domain != from_domain)
    else:
        has_reply_to_diff_domain = 0.0

    # 5. from_mid_domain_match
    if from_domain and mid_domain:
        from_mid_domain_match = float(from_domain == mid_domain)
    else:
        from_mid_domain_match = 0.0

    # 6. x_mailer_present
    x_mailer_present = float(bool(x_mailer))

    # 7. received_hop_count
    received_hop_count = float(received_count)

    # 8. duplicate_header_count
    if keys:
        from collections import Counter
        counts = Counter(keys)
        duplicate_header_count = float(sum(v - 1 for v in counts.values() if v > 1))
    else:
        duplicate_header_count = 0.0

    # 9. has_x_spam
    has_x_spam = float(bool(x_spam_status))

    # 10. sender_is_freemail
    sender_is_freemail = float(from_domain in _FREEMAIL_DOMAINS)

    # 11. from_display_name_has_domain
    dn_has_domain = float(
        "@" in display_name or bool(_DISPLAY_DOMAIN_RE.search(display_name))
    )

    # 12. subject_has_encoding
    subject_has_encoding = float(bool(_ENCODED_WORD_RE.search(subject)))

    return [
        multiple_from,
        from_encoded,
        from_return_path_match,
        mail_from_empty,
        has_reply_to_diff_domain,
        from_mid_domain_match,
        x_mailer_present,
        received_hop_count,
        duplicate_header_count,
        has_x_spam,
        sender_is_freemail,
        dn_has_domain,
        subject_has_encoding,
    ]


def extract_features_from_eml_bytes(raw: bytes) -> List[float]:
    """Parse a raw .eml byte string and return header features.

    Parameters
    ----------
    raw : bytes
        Raw bytes of the .eml file.

    Returns
    -------
    list of float
        Feature vector of length ``len(FEATURE_NAMES)``.
    """
    try:
        msg = message_from_bytes(raw)
    except Exception:
        return [0.0] * len(FEATURE_NAMES)

    # Ensure keys are plain strings (legacy parser may return str-like objects)
    keys: List[str] = [str(k) for k in msg.keys()]
    return extract_header_features(
        from_field=str(msg.get("From", "")),
        return_path=str(msg.get("Return-Path", "")),
        reply_to=str(msg.get("Reply-To", "")),
        message_id=str(msg.get("Message-ID", "")),
        subject=str(msg.get("Subject", "")),
        all_header_keys=keys,
        received_count=len(msg.get_all("Received") or []),
        x_mailer=str(msg.get("X-Mailer", "")),
        x_spam_status=str(msg.get("X-Spam-Status", "")),
    )


def extract_features_from_eml_path(path: Union[str, Path]) -> List[float]:
    """Load a .eml file from disk and return header features.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the .eml file.

    Returns
    -------
    list of float
        Feature vector of length ``len(FEATURE_NAMES)``.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return extract_features_from_eml_bytes(raw)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return [0.0] * len(FEATURE_NAMES)


def extract_features_from_csv_row(
    sender: str = "",
    subject: str = "",
) -> List[float]:
    """Extract header features from a pre-processed CSV row (e.g. CEAS_08).

    For rows sourced from flat CSVs (where raw headers are unavailable), only
    sender and subject fields are available.  Attack indicators that require
    raw headers (multiple_from, Reply-To mismatch, etc.) are set to 0.
    Positive-match features (from_return_path_match, from_mid_domain_match)
    are set to 1, reflecting the overwhelming likelihood that legitimate emails
    have consistent From/Return-Path/Message-ID domains.  received_hop_count
    is set to 2, a typical value for direct legitimate delivery.

    Parameters
    ----------
    sender : str
        Value of the sender/From field (may include display name).
    subject : str
        Value of the subject field.

    Returns
    -------
    list of float
        Feature vector of length ``len(FEATURE_NAMES)``.
    """
    from_domain = _extract_domain(sender)
    display_name = _extract_display_name(sender)

    # Attack indicators – absent in legitimate email → 0
    multiple_from = 0.0
    from_encoded = float(bool(_ENCODED_WORD_RE.search(sender)))
    mail_from_empty = 0.0
    has_reply_to_diff_domain = 0.0
    x_mailer_present = 0.0
    duplicate_header_count = 0.0
    has_x_spam = 0.0

    # Positive-match features – assumed true for legitimate email
    from_return_path_match = 1.0   # From and Return-Path align in legit email
    from_mid_domain_match = 1.0   # Message-ID domain aligns with From in legit email

    # Routing
    received_hop_count = 2.0  # typical for legitimate direct delivery

    # Sender classification – computable from sender string
    sender_is_freemail = float(from_domain in _FREEMAIL_DOMAINS)
    dn_has_domain = float(
        "@" in display_name or bool(_DISPLAY_DOMAIN_RE.search(display_name))
    )

    # Subject encoding – computable from subject string
    subject_has_encoding = float(bool(_ENCODED_WORD_RE.search(subject)))

    return [
        multiple_from,
        from_encoded,
        from_return_path_match,
        mail_from_empty,
        has_reply_to_diff_domain,
        from_mid_domain_match,
        x_mailer_present,
        received_hop_count,
        duplicate_header_count,
        has_x_spam,
        sender_is_freemail,
        dn_has_domain,
        subject_has_encoding,
    ]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class HeaderConfig:
    """Hyperparameters for :class:`HeaderModel`.

    Parameters
    ----------
    C : float
        Inverse regularisation strength for Logistic Regression.
    solver : str
        LR solver. ``"lbfgs"`` is suitable for small dense feature matrices.
    max_iter : int
        Maximum solver iterations.
    class_weight : str, dict, or None
        Class weight strategy. ``"balanced"`` compensates for label imbalance.
    scale_features : bool
        If ``True``, apply :class:`~sklearn.preprocessing.StandardScaler`
        before training (recommended for logistic regression on mixed scales).
    random_state : int
        Random seed.
    calibrate : bool
        If ``True``, wrap LR with :class:`~sklearn.calibration.CalibratedClassifierCV`
        for better probability estimates needed by the fusion meta-model.
    calibration_method : str
        ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.
    calibration_cv : int
        CV folds for calibration.
    """

    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Union[str, Dict, None] = "balanced"
    scale_features: bool = True
    random_state: int = 42
    calibrate: bool = True
    calibration_method: str = "sigmoid"
    calibration_cv: int = 3


# ---------------------------------------------------------------------------
# HeaderModel
# ---------------------------------------------------------------------------


class HeaderModel:
    """Phishing classifier based on email header structural features.

    Trains a Logistic Regression on :data:`FEATURE_NAMES` numeric features
    extracted from raw .eml headers.  Optionally wraps the classifier with
    ``CalibratedClassifierCV`` for well-calibrated probability outputs required
    by the downstream fusion meta-model.

    Public API
    ----------
    * :meth:`fit` – train on a feature matrix
    * :meth:`predict_proba` – probability of phishing (class 1)
    * :meth:`predict` – binary prediction with configurable threshold
    * :meth:`save` / :meth:`load` – persist and restore the fitted model

    Examples
    --------
    >>> model = HeaderModel()
    >>> model.fit(X_train, y_train)
    >>> proba = model.predict_proba(X_test)
    >>> preds = model.predict(X_test, threshold=0.5)
    >>> model.save("artifacts/header/header_model.joblib")
    >>> loaded = HeaderModel.load("artifacts/header/header_model.joblib")
    """

    def __init__(self, config: Optional[HeaderConfig] = None) -> None:
        self.config: HeaderConfig = config or HeaderConfig()
        self._scaler: Optional[StandardScaler] = (
            StandardScaler() if self.config.scale_features else None
        )
        self._classifier = self._build_classifier()
        self.fit_stats_: dict = {}

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _build_classifier(self):
        lr = LogisticRegression(
            C=self.config.C,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )
        if self.config.calibrate:
            return CalibratedClassifierCV(
                lr,
                method=self.config.calibration_method,
                cv=self.config.calibration_cv,
            )
        return lr

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self._scaler is None:
            return X
        if fit:
            return self._scaler.fit_transform(X)
        return self._scaler.transform(X)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: Union["pd.Series", np.ndarray],
    ) -> "HeaderModel":
        """Fit the classifier on pre-computed header feature matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n, len(FEATURE_NAMES))
            Header feature matrix.
        y : array-like of int
            Binary labels (0 = legitimate, 1 = phishing).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        logger.info("Fitting HeaderModel on %d samples, %d features.", *X.shape)
        X_scaled = self._scale(X, fit=True)
        self._classifier.fit(X_scaled, y_arr)
        self.fit_stats_ = {
            "dataset_size": len(y_arr),
            "class_balance": {
                str(k): int(v)
                for k, v in zip(*np.unique(y_arr, return_counts=True))
            },
        }
        logger.info("fit complete — fit_stats_=%s", self.fit_stats_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return phishing probability for each row in X.

        Parameters
        ----------
        X : np.ndarray of shape (n, len(FEATURE_NAMES))

        Returns
        -------
        proba : np.ndarray of shape (n,)
            Values in ``[0, 1]``.
        """
        X = np.asarray(X, dtype=float)
        X_scaled = self._scale(X, fit=False)
        return self._classifier.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction with configurable threshold.

        Parameters
        ----------
        X : np.ndarray of shape (n, len(FEATURE_NAMES))
        threshold : float
            Samples with ``predict_proba >= threshold`` → label 1.

        Returns
        -------
        labels : np.ndarray of shape (n,) with values in ``{0, 1}``
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Persist the fitted model to disk using joblib."""
        payload = {
            "config": self.config,
            "scaler": self._scaler,
            "classifier": self._classifier,
            "fit_stats": self.fit_stats_,
        }
        joblib.dump(payload, path)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "HeaderModel":
        """Load a persisted HeaderModel from disk."""
        payload = joblib.load(path)
        config = payload["config"]
        if not isinstance(config, HeaderConfig):
            config = HeaderConfig(**config)
        instance = cls(config=config)
        instance._scaler = payload["scaler"]
        instance._classifier = payload["classifier"]
        instance.fit_stats_ = payload.get("fit_stats", {})
        logger.info("Model loaded ← %s", path)
        return instance
