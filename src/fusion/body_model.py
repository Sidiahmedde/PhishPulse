"""BodyModel: TF-IDF + numeric features + Logistic Regression for phishing body detection.

Feature order (FEATURE_NAMES):
  0. urgent_keyword_count  – word-boundary regex occurrences of urgency terms
  1. url_count             – number of http/https URLs
  2. exclamation_count     – number of '!' characters
  3. uppercase_count       – number of uppercase letters
  4. length_chars          – character length of raw text
  5. length_words          – word count of raw text
  6. currency_token_count  – occurrences of $, usd, eur, gbp, bitcoin, etc.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex constants
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_QUOTED_LINE_RE = re.compile(r"^>+.*", re.MULTILINE)
# Matches common reply/forward headers and truncates everything after them.
_REPLY_HEADER_RE = re.compile(
    r"(?i)(?:[-_]{2,}\s*)?(?:original\s+message|forwarded\s+message"
    r"|on\s+.+\s+wrote\s*:|from\s*:\s*\S+@\S+).*",
    re.DOTALL,
)

URGENT_WORDS: List[str] = [
    "urgent", "action", "verify", "immediately", "password",
    "account", "suspended", "invoice", "payment", "confirm",
    "click", "login", "update", "security", "alert",
]
_URGENT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in URGENT_WORDS
]

_CURRENCY_RE = re.compile(
    r"\$|(?:\b(?:usd|eur|gbp|cad|aud|inr|bitcoin|btc|crypto)\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "urgent_keyword_count",
    "url_count",
    "exclamation_count",
    "uppercase_count",
    "length_chars",
    "length_words",
    "currency_token_count",
]

# Fallback thresholds: if thread_clean output is shorter than these, use light_clean.
_MIN_CLEAN_CHARS: int = 30
_MIN_CLEAN_WORDS: int = 5

# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------


def light_clean(text: str) -> str:
    """Strip HTML tags, normalize whitespace, and lowercase.

    Parameters
    ----------
    text : str
        Raw email body text.

    Returns
    -------
    str
        Cleaned, lowercased text.
    """
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()


def thread_clean(text: str) -> str:
    """Remove quoted reply blocks and reply-header sections, then light_clean.

    Strips lines beginning with ``>``, and truncates at common reply-header
    patterns such as "-----Original Message-----" or "On <date> wrote:".

    Parameters
    ----------
    text : str
        Raw email body text (may contain quoted threads).

    Returns
    -------
    str
        Cleaned text with quoted sections removed.
    """
    match = _REPLY_HEADER_RE.search(text)
    if match:
        text = text[: match.start()]
    text = _QUOTED_LINE_RE.sub("", text)
    return light_clean(text)


def clean_body(
    text: Union[str, float, None],
) -> Tuple[str, bool]:
    """Apply thread_clean with a safety fallback to light_clean.

    If the result of thread_clean is too short (fewer than
    :data:`_MIN_CLEAN_WORDS` words or :data:`_MIN_CLEAN_CHARS` characters),
    light_clean is used instead to avoid discarding useful content.

    Parameters
    ----------
    text : str or NaN/None
        Raw email body.

    Returns
    -------
    cleaned : str
        The cleaned body text.
    used_fallback : bool
        ``True`` when light_clean was used instead of thread_clean.
    """
    if pd.isna(text):
        return "", False
    raw = str(text)
    cleaned = thread_clean(raw)
    if len(cleaned) < _MIN_CLEAN_CHARS or len(cleaned.split()) < _MIN_CLEAN_WORDS:
        return light_clean(raw), True
    return cleaned, False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_body_features(raw_text: Union[str, float, None]) -> List[float]:
    """Extract numeric features from raw (uncleaned) email body text.

    Features are computed on the *raw* text so that signals such as uppercase
    letters and original URL count are preserved before casing normalisation.

    Feature order matches :data:`FEATURE_NAMES`:

    ==================== ================================================
    Feature              Description
    ==================== ================================================
    urgent_keyword_count Total regex-matched occurrences (word boundaries)
                         of :data:`URGENT_WORDS`.
    url_count            Number of http/https URLs.
    exclamation_count    Number of ``!`` characters.
    uppercase_count      Number of uppercase letters.
    length_chars         Character length.
    length_words         Word count.
    currency_token_count Occurrences of ``$``, usd, eur, gbp, bitcoin, etc.
    ==================== ================================================

    Parameters
    ----------
    raw_text : str or NaN/None
        Raw email body.

    Returns
    -------
    list of float
        Feature vector of length ``len(FEATURE_NAMES)``.
    """
    if pd.isna(raw_text):
        return [0.0] * len(FEATURE_NAMES)
    text = str(raw_text)

    urgent_count = sum(len(p.findall(text)) for p in _URGENT_PATTERNS)
    url_count = len(_URL_RE.findall(text))
    exclamation_count = text.count("!")
    uppercase_count = sum(1 for c in text if c.isupper())
    length_chars = len(text)
    length_words = len(text.split())
    currency_count = len(_CURRENCY_RE.findall(text))

    return [
        float(urgent_count),
        float(url_count),
        float(exclamation_count),
        float(uppercase_count),
        float(length_chars),
        float(length_words),
        float(currency_count),
    ]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class BodyConfig:
    """Hyperparameters for :class:`BodyModel`.

    Parameters
    ----------
    max_features : int
        TF-IDF vocabulary size cap.
    ngram_range : tuple of int
        Word n-gram range for TF-IDF (e.g. ``(1, 2)`` for unigrams + bigrams).
    min_df : int or float
        Minimum document frequency for TF-IDF terms.
    max_df : int or float
        Maximum document frequency for TF-IDF terms.
    C : float
        Inverse regularisation strength for Logistic Regression.
        Larger values → less regularisation.
    solver : str
        LR solver. ``"saga"`` is recommended for large sparse TF-IDF matrices.
    max_iter : int
        Maximum solver iterations.
    class_weight : str, dict, or None
        Class weight strategy. ``"balanced"`` compensates for imbalanced labels.
    random_state : int
        Random seed for reproducibility.
    calibrate : bool
        If ``True``, wrap LR with :class:`~sklearn.calibration.CalibratedClassifierCV`
        for better probability estimates (important for downstream fusion).
    calibration_method : str
        ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.
    calibration_cv : int
        Number of cross-validation folds used by the calibrator.
    """

    max_features: int = 50_000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: Union[int, float] = 2
    max_df: Union[int, float] = 1.0
    C: float = 1.0
    solver: str = "saga"
    max_iter: int = 5000
    class_weight: Union[str, Dict, None] = "balanced"
    random_state: int = 42
    calibrate: bool = False
    calibration_method: str = "sigmoid"
    calibration_cv: int = 3


# ---------------------------------------------------------------------------
# BodyModel
# ---------------------------------------------------------------------------


class BodyModel:
    """Phishing classifier for email body text.

    Combines TF-IDF word n-gram features with :data:`FEATURE_NAMES` handcrafted
    numeric features, trained with Logistic Regression (SAGA solver by default).

    The stable public API is:

    * :meth:`fit` – train on raw texts
    * :meth:`predict_proba` – probability of phishing (class 1)
    * :meth:`predict` – binary prediction with configurable threshold
    * :meth:`save` / :meth:`load` – persist and restore the fitted model

    Parameters
    ----------
    config : BodyConfig, optional
        Hyperparameter configuration. Defaults to :class:`BodyConfig` with
        production-suitable defaults.

    Examples
    --------
    >>> model = BodyModel()
    >>> model.fit(train_texts, y_train)
    >>> proba = model.predict_proba(test_texts)   # ndarray of shape (n,)
    >>> preds = model.predict(test_texts, threshold=0.5)
    >>> model.save("artifacts/body/body_model.joblib")
    >>> loaded = BodyModel.load("artifacts/body/body_model.joblib")
    """

    def __init__(self, config: Optional[BodyConfig] = None) -> None:
        self.config: BodyConfig = config or BodyConfig()
        self._vectorizer: TfidfVectorizer = self._build_vectorizer()
        self._classifier = self._build_classifier()
        # Populated after fit() with cleaning and vocabulary statistics.
        self.fit_stats_: dict = {}

    # ------------------------------------------------------------------
    # Internal construction helpers
    # ------------------------------------------------------------------

    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            stop_words="english",
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=True,
        )

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

    # ------------------------------------------------------------------
    # Internal preprocessing helpers
    # ------------------------------------------------------------------

    def _clean_texts(
        self, texts: List[str]
    ) -> Tuple[List[str], float, float]:
        """Clean texts and return (cleaned, empty_fraction, fallback_fraction)."""
        cleaned: List[str] = []
        fallback_count = 0
        empty_count = 0
        for t in texts:
            c, used_fallback = clean_body(t)
            cleaned.append(c)
            if used_fallback:
                fallback_count += 1
            if not c.strip():
                empty_count += 1
        n = max(len(texts), 1)
        return cleaned, empty_count / n, fallback_count / n

    def _compute_features(self, texts: List[str]) -> np.ndarray:
        return np.array([extract_body_features(t) for t in texts], dtype=float)

    def _combine(self, cleaned: List[str], custom: np.ndarray):
        tfidf = self._vectorizer.transform(cleaned)
        return hstack([tfidf, custom])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: Union["pd.Series", List[str]],
        y: Union["pd.Series", np.ndarray],
    ) -> "BodyModel":
        """Fit TF-IDF and classifier on raw email body texts.

        Cleaning statistics (empty_clean_fraction, fallback_fraction, vocab_size,
        dataset_size) are stored in :attr:`fit_stats_` after fitting.

        Parameters
        ----------
        texts : list-like of str
            Raw email body strings.
        y : array-like of int
            Binary labels (0 = ham, 1 = spam/phishing).

        Returns
        -------
        self
        """
        texts_list = list(texts)
        logger.info("Fitting BodyModel on %d samples.", len(texts_list))

        cleaned, empty_frac, fallback_frac = self._clean_texts(texts_list)
        logger.info(
            "Cleaning stats — empty: %.2f%%  fallback: %.2f%%",
            empty_frac * 100,
            fallback_frac * 100,
        )

        # Numeric features are computed from raw texts (before lowercasing/stripping).
        custom = self._compute_features(texts_list)

        X_tfidf = self._vectorizer.fit_transform(cleaned)
        logger.info("Vocabulary size: %d", len(self._vectorizer.vocabulary_))

        X = hstack([X_tfidf, custom])
        self._classifier.fit(X, np.asarray(y))

        self.fit_stats_ = {
            "dataset_size": len(texts_list),
            "empty_clean_fraction": round(empty_frac, 4),
            "fallback_fraction": round(fallback_frac, 4),
            "vocab_size": len(self._vectorizer.vocabulary_),
        }
        logger.info("fit complete — fit_stats_=%s", self.fit_stats_)
        return self

    def fit_preprocessed(
        self,
        cleaned_texts: Union["pd.Series", List[str]],
        custom_features: np.ndarray,
        y: Union["pd.Series", np.ndarray],
    ) -> "BodyModel":
        """Fit using pre-cleaned texts and pre-computed numeric features.

        Use this when a preprocessed CSV (produced by ``preprocess_body_data.py``)
        is available and you want to avoid redundant computation.

        Parameters
        ----------
        cleaned_texts : list-like of str
            Already-cleaned body texts (e.g. the ``body_clean`` column).
        custom_features : np.ndarray of shape (n, len(FEATURE_NAMES))
            Pre-computed numeric feature matrix.
        y : array-like of int
            Binary labels.

        Returns
        -------
        self
        """
        cleaned_list = list(cleaned_texts)
        logger.info(
            "Fitting BodyModel (preprocessed path) on %d samples.", len(cleaned_list)
        )
        X_tfidf = self._vectorizer.fit_transform(cleaned_list)
        logger.info("Vocabulary size: %d", len(self._vectorizer.vocabulary_))
        X = hstack([X_tfidf, custom_features])
        self._classifier.fit(X, np.asarray(y))
        self.fit_stats_ = {
            "dataset_size": len(cleaned_list),
            "vocab_size": len(self._vectorizer.vocabulary_),
            # Cleaning stats not available on the preprocessed path;
            # see preprocess_body_data.py output for those values.
            "empty_clean_fraction": None,
            "fallback_fraction": None,
        }
        return self

    def predict_proba(
        self,
        texts: Union["pd.Series", List[str]],
    ) -> np.ndarray:
        """Return probability of class 1 (phishing) for each input text.

        Parameters
        ----------
        texts : list-like of str
            Raw email body strings.

        Returns
        -------
        proba : np.ndarray of shape (n,)
            Probability scores in ``[0, 1]``.
        """
        texts_list = list(texts)
        cleaned, _, _ = self._clean_texts(texts_list)
        custom = self._compute_features(texts_list)
        X = self._combine(cleaned, custom)
        return self._classifier.predict_proba(X)[:, 1]

    def predict(
        self,
        texts: Union["pd.Series", List[str]],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary labels using a probability threshold.

        Parameters
        ----------
        texts : list-like of str
            Raw email body strings.
        threshold : float
            Decision boundary. Samples with ``predict_proba >= threshold``
            are classified as 1 (phishing). Default: ``0.5``.

        Returns
        -------
        labels : np.ndarray of shape (n,) with values in ``{0, 1}``
        """
        return (self.predict_proba(texts) >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Persist the fitted model to disk using joblib.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``artifacts/body/body_model.joblib``).
        """
        payload = {
            "config": self.config,
            "vectorizer": self._vectorizer,
            "classifier": self._classifier,
            "fit_stats": self.fit_stats_,
        }
        joblib.dump(payload, path)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "BodyModel":
        """Load a persisted BodyModel from disk.

        Parameters
        ----------
        path : str
            Path to a ``.joblib`` file produced by :meth:`save`.

        Returns
        -------
        BodyModel
            Fully restored model ready for inference.
        """
        payload = joblib.load(path)
        config = payload["config"]
        if not isinstance(config, BodyConfig):
            # Legacy compatibility: config saved as plain dict.
            config = BodyConfig(**config)
        instance = cls(config=config)
        instance._vectorizer = payload["vectorizer"]
        instance._classifier = payload["classifier"]
        instance.fit_stats_ = payload.get("fit_stats", {})
        logger.info("Model loaded ← %s", path)
        return instance
