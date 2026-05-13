"""Tests for HeaderModel: feature extraction, model behaviour, and persistence.

All tests are offline (no network, no disk I/O beyond temp files).
Run with:  pytest tests/test_header_model.py -v
"""

import os
import tempfile

import numpy as np
import pytest

from src.fusion.header_model import (
    FEATURE_NAMES,
    HeaderConfig,
    HeaderModel,
    _extract_domain,
    _extract_display_name,
    extract_features_from_csv_row,
    extract_header_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legit_features(**overrides) -> dict:
    """Keyword args for a typical legitimate email."""
    base = dict(
        from_field="Alice Smith <alice@company.com>",
        return_path="<alice@company.com>",
        reply_to="",
        message_id="<abc123@company.com>",
        subject="Meeting tomorrow at 10am",
        all_header_keys=["From", "To", "Subject", "Date", "Message-ID", "Return-Path"],
        received_count=2,
        x_mailer="",
        x_spam_status="",
    )
    base.update(overrides)
    return base


def _phishing_features(**overrides) -> dict:
    """Keyword args for a typical phishing email."""
    base = dict(
        from_field="PayPal Support <noreply@evil-domain.ru>",
        return_path="<>",
        reply_to="harvest@attacker.com",
        message_id="<msg@randomdomain.net>",
        subject="=?utf-8?B?VVJHRU5UOiBWZXJpZnkgeW91ciBhY2NvdW50?=",
        all_header_keys=["From", "From", "To", "Subject", "Return-Path"],
        received_count=7,
        x_mailer="PHPMailer 5.2",
        x_spam_status="Yes, score=8.5",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_extract_domain_simple(self):
        assert _extract_domain("alice@company.com") == "company.com"

    def test_extract_domain_with_display_name(self):
        assert _extract_domain("Alice <alice@company.com>") == "company.com"

    def test_extract_domain_empty(self):
        assert _extract_domain("") == ""

    def test_extract_domain_no_at(self):
        assert _extract_domain("nodomain") == ""

    def test_extract_display_name_angle_bracket(self):
        assert _extract_display_name("Alice Smith <alice@co.com>") == "Alice Smith"

    def test_extract_display_name_quoted(self):
        assert _extract_display_name('"PayPal" <noreply@evil.com>') == "PayPal"

    def test_extract_display_name_no_display(self):
        assert _extract_display_name("alice@co.com") == ""


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestExtractHeaderFeatures:
    def test_returns_correct_length(self):
        feats = extract_header_features(**_legit_features())
        assert len(feats) == len(FEATURE_NAMES)

    def test_all_floats(self):
        feats = extract_header_features(**_legit_features())
        assert all(isinstance(f, float) for f in feats)

    def test_legit_email_low_risk_features(self):
        feats = extract_header_features(**_legit_features())
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["multiple_from"] == 0.0
        assert feat["mail_from_empty"] == 0.0
        assert feat["from_return_path_match"] == 1.0
        assert feat["has_reply_to_diff_domain"] == 0.0

    def test_multiple_from_detected(self):
        feats = extract_header_features(
            **_legit_features(
                all_header_keys=["From", "From", "To", "Subject"]
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["multiple_from"] == 1.0

    def test_mail_from_empty_detected(self):
        feats = extract_header_features(**_legit_features(return_path="<>"))
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["mail_from_empty"] == 1.0

    def test_from_return_path_mismatch(self):
        feats = extract_header_features(
            **_legit_features(
                from_field="alice@company.com",
                return_path="<spammer@evil.com>",
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["from_return_path_match"] == 0.0

    def test_reply_to_diff_domain(self):
        feats = extract_header_features(
            **_legit_features(
                from_field="alice@company.com",
                reply_to="harvest@attacker.com",
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["has_reply_to_diff_domain"] == 1.0

    def test_reply_to_same_domain_not_flagged(self):
        feats = extract_header_features(
            **_legit_features(
                from_field="alice@company.com",
                reply_to="support@company.com",
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["has_reply_to_diff_domain"] == 0.0

    def test_from_encoded_word_detected(self):
        feats = extract_header_features(
            **_legit_features(
                from_field="=?utf-8?B?UGF5UGFs?= <noreply@evil.ru>"
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["from_encoded"] == 1.0

    def test_subject_encoded_word_detected(self):
        feats = extract_header_features(
            **_legit_features(
                subject="=?utf-8?B?VVJHRU5U?= action required"
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["subject_has_encoding"] == 1.0

    def test_x_mailer_present(self):
        feats = extract_header_features(**_legit_features(x_mailer="PHPMailer 5.2"))
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["x_mailer_present"] == 1.0

    def test_x_mailer_absent(self):
        feats = extract_header_features(**_legit_features(x_mailer=""))
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["x_mailer_present"] == 0.0

    def test_received_hop_count(self):
        feats = extract_header_features(**_legit_features(received_count=5))
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["received_hop_count"] == 5.0

    def test_duplicate_headers_detected(self):
        feats = extract_header_features(
            **_legit_features(
                all_header_keys=["From", "From", "To", "Subject", "Subject"]
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["duplicate_header_count"] == 2.0

    def test_sender_is_freemail(self):
        feats = extract_header_features(
            **_legit_features(from_field="user@gmail.com")
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["sender_is_freemail"] == 1.0

    def test_sender_is_not_freemail(self):
        feats = extract_header_features(
            **_legit_features(from_field="user@company.com")
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["sender_is_freemail"] == 0.0

    def test_phishing_email_high_risk_features(self):
        feats = extract_header_features(**_phishing_features())
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["multiple_from"] == 1.0
        assert feat["mail_from_empty"] == 1.0
        assert feat["has_reply_to_diff_domain"] == 1.0
        assert feat["x_mailer_present"] == 1.0
        assert feat["has_x_spam"] == 1.0
        assert feat["subject_has_encoding"] == 1.0

    def test_display_name_has_domain(self):
        feats = extract_header_features(
            **_legit_features(
                from_field="paypal.com Security <attacker@evil.ru>"
            )
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["from_display_name_has_domain"] == 1.0


# ---------------------------------------------------------------------------
# CSV row extraction tests
# ---------------------------------------------------------------------------

class TestExtractFeaturesFromCsvRow:
    def test_returns_correct_length(self):
        feats = extract_features_from_csv_row(
            sender="alice@company.com", subject="Hello"
        )
        assert len(feats) == len(FEATURE_NAMES)

    def test_no_structural_attacks_for_legit_sender(self):
        feats = extract_features_from_csv_row(
            sender="Alice <alice@company.com>", subject="Re: meeting"
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["multiple_from"] == 0.0
        assert feat["mail_from_empty"] == 0.0
        assert feat["has_reply_to_diff_domain"] == 0.0

    def test_freemail_sender_detected(self):
        feats = extract_features_from_csv_row(
            sender="user@gmail.com", subject="hi"
        )
        feat = dict(zip(FEATURE_NAMES, feats))
        assert feat["sender_is_freemail"] == 1.0


# ---------------------------------------------------------------------------
# HeaderModel fit / predict tests
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 200, seed: int = 0) -> tuple:
    """Generate synthetic (X, y) where phishing rows have distinctive features."""
    rng = np.random.default_rng(seed)
    n_phish = n // 2
    n_legit = n - n_phish

    # Phishing rows: multiple_from=1, mail_from_empty=1, received_hop_count=7
    X_phish = np.zeros((n_phish, len(FEATURE_NAMES)))
    X_phish[:, FEATURE_NAMES.index("multiple_from")] = 1.0
    X_phish[:, FEATURE_NAMES.index("mail_from_empty")] = 1.0
    X_phish[:, FEATURE_NAMES.index("received_hop_count")] = rng.integers(5, 10, n_phish)
    X_phish[:, FEATURE_NAMES.index("has_reply_to_diff_domain")] = 1.0
    X_phish += rng.normal(0, 0.05, X_phish.shape)

    # Legit rows: from_return_path_match=1, low hops
    X_legit = np.zeros((n_legit, len(FEATURE_NAMES)))
    X_legit[:, FEATURE_NAMES.index("from_return_path_match")] = 1.0
    X_legit[:, FEATURE_NAMES.index("received_hop_count")] = rng.integers(1, 3, n_legit)
    X_legit += rng.normal(0, 0.05, X_legit.shape)

    X = np.vstack([X_phish, X_legit])
    y = np.array([1] * n_phish + [0] * n_legit)
    idx = rng.permutation(n)
    return X[idx], y[idx]


class TestHeaderModelFit:
    def test_fit_returns_self(self):
        X, y = _make_dataset()
        model = HeaderModel(HeaderConfig(calibrate=False))
        result = model.fit(X, y)
        assert result is model

    def test_predict_proba_shape(self):
        X, y = _make_dataset()
        model = HeaderModel(HeaderConfig(calibrate=False))
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10,)

    def test_predict_proba_range(self):
        X, y = _make_dataset()
        model = HeaderModel(HeaderConfig(calibrate=False))
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_binary_output(self):
        X, y = _make_dataset()
        model = HeaderModel(HeaderConfig(calibrate=False))
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_model_learns_signal(self):
        """Model should score phishing rows higher on average than legit rows."""
        X, y = _make_dataset(n=400)
        model = HeaderModel(HeaderConfig(calibrate=False))
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba[y == 1].mean() > proba[y == 0].mean()

    def test_fit_stats_populated(self):
        X, y = _make_dataset()
        model = HeaderModel(HeaderConfig(calibrate=False))
        model.fit(X, y)
        assert "dataset_size" in model.fit_stats_
        assert model.fit_stats_["dataset_size"] == len(y)

    def test_calibrated_model_works(self):
        X, y = _make_dataset(n=300)
        model = HeaderModel(HeaderConfig(calibrate=True, calibration_cv=3))
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5,)


# ---------------------------------------------------------------------------
# Save / load tests
# ---------------------------------------------------------------------------

class TestHeaderModelPersistence:
    def test_save_and_load(self):
        X, y = _make_dataset()
        model = HeaderModel(HeaderConfig(calibrate=False))
        model.fit(X, y)
        original_proba = model.predict_proba(X[:10])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = HeaderModel.load(path)
            loaded_proba = loaded.predict_proba(X[:10])
            np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=6)
        finally:
            os.unlink(path)

    def test_loaded_config_matches(self):
        X, y = _make_dataset()
        config = HeaderConfig(C=0.5, calibrate=False, scale_features=True)
        model = HeaderModel(config)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = HeaderModel.load(path)
            assert loaded.config.C == 0.5
            assert loaded.config.calibrate is False
        finally:
            os.unlink(path)
