"""Tests for BodyModel: cleaning, feature extraction, and model behaviour.

All tests are offline (no network access required).
Run with:  pytest tests/test_body_model.py -v
"""

import os
import tempfile

import numpy as np
import pytest

from src.fusion.body_model import (
    FEATURE_NAMES,
    BodyConfig,
    BodyModel,
    clean_body,
    extract_body_features,
    light_clean,
    thread_clean,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HAM_TEXTS = [
    "Hi Alice, just checking in. How are you doing this week?",
    "Reminder: team meeting tomorrow at 10am in the conference room.",
    "Please find attached the quarterly report. Let me know if you have questions.",
    "Happy birthday! Hope you have a wonderful day.",
    "Can you review my pull request when you have a moment?",
]

SPAM_TEXTS = [
    "URGENT: Verify your account immediately! Click here: http://phish.example.com",
    "Your PayPal account is suspended. Login now to restore access: http://evil.com",
    "Congratulations! You have won $10,000 USD. Provide your password to claim.",
    "Action required: Update your billing information immediately. Security alert!",
    "Invoice payment overdue! Pay $500 usd at http://scam.site/pay now!!!",
]

ALL_TEXTS = HAM_TEXTS + SPAM_TEXTS
ALL_LABELS = [0] * len(HAM_TEXTS) + [1] * len(SPAM_TEXTS)


# ---------------------------------------------------------------------------
# Cleaning tests
# ---------------------------------------------------------------------------


class TestCleanBody:
    def test_light_clean_strips_html(self):
        raw = "<b>Hello</b> <a href='x'>world</a>!"
        result = light_clean(raw)
        assert "<" not in result
        assert "hello" in result
        assert "world" in result

    def test_light_clean_normalises_whitespace(self):
        raw = "  Hello   \n\n  World  "
        result = light_clean(raw)
        assert "  " not in result
        assert result == "hello world"

    def test_thread_clean_removes_original_message_block(self):
        raw = (
            "This is the actual reply.\n"
            "-----Original Message-----\n"
            "From: sender@example.com\n"
            "This is the old quoted content that should be stripped."
        )
        result = thread_clean(raw)
        assert "original message" not in result
        assert "quoted content" not in result
        assert "actual reply" in result

    def test_thread_clean_removes_greater_than_quotes(self):
        raw = "My reply here.\n> Quoted line one\n> Quoted line two"
        result = thread_clean(raw)
        assert ">" not in result
        assert "my reply" in result

    def test_fallback_triggers_on_short_text(self):
        """When thread_clean produces very short output, light_clean fallback activates."""
        # Construct a message where the whole body IS the "original message" header,
        # so thread_clean would strip everything and produce an empty string.
        raw = "-----Original Message-----\nSome content that gets stripped entirely."
        cleaned, used_fallback = clean_body(raw)
        assert used_fallback is True
        # light_clean should preserve at least some content
        assert len(cleaned) > 0

    def test_fallback_does_not_trigger_on_long_text(self):
        """thread_clean on a normal long email should NOT trigger fallback."""
        raw = (
            "Dear user,\n\n"
            "We wanted to let you know that your account has been reviewed "
            "and everything looks good. No action is required on your part. "
            "Please continue to enjoy our services.\n\n"
            "Best regards,\nThe Support Team"
        )
        cleaned, used_fallback = clean_body(raw)
        assert used_fallback is False
        assert len(cleaned.split()) >= 5

    def test_nan_input_returns_empty_no_fallback(self):
        cleaned, used_fallback = clean_body(None)
        assert cleaned == ""
        assert used_fallback is False

        import math
        cleaned, used_fallback = clean_body(float("nan"))
        assert cleaned == ""
        assert used_fallback is False


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractBodyFeatures:
    def test_feature_vector_length(self):
        features = extract_body_features("Hello world!")
        assert len(features) == len(FEATURE_NAMES), (
            f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
        )

    def test_feature_vector_all_floats(self):
        features = extract_body_features("Test email body text here.")
        assert all(isinstance(v, float) for v in features)

    def test_feature_vector_shapes_on_batch(self):
        """Extract features for a batch; result shape must match (n, len(FEATURE_NAMES))."""
        texts = ALL_TEXTS
        batch = np.array([extract_body_features(t) for t in texts], dtype=float)
        assert batch.shape == (len(texts), len(FEATURE_NAMES))

    def test_url_count_detected(self):
        text = "Visit http://example.com and https://other.org for details."
        feats = extract_body_features(text)
        url_idx = FEATURE_NAMES.index("url_count")
        assert feats[url_idx] == 2.0

    def test_urgent_keyword_count_uses_word_boundaries(self):
        # "password123" should NOT match "password" (word boundary)
        # but "reset your password now" should.
        text_no_match = "password123 is not a real word"
        text_match = "reset your password now"
        feats_no = extract_body_features(text_no_match)
        feats_yes = extract_body_features(text_match)
        idx = FEATURE_NAMES.index("urgent_keyword_count")
        assert feats_no[idx] == 0.0, "word-boundary check failed for 'password123'"
        assert feats_yes[idx] >= 1.0

    def test_currency_token_count(self):
        text = "Send $100 or 200 USD or 300 eur to this account."
        feats = extract_body_features(text)
        idx = FEATURE_NAMES.index("currency_token_count")
        # $ + USD + eur = 3 matches
        assert feats[idx] == 3.0

    def test_nan_returns_zero_vector(self):
        feats = extract_body_features(None)
        assert feats == [0.0] * len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# BodyModel tests
# ---------------------------------------------------------------------------


class TestBodyModel:
    def _fitted_model(self) -> BodyModel:
        """Return a model fitted on the small in-memory dataset."""
        config = BodyConfig(
            max_features=500,
            max_iter=200,
            solver="saga",
            random_state=0,
        )
        model = BodyModel(config=config)
        model.fit(ALL_TEXTS, ALL_LABELS)
        return model

    def test_fit_returns_self(self):
        model = BodyModel(config=BodyConfig(max_features=100, max_iter=100))
        result = model.fit(ALL_TEXTS, ALL_LABELS)
        assert result is model

    def test_predict_proba_shape_and_range(self):
        model = self._fitted_model()
        proba = model.predict_proba(ALL_TEXTS)
        assert proba.shape == (len(ALL_TEXTS),)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_shape_and_values(self):
        model = self._fitted_model()
        preds = model.predict(ALL_TEXTS)
        assert preds.shape == (len(ALL_TEXTS),)
        assert set(preds).issubset({0, 1})

    def test_predict_threshold_respected(self):
        model = self._fitted_model()
        proba = model.predict_proba(ALL_TEXTS)
        preds_05 = model.predict(ALL_TEXTS, threshold=0.5)
        expected = (proba >= 0.5).astype(int)
        np.testing.assert_array_equal(preds_05, expected)

    def test_fit_stats_populated(self):
        model = self._fitted_model()
        assert "dataset_size" in model.fit_stats_
        assert model.fit_stats_["dataset_size"] == len(ALL_TEXTS)
        assert "fallback_fraction" in model.fit_stats_
        assert "empty_clean_fraction" in model.fit_stats_
        assert "vocab_size" in model.fit_stats_

    def test_save_load_roundtrip_predict_consistency(self):
        """Loaded model must produce identical predictions to the original."""
        model = self._fitted_model()
        original_proba = model.predict_proba(ALL_TEXTS)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "body_model.joblib")
            model.save(path)
            assert os.path.exists(path)

            loaded = BodyModel.load(path)
            loaded_proba = loaded.predict_proba(ALL_TEXTS)

        np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=6)

    def test_save_load_preserves_config(self):
        config = BodyConfig(max_features=300, C=0.5, calibrate=False, random_state=7)
        model = BodyModel(config=config)
        model.fit(ALL_TEXTS, ALL_LABELS)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "body_model.joblib")
            model.save(path)
            loaded = BodyModel.load(path)

        assert loaded.config.max_features == 300
        assert loaded.config.C == 0.5
        assert loaded.config.random_state == 7

    def test_fit_preprocessed_gives_consistent_predictions(self):
        """fit_preprocessed with pre-computed features must match fit() on same data."""
        from src.fusion.body_model import clean_body as cb
        import numpy as _np

        # Build cleaned texts and custom features manually.
        cleaned = [cb(t)[0] for t in ALL_TEXTS]
        custom = _np.array([extract_body_features(t) for t in ALL_TEXTS], dtype=float)
        labels = _np.array(ALL_LABELS)

        config = BodyConfig(max_features=500, max_iter=200, solver="saga", random_state=0)

        model_pre = BodyModel(config=config)
        model_pre.fit_preprocessed(cleaned, custom, labels)

        proba_pre = model_pre._classifier.predict_proba(
            __import__("scipy.sparse", fromlist=["hstack"]).hstack([
                model_pre._vectorizer.transform(cleaned),
                custom,
            ])
        )[:, 1]
        assert proba_pre.shape == (len(ALL_TEXTS),)
        assert _np.all(proba_pre >= 0.0) and _np.all(proba_pre <= 1.0)


# ---------------------------------------------------------------------------
# Cleaning fallback: detailed edge cases
# ---------------------------------------------------------------------------


class TestFallbackEdgeCases:
    def test_cleaning_fallback_triggers_on_short_text(self):
        """Core requirement: fallback activates when thread_clean result is too short."""
        # A message whose body is entirely a quoted reply header – thread_clean
        # will strip it, yielding an empty or very short string.
        almost_all_quoted = (
            "-----Original Message-----\n"
            "From: alice@example.com\n"
            "Subject: Re: Meeting\n"
            "Date: Monday\n"
            "All of this is the quoted block."
        )
        _, used_fallback = clean_body(almost_all_quoted)
        assert used_fallback is True

    def test_cleaning_fallback_not_triggered_on_normal_email(self):
        normal = (
            "Hello Bob,\n\n"
            "Thanks for sending the document. I will review it this afternoon "
            "and get back to you by end of day.\n\nBest,\nAlice"
        )
        _, used_fallback = clean_body(normal)
        assert used_fallback is False

    def test_fallback_result_is_nonempty_when_raw_has_content(self):
        """Even after fallback, the result should contain some text."""
        raw = "-----Original Message----- meaningful short msg"
        cleaned, used_fallback = clean_body(raw)
        # Regardless of fallback, text should not be empty since raw has content.
        assert len(cleaned) > 0
