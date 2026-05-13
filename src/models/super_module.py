from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    from .body_module import body_module
    from .header_module import header_module
    from .sender_module import sender_module
    from .subject_module import subject_module
    from .url_module import url_module
except ImportError:
    from body_module import body_module
    from header_module import header_module
    from sender_module import sender_module
    from subject_module import subject_module
    from url_module import url_module


# =========================================================
# 🔧 CONFIG
# =========================================================

MODEL_PATH = Path(__file__).resolve().parent / "fusion_model_super_eval.pkl"
DEFAULT_FEATURE_ORDER = ["subject", "body", "header", "url", "sender", "has_url", "url_count"]
URL_TOKEN_RE = re.compile(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+", re.IGNORECASE)

USE_TRAINED_FUSION = True  # toggle between learned vs manual weights

WEIGHTS = {
    "subject": 1.0,
    "body": 1.0,
    "header": 1.0,
    "sender": 1.0,
    "url": 1.0,
}


def _clean_email_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _extract_url_tokens(url_text):
    text = _clean_email_value(url_text)
    if not text:
        return []
    matches = URL_TOKEN_RE.findall(text)
    if matches:
        return list(dict.fromkeys(matches))
    parts = [part.strip() for part in text.split() if part.strip()]
    return list(dict.fromkeys(parts))


def _build_feature_map(email):
    scores = {}

    subject = _clean_email_value(email.get("subject"))
    body = _clean_email_value(email.get("body"))
    headers = _clean_email_value(email.get("headers"))
    urls = _clean_email_value(email.get("urls") or email.get("url"))
    sender = _clean_email_value(email.get("sender"))

    if subject:
        scores["subject"] = subject_module(subject)

    if body:
        scores["body"] = body_module(body)

    if headers:
        scores["header"] = header_module(headers)

    if sender:
        scores["sender"] = sender_module(sender)

    url_tokens = _extract_url_tokens(urls)
    scores["has_url"] = float(bool(url_tokens))
    scores["url_count"] = float(len(url_tokens))
    if url_tokens:
        scores["url"] = url_module(url_tokens[0])

    return scores


def _vector_from_feature_map(feature_map, feature_order):
    return [feature_map.get(name, 0.0) for name in feature_order]


# =========================================================
# 🧠 MODULE SCORE COLLECTION
# =========================================================

def get_module_scores(email):
    return _build_feature_map(email)


# =========================================================
# 🔥 FALLBACK: WEIGHTED AVERAGE
# =========================================================

def weighted_fusion(scores):
    total_weight = 0
    weighted_sum = 0

    for key, value in scores.items():
        weight = WEIGHTS.get(key, 0)
        weighted_sum += weight * value
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0


# =========================================================
# 🤖 META-MODEL (TRAINED FUSION)
# =========================================================

def train_fusion_model(dataset, model_path=None, feature_order=None):
    """
    dataset: list of dicts like:
    {
        "subject": "...",
        "body": "...",
        "headers": "...",
        "sender": "...",
        "label": 0 or 1
    }
    """

    feature_order = list(feature_order or DEFAULT_FEATURE_ORDER)
    X = []
    y = []

    for email in dataset:
        feature_map = _build_feature_map(email)
        X.append(_vector_from_feature_map(feature_map, feature_order))
        y.append(email["label"])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n=== Fusion Model Evaluation ===")
    print(classification_report(y_test, preds))

    output_path = Path(model_path) if model_path is not None else MODEL_PATH
    bundle = {"model": model, "feature_order": feature_order}
    joblib.dump(bundle, output_path)
    print(f"\nFusion model saved to {output_path}")

    return model


def load_fusion_model(model_path=None):
    path = Path(model_path) if model_path is not None else MODEL_PATH
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded
    return {"model": loaded, "feature_order": list(DEFAULT_FEATURE_ORDER)}


# =========================================================
# 🚀 MAIN SUPER MODULE
# =========================================================

def super_module(email, model_path=None):
    scores = get_module_scores(email)

    if USE_TRAINED_FUSION:
        try:
            bundle = load_fusion_model(model_path=model_path)
            feature_vector = pd.DataFrame(
                [_vector_from_feature_map(scores, bundle["feature_order"])],
                columns=bundle["feature_order"],
            )
            final_score = bundle["model"].predict_proba(feature_vector)[0][1]
        except Exception:
            print("⚠️ Fusion model not found. Falling back to weighted average.")
            final_score = weighted_fusion(scores)
    else:
        final_score = weighted_fusion(scores)

    return {
        "final_score": float(final_score),
        "prediction": "Phishing" if final_score >= 0.5 else "Legitimate",
        "module_scores": scores
    }


# =========================================================
# 🧪 EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":

    # --- Example inference ---
    email = {
        "subject": "URGENT: Verify your account now!",
        "body": "",
        "headers": "",
        "sender": ""
    }

    result = super_module(email)

    print("\n=== Inference ===")
    print("Final Score:", result["final_score"])
    print("Prediction:", result["prediction"])
    print("Module Breakdown:", result["module_scores"])

    df = pd.read_csv("../../data/processed/cleaned_phishing_dataset.csv")
    # print(df.head())

    # --- Example training (you will replace this dataset) ---
    # This is just a placeholder example
    sample_dataset = df.to_dict(orient='records')

    # Uncomment to train fusion model
    #train_fusion_model(sample_dataset)
