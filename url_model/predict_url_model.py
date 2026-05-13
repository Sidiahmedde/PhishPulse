import json
from pathlib import Path

import joblib
import pandas as pd

from url_features import FEATURE_COLUMNS, extract_url_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "url_model.joblib"
COLUMNS_PATH = BASE_DIR / "url_feature_columns.json"
METADATA_PATH = BASE_DIR / "url_model_metadata.json"

model = joblib.load(MODEL_PATH)

with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
    SAVED_COLUMNS = json.load(f)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

THRESHOLD = float(METADATA.get("threshold", 0.5))


def predict_url_module(url: str) -> dict:
    features = extract_url_features(url)
    row = pd.DataFrame([features])

    for col in SAVED_COLUMNS:
        if col not in row.columns:
            row[col] = 0

    row = row[SAVED_COLUMNS]

    phishing_probability = float(model.predict_proba(row)[0][1])
    prediction = 1 if phishing_probability >= THRESHOLD else 0

    return {
        "module": "url",
        "phishing_probability": round(phishing_probability, 6),
        "prediction": prediction,
        "threshold": THRESHOLD,
        "features_used": len(FEATURE_COLUMNS),
    }


if __name__ == "__main__":
    sample_url = "http://paypal-login-secure-update.ga/reset"
    print(predict_url_module(sample_url))