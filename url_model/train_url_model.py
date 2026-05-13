import json
import os
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from url_features import FEATURE_COLUMNS, urls_to_feature_df

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

CSV_PATH = REPO_ROOT / "data" / "raw" / "SpamAssasin.csv"
MODEL_PATH = BASE_DIR / "url_model.joblib"
COLUMNS_PATH = BASE_DIR / "url_feature_columns.json"
METADATA_PATH = BASE_DIR / "url_model_metadata.json"

URL_REGEX = r"https?://[^\s]+|www\.[^\s]+"

DEFAULT_THRESHOLD = 0.50


def extract_first_url(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(URL_REGEX, text)
    return match.group(0) if match else ""


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Put SpamAssasin.csv under data/raw/ or update CSV_PATH."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"body", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def evaluate_model(model, X_test, y_test, threshold: float) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(
            y_test, preds, digits=4, zero_division=0
        ),
    }
    return metrics


def train_and_compare(X_train, y_train, X_test, y_test):
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_name = None
    best_model = None
    best_score = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, DEFAULT_THRESHOLD)
        results[name] = metrics

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_name = name
            best_model = model

    return best_name, best_model, results


def main():
    df = load_dataset(CSV_PATH)

    df["extracted_url"] = df["body"].apply(extract_first_url)
    df = df[df["extracted_url"] != ""].copy()

    print(f"Rows with URLs: {len(df)}")

    X = urls_to_feature_df(df["extracted_url"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    best_name, best_model, results = train_and_compare(X_train, y_train, X_test, y_test)
    best_metrics = results[best_name]

    joblib.dump(best_model, MODEL_PATH)

    with open(COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    metadata = {
        "model_name": best_name,
        "threshold": DEFAULT_THRESHOLD,
        "dataset_path": str(CSV_PATH),
        "num_rows_with_urls": int(len(df)),
        "num_features": len(FEATURE_COLUMNS),
        "metrics": {
            "roc_auc": best_metrics["roc_auc"],
            "f1": best_metrics["f1"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "confusion_matrix": best_metrics["confusion_matrix"],
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved columns to: {COLUMNS_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")

    for model_name, metrics in results.items():
        print(f"\n=== {model_name} ===")
        print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"F1:       {metrics['f1']:.4f}")
        print(f"Precision:{metrics['precision']:.4f}")
        print(f"Recall:   {metrics['recall']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print(metrics["classification_report"])


if __name__ == "__main__":
    main()