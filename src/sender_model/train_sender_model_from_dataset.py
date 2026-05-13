"""Train a sender-only phishing model from a CSV dataset.

Usage:
  .venv/bin/python src/sender_model/train_sender_model_from_dataset.py \
      --data_path dataset/processed/super_model_training.csv \
      --model_path src/models/sender_model_super_v2.pkl \
      --vectorizer_path src/models/sender_vectorizer_super_v2.pkl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def clean_sender(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sender-only phishing model.")
    parser.add_argument(
        "--data_path",
        default="dataset/processed/super_model_training.csv",
        help="CSV with sender and label columns.",
    )
    parser.add_argument(
        "--model_path",
        default="src/models/sender_model_super_v2.pkl",
        help="Output path for the sender model.",
    )
    parser.add_argument(
        "--vectorizer_path",
        default="src/models/sender_vectorizer_super_v2.pkl",
        help="Output path for the sender vectorizer.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    if "sender" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Required columns missing. Found: {df.columns.tolist()}")

    df = df.copy()
    df["sender"] = df["sender"].map(clean_sender)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    df = df[df["sender"] != ""]

    X = df["sender"]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(classification_report(y_test, preds, digits=4, zero_division=0))

    model_path = Path(args.model_path)
    vectorizer_path = Path(args.vectorizer_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved sender model to {model_path}")
    print(f"Saved sender vectorizer to {vectorizer_path}")


if __name__ == "__main__":
    main()
