"""Build a sanitized training dataset for src/models/super_module.py.

This combines:
  - an existing flat CSV with sender/subject/body/label data
  - the EPVME export with raw header blocks

Output columns:
  sender, subject, body, urls, headers, label, source

Usage:
  .venv/bin/python -m src.preprocess.build_super_model_dataset \
      --base_csv data/processed/cleaned_phishing_dataset.csv \
      --epvme_csv dataset/processed/epvme_email_fields.csv \
      --out_path dataset/processed/super_model_training.csv \
      --balance
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
WHITESPACE_RE = re.compile(r"\s+")

OUTPUT_COLUMNS = ["sender", "subject", "body", "urls", "headers", "label", "source"]


def sanitize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_urls(value: object, fallback_text: str) -> str:
    if pd.isna(value):
        raw = ""
    else:
        raw = str(value).strip()

    if raw and raw.lower() != "nan":
        matches = URL_RE.findall(raw)
        if matches:
            return " ".join(dict.fromkeys(matches))

    body_matches = URL_RE.findall(fallback_text)
    return " ".join(dict.fromkeys(body_matches))


def normalize_base_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {
        "from": "sender",
        "sener": "sender",
        "subejct": "subject",
        "url": "urls",
    }
    df = df.rename(columns=rename_map)

    for column in ["sender", "subject", "body", "label"]:
        if column not in df.columns:
            raise ValueError(f"{path} is missing required column: {column}")

    if "urls" not in df.columns:
        df["urls"] = ""
    if "source" not in df.columns:
        df["source"] = str(path)

    out = pd.DataFrame()
    out["sender"] = df["sender"].map(sanitize_text)
    out["subject"] = df["subject"].map(sanitize_text)
    out["body"] = df["body"].map(sanitize_text)
    out["urls"] = [
        normalize_urls(url_value, body_text)
        for url_value, body_text in zip(df["urls"], out["body"])
    ]
    out["headers"] = ""
    out["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    out["source"] = df["source"].map(sanitize_text)
    return out


def normalize_epvme_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["sender", "subject", "body", "url", "header", "label", "source"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = pd.DataFrame()
    out["sender"] = df["sender"].map(sanitize_text)
    out["subject"] = df["subject"].map(sanitize_text)
    out["body"] = df["body"].map(sanitize_text)
    out["urls"] = [
        normalize_urls(url_value, body_text)
        for url_value, body_text in zip(df["url"], out["body"])
    ]
    out["headers"] = df["header"].map(sanitize_text)
    out["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    out["source"] = df["source"].map(sanitize_text)
    return out


def finalize_dataset(df: pd.DataFrame, balance: bool, random_state: int) -> pd.DataFrame:
    df = df[OUTPUT_COLUMNS].copy()
    df = df[df["label"].isin([0, 1])]

    has_signal = (
        df["subject"].str.len().gt(0)
        | df["body"].str.len().gt(0)
        | df["headers"].str.len().gt(0)
        | df["sender"].str.len().gt(0)
    )
    df = df[has_signal]

    df = df.drop_duplicates(subset=["sender", "subject", "body", "headers", "label"])
    df = df.reset_index(drop=True)

    if balance:
        counts = df["label"].value_counts()
        if len(counts) == 2:
            sample_size = int(counts.min())
            parts = []
            for label_value in [0, 1]:
                group = df[df["label"] == label_value]
                parts.append(group.sample(n=sample_size, random_state=random_state))
            df = pd.concat(parts, ignore_index=True)
            df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a sanitized dataset for super model training."
    )
    parser.add_argument(
        "--base_csv",
        default="data/processed/cleaned_phishing_dataset.csv",
        help="CSV with sender/subject/body/label data.",
    )
    parser.add_argument(
        "--epvme_csv",
        default="dataset/processed/epvme_email_fields.csv",
        help="CSV exported from EPVME .eml files.",
    )
    parser.add_argument(
        "--out_path",
        default="dataset/processed/super_model_training.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample classes to a 1:1 balance.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed used for balancing and shuffling.",
    )
    args = parser.parse_args()

    base_df = normalize_base_dataset(Path(args.base_csv))
    epvme_df = normalize_epvme_dataset(Path(args.epvme_csv))
    combined = pd.concat([base_df, epvme_df], ignore_index=True)
    final_df = finalize_dataset(
        combined,
        balance=args.balance,
        random_state=args.random_state,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"Saved {len(final_df)} rows to {out_path}")
    print(final_df["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
