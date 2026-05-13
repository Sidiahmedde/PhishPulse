"""Audit the fusion pipeline for leakage and shortcut features.

Usage:
  .venv/bin/python scripts/audit_leakage.py \
      --data_path dataset/processed/super_model_training.csv \
      --seed 42 \
      --out_dir audit_artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.super_module import _clean_email_value, _extract_url_tokens
from src.models.body_module import body_module
from src.models.header_module import header_module
from src.models.sender_module import sender_module
from src.models.subject_module import subject_module
from src.models.url_module import url_module


FEATURE_ORDER = ["subject", "body", "header", "url", "sender", "has_url", "url_count"]
REQUIRED_COLUMNS = ["sender", "subject", "body", "urls", "headers", "label"]
VERDICT_HEADERS = [
    "X-Spam-Status",
    "X-Spam-Flag",
    "X-Spam-Score",
    "X-Spam-Level",
    "X-Spam-Checker-Version",
    "X-Spam-Report",
    "SpamAssassin",
    "X-Microsoft-Antispam",
    "X-Microsoft-Antispam-Message-Info",
    "X-Forefront-Antispam-Report",
    "X-MS-Exchange-Organization-SCL",
    "X-Phish",
    "X-Phishing",
    "X-Virus-Scanned",
    "X-Virus-Status",
    "X-Proofpoint-Spam-Details",
]
SANITIZE_HEADERS = [
    "X-Spam-Status",
    "X-Spam-Flag",
    "X-Spam-Score",
    "X-Spam-Level",
    "X-Spam-Checker-Version",
    "X-Spam-Report",
    "X-Microsoft-Antispam",
    "X-Microsoft-Antispam-Message-Info",
    "X-Forefront-Antispam-Report",
    "X-MS-Exchange-Organization-SCL",
    "X-Phish",
    "X-Phishing",
    "X-Virus-Scanned",
    "X-Virus-Status",
    "X-Proofpoint-Spam-Details",
]
HEADER_NAME_LOOKAHEAD = r"(?=(?:\s+[A-Za-z0-9-]+:)|$)"


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    subject_cache: dict[str, float] = {}
    body_cache: dict[str, float] = {}
    header_cache: dict[str, float] = {}
    sender_cache: dict[str, float] = {}
    url_cache: dict[str, float] = {}
    rows: list[dict[str, float]] = []

    for record in df.to_dict(orient="records"):
        scores: dict[str, float] = {}
        subject = _clean_email_value(record.get("subject"))
        body = _clean_email_value(record.get("body"))
        headers = _clean_email_value(record.get("headers"))
        urls = _clean_email_value(record.get("urls") or record.get("url"))
        sender = _clean_email_value(record.get("sender"))

        if subject:
            if subject not in subject_cache:
                subject_cache[subject] = float(subject_module(subject))
            scores["subject"] = subject_cache[subject]

        if body:
            if body not in body_cache:
                body_cache[body] = float(body_module(body))
            scores["body"] = body_cache[body]

        if headers:
            if headers not in header_cache:
                header_cache[headers] = float(header_module(headers))
            scores["header"] = header_cache[headers]

        if sender:
            if sender not in sender_cache:
                sender_cache[sender] = float(sender_module(sender))
            scores["sender"] = sender_cache[sender]

        url_tokens = _extract_url_tokens(urls)
        scores["has_url"] = float(bool(url_tokens))
        scores["url_count"] = float(len(url_tokens))
        if url_tokens:
            first_url = url_tokens[0]
            if first_url not in url_cache:
                url_cache[first_url] = float(url_module(first_url))
            scores["url"] = url_cache[first_url]

        rows.append(scores)

    feature_df = pd.DataFrame(rows)
    for feature in FEATURE_ORDER:
        if feature not in feature_df.columns:
            feature_df[feature] = 0.0
    return feature_df[FEATURE_ORDER].fillna(0.0).astype(float)


def fit_and_score(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test.to_numpy(), y_score), y_score


def inspect_split_order(root: Path) -> dict[str, object]:
    target = root / "src/models/train_super_model_advanced.py"
    text = target.read_text(encoding="utf-8")
    split_idx = text.find("train_test_split(")
    feature_idx = text.find("X = build_feature_frame(df)")
    leakage_detected = feature_idx != -1 and split_idx != -1 and feature_idx < split_idx
    return {
        "file": str(target.relative_to(root)),
        "build_feature_frame_before_split": leakage_detected,
        "feature_build_position": feature_idx,
        "split_position": split_idx,
        "conclusion": (
            "Leakage risk detected: module probabilities are built for the full dataset before the train/test split."
            if leakage_detected
            else "No split-order issue detected in the inspected training script."
        ),
    }


def header_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    headers = df["headers"].fillna("").astype(str)
    for indicator in VERDICT_HEADERS:
        mask = headers.str.contains(re.escape(indicator), case=False, na=False, regex=True)
        row = {
            "indicator": indicator,
            "overall_count": int(mask.sum()),
            "overall_rate": float(mask.mean()),
        }
        for label_value in sorted(df["label"].unique()):
            label_mask = df["label"] == label_value
            row[f"class_{label_value}_count"] = int((mask & label_mask).sum())
            row[f"class_{label_value}_rate"] = float((mask & label_mask).sum() / max(int(label_mask.sum()), 1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["overall_count", "indicator"], ascending=[False, True])


def sanitize_headers(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    sanitized = text
    for header_name in SANITIZE_HEADERS:
        pattern = re.compile(
            rf"(^|\s+){re.escape(header_name)}:.*?{HEADER_NAME_LOOKAHEAD}",
            flags=re.IGNORECASE,
        )
        sanitized = pattern.sub(" ", sanitized)
    sanitized = re.sub(r"SpamAssassin", " ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def record_hash(row: pd.Series) -> str:
    parts = [
        normalize_text(row["body"]),
        normalize_text(row["subject"]),
        normalize_text(row["sender"]),
        normalize_text(row["headers"]),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def duplicate_overlap_report(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    indexed = df.copy()
    indexed["row_id"] = indexed.index
    indexed["content_hash"] = indexed.apply(record_hash, axis=1)
    train_df = indexed.loc[train_idx, ["row_id", "label", "source", "content_hash"]]
    test_df = indexed.loc[test_idx, ["row_id", "label", "source", "content_hash"]]
    overlap = train_df.merge(
        test_df,
        on="content_hash",
        how="inner",
        suffixes=("_train", "_test"),
    )
    overlap = overlap.sort_values(["content_hash", "row_id_train", "row_id_test"]).reset_index(drop=True)
    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "unique_train_hashes": int(train_df["content_hash"].nunique()),
        "unique_test_hashes": int(test_df["content_hash"].nunique()),
        "overlap_pairs": int(len(overlap)),
        "overlap_unique_hashes": int(overlap["content_hash"].nunique()) if not overlap.empty else 0,
    }
    return overlap, summary


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the fusion pipeline for leakage and shortcut features.")
    parser.add_argument(
        "--data_path",
        default="dataset/processed/super_model_training.csv",
        help="Path to the CSV used for fusion training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out_dir", default="audit_artifacts", help="Output directory for audit artifacts.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)

    split_order = inspect_split_order(ROOT)
    write_json(out_dir / "split_order_check.json", split_order)

    header_counts = header_frequency_table(df)
    header_counts.to_csv(out_dir / "header_shortcut_frequency.csv", index=False)
    write_json(
        out_dir / "header_shortcut_frequency.json",
        header_counts.to_dict(orient="records"),
    )

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label"],
    )

    X = build_feature_frame(df)
    y = df["label"]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    baseline_metrics, baseline_scores = fit_and_score(X_train, X_test, y_train, y_test, args.seed)
    baseline_metrics["rows"] = int(len(df))
    baseline_metrics["train_rows"] = int(len(train_idx))
    baseline_metrics["test_rows"] = int(len(test_idx))
    write_json(out_dir / "metrics_before_sanitization.json", baseline_metrics)

    sanitized_df = df.copy()
    sanitized_df["headers"] = sanitized_df["headers"].map(sanitize_headers)
    sanitized_changed = int((sanitized_df["headers"] != df["headers"].fillna("").astype(str)).sum())
    X_sanitized = build_feature_frame(sanitized_df)
    sanitized_metrics, sanitized_scores = fit_and_score(
        X_sanitized.iloc[train_idx],
        X_sanitized.iloc[test_idx],
        y_train,
        y_test,
        args.seed,
    )
    sanitized_metrics["rows_with_header_changes"] = sanitized_changed
    write_json(out_dir / "metrics_after_header_sanitization.json", sanitized_metrics)

    shuffled_y_train = y_train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    shuffled_y_test = y_test.reset_index(drop=True)
    shuffled_metrics, _ = fit_and_score(
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        shuffled_y_train,
        shuffled_y_test,
        args.seed,
    )
    write_json(out_dir / "shuffle_label_metrics.json", shuffled_metrics)

    overlap_df, overlap_summary = duplicate_overlap_report(df, train_idx, test_idx)
    overlap_df.to_csv(out_dir / "duplicate_overlap_pairs.csv", index=False)
    write_json(out_dir / "duplicate_overlap_summary.json", overlap_summary)

    comparison = pd.DataFrame(
        [
            {"scenario": "before_sanitization", **baseline_metrics},
            {"scenario": "after_header_sanitization", **sanitized_metrics},
            {"scenario": "shuffle_label_test", **shuffled_metrics},
        ]
    )
    comparison.to_csv(out_dir / "metric_comparison.csv", index=False)

    top_header_indicators = header_counts[header_counts["overall_count"] > 0].head(10)
    conclusion_parts: list[str] = []
    if split_order["build_feature_frame_before_split"]:
        conclusion_parts.append(
            "The advanced fusion training script computes module-level features for the full dataset before splitting, so the meta-model evaluation is not a leak-safe stacked estimate."
        )
    if overlap_summary["overlap_unique_hashes"] > 0:
        conclusion_parts.append(
            f"Exact duplicate content overlaps were found across the train/test split ({overlap_summary['overlap_unique_hashes']} unique hashes)."
        )
    if top_header_indicators["overall_count"].sum() > 0:
        conclusion_parts.append(
            "Verdict-like headers are present in the dataset and can provide shortcut information to the header module."
        )
    if not conclusion_parts:
        conclusion_parts.append("No single severe shortcut dominated this audit, but the stacking order still needs to be verified end-to-end.")

    report_lines = [
        "# Leakage Audit Report",
        "",
        f"- Data path: `{data_path}`",
        f"- Seed: `{args.seed}`",
        "",
        "## 1. Split Order Check",
        "",
        f"- Inspected file: `{split_order['file']}`",
        f"- `build_feature_frame(df)` occurs before `train_test_split(...)`: `{split_order['build_feature_frame_before_split']}`",
        f"- Conclusion: {split_order['conclusion']}",
        "",
        "## 2. Header Shortcut Check",
        "",
        "- The `headers` column has already been whitespace-flattened in dataset preprocessing, so sanitization here removes known verdict header fields from a flattened string rather than true raw header lines.",
        f"- Rows with header changes after sanitization: `{sanitized_changed}`",
        "",
        "### Verdict Header Frequency",
        "",
        "| Indicator | Overall | Rate | Legitimate | Legit Rate | Phishing | Phish Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top_header_indicators.iterrows():
        report_lines.append(
            f"| `{row['indicator']}` | {int(row['overall_count'])} | {row['overall_rate']:.4f} | "
            f"{int(row.get('class_0_count', 0))} | {row.get('class_0_rate', 0.0):.4f} | "
            f"{int(row.get('class_1_count', 0))} | {row.get('class_1_rate', 0.0):.4f} |"
        )
    if top_header_indicators.empty:
        report_lines.append("| none detected | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |")

    report_lines.extend(
        [
            "",
            "### Before vs After Header Sanitization",
            "",
            "| Scenario | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Before | {baseline_metrics['accuracy']:.4f} | {baseline_metrics['precision']:.4f} | {baseline_metrics['recall']:.4f} | {baseline_metrics['f1']:.4f} | {baseline_metrics['roc_auc']:.4f} | {baseline_metrics['average_precision']:.4f} |",
            f"| After sanitization | {sanitized_metrics['accuracy']:.4f} | {sanitized_metrics['precision']:.4f} | {sanitized_metrics['recall']:.4f} | {sanitized_metrics['f1']:.4f} | {sanitized_metrics['roc_auc']:.4f} | {sanitized_metrics['average_precision']:.4f} |",
            "",
            "## 3. Shuffle-Label Test",
            "",
            f"- Accuracy: `{shuffled_metrics['accuracy']:.4f}`",
            f"- Precision: `{shuffled_metrics['precision']:.4f}`",
            f"- Recall: `{shuffled_metrics['recall']:.4f}`",
            f"- F1: `{shuffled_metrics['f1']:.4f}`",
            f"- ROC-AUC: `{shuffled_metrics['roc_auc']:.4f}`",
            f"- PR-AUC: `{shuffled_metrics['average_precision']:.4f}`",
            "",
            "## 4. Duplicate Leakage",
            "",
            f"- Train rows: `{overlap_summary['train_rows']}`",
            f"- Test rows: `{overlap_summary['test_rows']}`",
            f"- Unique train hashes: `{overlap_summary['unique_train_hashes']}`",
            f"- Unique test hashes: `{overlap_summary['unique_test_hashes']}`",
            f"- Overlap pairs: `{overlap_summary['overlap_pairs']}`",
            f"- Overlap unique hashes: `{overlap_summary['overlap_unique_hashes']}`",
            "",
            "## Conclusion",
            "",
            "- " + " ".join(conclusion_parts),
            "- Smallest fix: change the fusion evaluation to a leak-safe stacking procedure: split first, train each base model only on the training fold, generate out-of-fold training probabilities plus held-out test probabilities, then fit the meta-model on train probabilities and evaluate only on test probabilities.",
        ]
    )
    (out_dir / "audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote audit artifacts to {out_dir}")
    print((out_dir / "audit_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
