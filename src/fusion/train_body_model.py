"""Train and evaluate BodyModel on email body data.

Produces a self-contained artifact directory with:
  body_model.joblib    – fitted model
  body_metrics.json    – evaluation metrics + dataset/config metadata
  body_roc.png         – ROC curve
  body_pr.png          – Precision-Recall curve
  splits.npz           – train_idx / test_idx (df.index values)

Usage
-----
  # From raw CSV (features computed on-the-fly):
  python -m src.fusion.train_body_model \\
      --data_path src/CEAS_08.csv \\
      --out_dir artifacts/body \\
      --seed 42

  # From preprocessed CSV (features read from columns):
  python -m src.fusion.train_body_model \\
      --data_path data/processed/body_preprocessed.csv \\
      --out_dir artifacts/body \\
      --seed 42 \\
      --use_preprocessed
"""

import argparse
import json
import logging
import os
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.fusion.body_model import BodyConfig, BodyModel, FEATURE_NAMES

logger = logging.getLogger(__name__)

# Columns expected in a preprocessed CSV (produced by preprocess_body_data.py).
_PREPROCESSED_COLS = ["body_clean"] + FEATURE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _save_roc_plot(path: str, fpr, tpr, auc: float) -> None:
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Body Model – ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("ROC plot → %s", path)


def _save_pr_plot(path: str, recall, precision, ap: float) -> None:
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.set_title("Body Model – Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("PR plot → %s", path)


def _load_preprocess_stats(data_path: str) -> dict:
    """Try to load sidecar stats JSON produced by preprocess_body_data.py."""
    stats_path = data_path + ".stats.json"
    if os.path.exists(stats_path):
        try:
            with open(stats_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate BodyModel for phishing detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to input CSV (raw or preprocessed).",
    )
    parser.add_argument(
        "--out_dir", default="artifacts/body",
        help="Directory for all output artifacts (model, metrics, plots, splits).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/test split and model.",
    )
    parser.add_argument(
        "--max_features", type=int, default=50_000,
        help="TF-IDF vocabulary size cap.",
    )
    parser.add_argument(
        "--ngram_range", nargs=2, type=int, default=(1, 2),
        help="Word n-gram range for TF-IDF (e.g., 1 2).",
    )
    parser.add_argument(
        "--min_df", type=float, default=2,
        help="Minimum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--max_df", type=float, default=1.0,
        help="Maximum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="LR regularisation inverse strength.",
    )
    parser.add_argument(
        "--class_weight", default="balanced",
        help="LR class_weight: 'balanced' or 'none'.",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Wrap LR with CalibratedClassifierCV for better probability estimates.",
    )
    parser.add_argument(
        "--calibration_cv", type=int, default=3,
        help="Number of CV folds for calibration (requires --calibrate).",
    )
    parser.add_argument(
        "--use_preprocessed", action="store_true",
        help=(
            "Expect preprocessed columns in the CSV: body_clean + "
            + ", ".join(FEATURE_NAMES)
            + ". Raises an error if any are missing. "
            "Default: False (features computed on-the-fly from raw 'body' column)."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading data from %s", args.data_path)
    df = pd.read_csv(args.data_path)
    logger.info("Loaded %d rows, %d columns: %s", len(df), len(df.columns), df.columns.tolist())

    if "label" not in df.columns:
        raise ValueError(
            f"Missing required 'label' column. Available: {df.columns.tolist()}"
        )

    y = df["label"].values
    class_balance = {
        str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))
    }
    logger.info("Class balance: %s", class_balance)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    use_preprocessed: bool = args.use_preprocessed
    empty_frac: Optional[float] = None
    fallback_frac: Optional[float] = None

    if use_preprocessed:
        missing = [c for c in _PREPROCESSED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"--use_preprocessed was set but the following columns are missing: "
                f"{missing}. Available columns: {df.columns.tolist()}"
            )
        logger.info("Using preprocessed columns from CSV.")
        texts = df["body_clean"].fillna("").tolist()
        custom = df[FEATURE_NAMES].fillna(0).to_numpy(dtype=float)
        empty_frac = float((df["body_clean"].fillna("").str.strip() == "").mean())
        # Attempt to recover fallback fraction from sidecar file.
        preproc_stats = _load_preprocess_stats(args.data_path)
        if "fallback_fraction" in preproc_stats:
            fallback_frac = preproc_stats["fallback_fraction"]
            logger.info(
                "Fallback fraction from preprocess stats: %.4f", fallback_frac
            )
        else:
            logger.info(
                "Fallback fraction not available (run preprocess_body_data.py to capture it)."
            )
    else:
        if "body" not in df.columns:
            raise ValueError(
                f"Missing required 'body' column. Available: {df.columns.tolist()}"
            )
        logger.info("Features will be computed on-the-fly from the raw 'body' column.")
        texts = df["body"].fillna("").tolist()
        custom = None

    # ------------------------------------------------------------------
    # Stratified 80/20 split – save df.index values for reproducibility
    # ------------------------------------------------------------------
    positions = np.arange(len(df))
    train_pos, test_pos = train_test_split(
        positions, test_size=0.2, random_state=args.seed, stratify=y,
    )

    # Persist actual index values (not positional integers) for traceability.
    train_idx = df.index[train_pos].values
    test_idx = df.index[test_pos].values
    splits_path = os.path.join(args.out_dir, "splits.npz")
    np.savez(splits_path, train_idx=train_idx, test_idx=test_idx)
    logger.info(
        "Split indices saved → %s  (train=%d, test=%d)",
        splits_path, len(train_idx), len(test_idx),
    )

    texts_train = [texts[i] for i in train_pos]
    texts_test = [texts[i] for i in test_pos]
    y_train = y[train_pos]
    y_test = y[test_pos]
    if use_preprocessed:
        custom_train = custom[train_pos]
        custom_test = custom[test_pos]

    # ------------------------------------------------------------------
    # Build config and model
    # ------------------------------------------------------------------
    class_weight_val = (
        None if args.class_weight.lower() == "none" else args.class_weight
    )
    min_df = int(args.min_df) if args.min_df >= 1 else float(args.min_df)
    max_df = int(args.max_df) if args.max_df > 1 else float(args.max_df)
    config = BodyConfig(
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        min_df=min_df,
        max_df=max_df,
        C=args.C,
        class_weight=class_weight_val,
        calibrate=args.calibrate,
        calibration_cv=args.calibration_cv,
        random_state=args.seed,
    )
    logger.info("BodyConfig: %s", config)
    model = BodyModel(config=config)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting training…")
    t0 = time.perf_counter()

    if use_preprocessed:
        model.fit_preprocessed(texts_train, custom_train, y_train)
        # Override empty_frac in fit_stats_ with the computed value.
        model.fit_stats_["empty_clean_fraction"] = round(empty_frac, 4)
        model.fit_stats_["fallback_fraction"] = fallback_frac
        model.fit_stats_["dataset_size"] = len(df)
    else:
        model.fit(texts_train, y_train)
        empty_frac = model.fit_stats_.get("empty_clean_fraction")
        fallback_frac = model.fit_stats_.get("fallback_fraction")

    elapsed = time.perf_counter() - t0
    logger.info("Training completed in %.2f s.", elapsed)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    logger.info("Evaluating on %d test samples…", len(y_test))

    if use_preprocessed:
        from scipy.sparse import hstack as sp_hstack
        X_tfidf_test = model._vectorizer.transform(texts_test)
        X_test = sp_hstack([X_tfidf_test, custom_test])
        y_proba = model._classifier.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict_proba(texts_test)

    y_pred = (y_proba >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = float(roc_auc_score(y_test, y_proba))
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
    ap = float(average_precision_score(y_test, y_proba))

    logger.info(
        "Metrics — Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC=%.4f  AP=%.4f",
        acc, prec, rec, f1, auc, ap,
    )

    # ------------------------------------------------------------------
    # Save artifacts (all into out_dir)
    # ------------------------------------------------------------------
    # Model
    model_path = os.path.join(args.out_dir, "body_model.joblib")
    model.save(model_path)

    # Metrics JSON
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": auc,
        "average_precision": ap,
        "dataset_size": len(df),
        "train_size": len(train_pos),
        "test_size": len(test_pos),
        "class_balance": class_balance,
        "empty_clean_fraction": empty_frac,
        "fallback_fraction": fallback_frac,
        "train_time_seconds": round(elapsed, 3),
        "config": {
            "max_features": config.max_features,
            "ngram_range": list(config.ngram_range),
            "C": config.C,
            "solver": config.solver,
            "max_iter": config.max_iter,
            "class_weight": str(config.class_weight),
            "calibrate": config.calibrate,
            "calibration_method": config.calibration_method,
            "calibration_cv": config.calibration_cv,
            "random_state": config.random_state,
        },
    }
    metrics_path = os.path.join(args.out_dir, "body_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics → %s", metrics_path)

    # Plots
    _save_roc_plot(os.path.join(args.out_dir, "body_roc.png"), fpr, tpr, auc)
    _save_pr_plot(os.path.join(args.out_dir, "body_pr.png"), pr_recall, pr_precision, ap)

    logger.info(
        "All artifacts in %s:\n"
        "  body_model.joblib   – fitted model\n"
        "  body_metrics.json   – evaluation metrics\n"
        "  body_roc.png        – ROC curve\n"
        "  body_pr.png         – Precision-Recall curve\n"
        "  splits.npz          – train/test index split",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
