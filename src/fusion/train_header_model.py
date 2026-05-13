"""Train and evaluate HeaderModel on email header features.

Produces a self-contained artifact directory with:
  header_model.joblib   – fitted model
  header_metrics.json   – evaluation metrics + config metadata
  header_roc.png        – ROC curve
  header_pr.png         – Precision-Recall curve
  splits.npz            – train_idx / test_idx (df.index values)

Usage
-----
  python -m src.fusion.train_header_model \\
      --data_path dataset/processed/header_features.csv \\
      --out_dir dataset/artifacts/header \\
      --seed 42

  # With calibration (recommended for fusion meta-model):
  python -m src.fusion.train_header_model \\
      --data_path dataset/processed/header_features.csv \\
      --out_dir dataset/artifacts/header \\
      --calibrate

  # Tune regularisation:
  python -m src.fusion.train_header_model \\
      --data_path dataset/processed/header_features.csv \\
      --out_dir dataset/artifacts/header \\
      --C 0.5 --calibrate
"""

import argparse
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
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

from src.fusion.header_model import FEATURE_NAMES, HeaderConfig, HeaderModel

logger = logging.getLogger(__name__)


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
    ax.set_title("Header Model – ROC Curve")
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
    ax.set_title("Header Model – Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("PR plot → %s", path)


def _save_feature_importance_plot(path: str, model: HeaderModel) -> None:
    """Plot logistic regression coefficients as a feature importance bar chart."""
    try:
        # Unwrap calibrated classifier if needed
        clf = model._classifier
        if hasattr(clf, "calibrated_classifiers_"):
            # Average coefficients across folds
            coefs = np.mean(
                [c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0
            )
        else:
            coefs = clf.coef_[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = np.arange(len(FEATURE_NAMES))
        colors = ["tomato" if c > 0 else "steelblue" for c in coefs]
        ax.barh(y_pos, coefs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Header Model – Feature Coefficients\n(red = phishing signal)")
        ax.set_xlabel("Logistic Regression Coefficient")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Feature importance plot → %s", path)
    except Exception as exc:
        logger.warning("Could not save feature importance plot: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate HeaderModel for phishing detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to header_features.csv produced by extract_header_dataset.py.",
    )
    parser.add_argument(
        "--out_dir",
        default="dataset/artifacts/header",
        help="Directory for all output artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="LR regularisation inverse strength.",
    )
    parser.add_argument(
        "--class_weight", default="balanced",
        help="LR class_weight: 'balanced' or 'none'.",
    )
    parser.add_argument(
        "--no_scale", action="store_true",
        help="Disable StandardScaler (not recommended).",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Wrap LR with CalibratedClassifierCV (recommended for fusion).",
    )
    parser.add_argument(
        "--calibration_cv", type=int, default=3,
        help="CV folds for calibration.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold used for binary metrics in the report.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading data from %s", args.data_path)
    df = pd.read_csv(args.data_path)
    logger.info("Loaded %d rows, columns: %s", len(df), df.columns.tolist())

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Missing required 'label' column.")

    X = df[FEATURE_NAMES].fillna(0.0).to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    class_balance = {
        str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))
    }
    logger.info("Class balance: %s", class_balance)

    # ------------------------------------------------------------------
    # Stratified 80/20 split
    # ------------------------------------------------------------------
    positions = np.arange(len(df))
    train_pos, test_pos = train_test_split(
        positions, test_size=0.2, random_state=args.seed, stratify=y,
    )
    train_idx = df.index[train_pos].values
    test_idx = df.index[test_pos].values
    splits_path = os.path.join(args.out_dir, "splits.npz")
    np.savez(splits_path, train_idx=train_idx, test_idx=test_idx)
    logger.info(
        "Split → %s  (train=%d, test=%d)", splits_path, len(train_idx), len(test_idx)
    )

    X_train, X_test = X[train_pos], X[test_pos]
    y_train, y_test = y[train_pos], y[test_pos]

    # ------------------------------------------------------------------
    # Build and train model
    # ------------------------------------------------------------------
    class_weight_val = (
        None if args.class_weight.lower() == "none" else args.class_weight
    )
    config = HeaderConfig(
        C=args.C,
        class_weight=class_weight_val,
        scale_features=not args.no_scale,
        random_state=args.seed,
        calibrate=args.calibrate,
        calibration_cv=args.calibration_cv,
    )
    logger.info("HeaderConfig: %s", config)

    model = HeaderModel(config=config)

    logger.info("Training…")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Training done in %.2f s.", elapsed)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    logger.info("Evaluating on %d test samples…", len(y_test))
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba >= args.threshold).astype(int)

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
    print("\n" + report)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    model_path = os.path.join(args.out_dir, "header_model.joblib")
    model.save(model_path)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": auc,
        "average_precision": ap,
        "decision_threshold": args.threshold,
        "dataset_size": len(df),
        "train_size": len(train_pos),
        "test_size": len(test_pos),
        "class_balance": class_balance,
        "train_time_seconds": round(elapsed, 3),
        "feature_names": FEATURE_NAMES,
        "config": {
            "C": config.C,
            "solver": config.solver,
            "max_iter": config.max_iter,
            "class_weight": str(config.class_weight),
            "scale_features": config.scale_features,
            "calibrate": config.calibrate,
            "calibration_method": config.calibration_method,
            "calibration_cv": config.calibration_cv,
            "random_state": config.random_state,
        },
    }
    metrics_path = os.path.join(args.out_dir, "header_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics → %s", metrics_path)

    _save_roc_plot(os.path.join(args.out_dir, "header_roc.png"), fpr, tpr, auc)
    _save_pr_plot(os.path.join(args.out_dir, "header_pr.png"), pr_recall, pr_precision, ap)
    _save_feature_importance_plot(
        os.path.join(args.out_dir, "header_feature_importance.png"), model
    )

    logger.info(
        "All artifacts in %s:\n"
        "  header_model.joblib           – fitted model\n"
        "  header_metrics.json           – evaluation metrics\n"
        "  header_roc.png                – ROC curve\n"
        "  header_pr.png                 – Precision-Recall curve\n"
        "  header_feature_importance.png – coefficient bar chart\n"
        "  splits.npz                    – train/test index split",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
