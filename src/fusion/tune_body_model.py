"""Lightweight hyperparameter tuning for BodyModel.

Runs a small grid search on a train/validation split derived from the
existing train/test split (splits.npz). Selects the best config by F1
on the validation set, then evaluates on test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src.fusion.body_model import BodyConfig, BodyModel


def _eval_thresholds(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, object]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = (-1.0, 0.5)
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1[0]:
            best_f1 = (f1, float(t))
    preds = (proba >= best_f1[1]).astype(int)
    return {
        "threshold": best_f1[1],
        "acc": float(accuracy_score(y_true, preds)),
        "prec": float(precision_score(y_true, preds)),
        "rec": float(recall_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune BodyModel on a small grid.")
    parser.add_argument("--data_path", required=True, help="CSV with body+label.")
    parser.add_argument("--split_path", required=True, help="splits.npz from train.")
    parser.add_argument("--out_path", required=True, help="JSON report output path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    texts = df["body"].fillna("").tolist()
    labels = df["label"].astype(int).values

    splits = np.load(args.split_path)
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    # Build a validation split from the train portion.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_sub_idx, val_idx = next(sss.split(train_idx, labels[train_idx]))
    train_sub = train_idx[train_sub_idx]
    val = train_idx[val_idx]

    grid = [
        {"max_features": 30_000, "ngram_range": (1, 2), "C": 1.0, "min_df": 2},
        {"max_features": 50_000, "ngram_range": (1, 2), "C": 1.0, "min_df": 2},
        {"max_features": 30_000, "ngram_range": (1, 3), "C": 1.0, "min_df": 2},
        {"max_features": 50_000, "ngram_range": (1, 3), "C": 1.0, "min_df": 2},
    ]

    best = {"f1": -1.0}
    trials = []
    for params in grid:
        config = BodyConfig(
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            min_df=params["min_df"],
            max_df=1.0,
            C=params["C"],
            class_weight="balanced",
            max_iter=6000,
            random_state=args.seed,
        )
        model = BodyModel(config=config)
        model.fit([texts[i] for i in train_sub], labels[train_sub])
        proba = model.predict_proba([texts[i] for i in val])
        metrics = _eval_thresholds(labels[val], proba)
        trial = {
            "params": {
                "max_features": config.max_features,
                "ngram_range": list(config.ngram_range),
                "min_df": config.min_df,
                "C": config.C,
            },
            "val": metrics,
        }
        trials.append(trial)
        if metrics["f1"] > best["f1"]:
            best = {
                "f1": metrics["f1"],
                "threshold": metrics["threshold"],
                "config": asdict(config),
            }

    # Retrain best on full train_idx, then evaluate on test with val threshold.
    best_config = BodyConfig(**best["config"])
    final_model = BodyModel(config=best_config)
    final_model.fit([texts[i] for i in train_idx], labels[train_idx])
    test_proba = final_model.predict_proba([texts[i] for i in test_idx])
    test_preds = (test_proba >= best["threshold"]).astype(int)
    test_metrics = {
        "acc": float(accuracy_score(labels[test_idx], test_preds)),
        "prec": float(precision_score(labels[test_idx], test_preds)),
        "rec": float(recall_score(labels[test_idx], test_preds)),
        "f1": float(f1_score(labels[test_idx], test_preds)),
    }

    report = {
        "best_config": best_config.__dict__,
        "val_threshold": best["threshold"],
        "val_best_f1": best["f1"],
        "test_metrics_at_val_threshold": test_metrics,
        "trials": trials,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote tuning report to {out_path}")


if __name__ == "__main__":
    main()
