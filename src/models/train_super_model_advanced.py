"""Train and evaluate a leak-safe fusion model with report artifacts.

Outputs:
  - trained model bundle (.pkl)
  - confusion matrix (.csv, .png)
  - ROC curve (.png)
  - PR curve (.png)
  - metrics table (.csv, .json)
  - baseline comparison (.csv)
  - coefficient table (.csv)
  - ablation table (.csv)

This version uses a strict stacked evaluation:
  1. optionally filter to source groups present in both classes
  2. split train/test
  3. train each base model only on the training fold
  4. generate out-of-fold probabilities on the training fold
  5. generate held-out probabilities on the test fold
  6. train the meta-model on training probabilities only
  7. evaluate the meta-model on test probabilities only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from url_model.url_features import urls_to_feature_df


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
HEADER_NAME_LOOKAHEAD = r"(?=(?:\s+[A-Za-z0-9-]+:)|$)"
TEXT_COLUMNS = ["subject", "body", "sender", "urls", "headers", "source"]
REQUIRED_COLUMNS = ["subject", "body", "sender", "urls", "headers", "label", "source"]
SOURCE_DEPTH = 3


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Fusion Model Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"Fusion ROC-AUC = {auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Fusion Model ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return float(auc)


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, label=f"Fusion AP = {ap:.4f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Fusion Model Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return float(ap)


def sanitize_headers(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    sanitized = text
    for header_name in VERDICT_HEADERS:
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
    return re.sub(r"\s+", " ", str(value)).strip()


def source_group(value: object, depth: int = SOURCE_DEPTH) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return "/".join(text.split("/")[:depth])


def sender_domain(value: object) -> str:
    text = normalize_text(value).lower()
    match = re.search(r"@([^>\s]+)", text)
    if match:
        return match.group(1)
    return text


def filter_common_sources(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    working = df.copy()
    working["source_group"] = working["source"].map(source_group)
    counts = (
        working.groupby(["source_group", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "label_0", 1: "label_1"})
        .reset_index()
    )
    counts["keep"] = counts.get("label_0", 0).gt(0) & counts.get("label_1", 0).gt(0)
    keep_groups = set(counts.loc[counts["keep"], "source_group"])
    filtered = working[working["source_group"].isin(keep_groups)].copy()
    summary = {
        "original_rows": int(len(df)),
        "filtered_rows": int(len(filtered)),
        "kept_source_groups": sorted(keep_groups),
        "dropped_source_groups": sorted(set(counts["source_group"]) - keep_groups),
    }
    return filtered.drop(columns=["source_group"]), summary


def prepare_dataset(df: pd.DataFrame, source_mode: str) -> tuple[pd.DataFrame, dict[str, object]]:
    working = df.copy()
    for column in TEXT_COLUMNS:
        working[column] = working[column].map(normalize_text)
    working["label"] = pd.to_numeric(working["label"], errors="coerce").astype(int)
    working = working[working["label"].isin([0, 1])].reset_index(drop=True)
    working["headers"] = working["headers"].map(sanitize_headers)

    summary = {
        "source_mode": source_mode,
        "original_rows": int(len(working)),
        "class_balance_before_filter": {
            str(label): int(count)
            for label, count in working["label"].value_counts().sort_index().items()
        },
    }

    if source_mode == "common_only":
        working, source_summary = filter_common_sources(working)
        summary.update(source_summary)
    else:
        summary["filtered_rows"] = int(len(working))
        summary["kept_source_groups"] = sorted({source_group(value) for value in working["source"]})
        summary["dropped_source_groups"] = []

    summary["class_balance_after_filter"] = {
        str(label): int(count)
        for label, count in working["label"].value_counts().sort_index().items()
    }
    summary["non_empty_header_rate"] = float(working["headers"].ne("").mean())
    return working.reset_index(drop=True), summary


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    split_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if split_mode == "random":
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
            stratify=df["label"],
        )
        summary = {
            "split_mode": split_mode,
            "group_count": None,
        }
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True), summary

    if split_mode == "sender_domain_grouped":
        groups = df["sender"].map(sender_domain)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, df["label"], groups=groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        summary = {
            "split_mode": split_mode,
            "group_count": int(groups.nunique()),
            "train_group_count": int(groups.iloc[train_idx].nunique()),
            "test_group_count": int(groups.iloc[test_idx].nunique()),
        }
        return train_df, test_df, summary

    raise ValueError(f"Unsupported split_mode: {split_mode}")


def available_modules(train_df: pd.DataFrame, y_train: pd.Series) -> list[str]:
    modules: list[str] = []
    if train_df["subject"].ne("").any():
        modules.append("subject")
    if train_df["body"].ne("").any():
        modules.append("body")
    if train_df["sender"].ne("").any():
        modules.append("sender")
    if train_df["urls"].ne("").any():
        modules.append("url")
    header_by_class = train_df.assign(label=y_train.values).groupby("label")["headers"].apply(lambda s: s.ne("").any())
    if len(header_by_class) == 2 and bool(header_by_class.all()):
        modules.append("header")
    return modules


def resolve_modules(
    available: list[str],
    requested: list[str] | None,
) -> list[str]:
    if not requested:
        return available
    requested_set = [name.strip() for name in requested if name.strip()]
    invalid = [name for name in requested_set if name not in available]
    if invalid:
        raise ValueError(f"Requested modules are not available for this dataset/split: {invalid}. Available: {available}")
    return requested_set


def module_estimator(name: str):
    if name == "subject":
        return make_pipeline(
            TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    if name == "body":
        return make_pipeline(
            TfidfVectorizer(
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
            ),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    if name == "sender":
        return make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, lowercase=True),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    if name == "header":
        return make_pipeline(
            TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    if name == "url":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    raise ValueError(f"Unknown module: {name}")


def module_features(name: str, df: pd.DataFrame):
    if name == "url":
        return urls_to_feature_df(df["urls"])
    return df[name]


def auxiliary_features(df: pd.DataFrame) -> pd.DataFrame:
    url_tokens = df["urls"].fillna("").astype(str).str.split().map(lambda parts: [part for part in parts if part.strip()])
    return pd.DataFrame(
        {
            "has_url": url_tokens.map(lambda parts: float(bool(parts))),
            "url_count": url_tokens.map(lambda parts: float(len(parts))),
        }
    )


def build_stacked_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    module_names: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_train_parts: dict[str, np.ndarray] = {}
    X_test_parts: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict[str, float]] = {}

    for module_name in module_names:
        estimator = module_estimator(module_name)
        train_X = module_features(module_name, train_df)
        test_X = module_features(module_name, test_df)

        oof = cross_val_predict(
            estimator,
            train_X,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        fitted = clone(estimator).fit(train_X, y_train)
        test_score = fitted.predict_proba(test_X)[:, 1]

        X_train_parts[module_name] = oof
        X_test_parts[module_name] = test_score
        test_metrics[module_name] = compute_metrics(test_df["label"].to_numpy(), test_score)

    aux_train = auxiliary_features(train_df)
    aux_test = auxiliary_features(test_df)
    for column in aux_train.columns:
        X_train_parts[column] = aux_train[column].to_numpy()
        X_test_parts[column] = aux_test[column].to_numpy()

    feature_order = module_names + list(aux_train.columns)
    X_train = pd.DataFrame({name: X_train_parts[name] for name in feature_order})
    X_test = pd.DataFrame({name: X_test_parts[name] for name in feature_order})
    return X_train, X_test, test_metrics


def fit_meta_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)
    return model


def equal_weight_average(df: pd.DataFrame, probability_features: list[str]) -> np.ndarray:
    return df[probability_features].mean(axis=1).to_numpy()


def best_single_module_name(df: pd.DataFrame, y_true: np.ndarray, probability_features: list[str]) -> str:
    best_name = probability_features[0]
    best_f1 = -1.0
    for name in probability_features:
        score = df[name].to_numpy()
        f1 = f1_score(y_true, (score >= 0.5).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
    return best_name


def save_coefficients(model: LogisticRegression, feature_order: list[str], out_path: Path) -> pd.DataFrame:
    coef = model.coef_[0]
    df = pd.DataFrame(
        {
            "feature": feature_order,
            "coefficient": coef,
            "odds_ratio": [math.exp(value) for value in coef],
        }
    ).sort_values("coefficient", ascending=False)
    df.to_csv(out_path, index=False)
    return df


def save_metrics(metrics: dict, out_csv: Path, out_json: Path) -> None:
    table = pd.DataFrame(
        [
            {"metric": "accuracy", "value": metrics["accuracy"]},
            {"metric": "precision", "value": metrics["precision"]},
            {"metric": "recall", "value": metrics["recall"]},
            {"metric": "f1", "value": metrics["f1"]},
            {"metric": "roc_auc", "value": metrics["roc_auc"]},
            {"metric": "average_precision", "value": metrics["average_precision"]},
        ]
    )
    table.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_baselines(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fusion_score: np.ndarray,
    probability_features: list[str],
    out_path: Path,
) -> pd.DataFrame:
    best_name = best_single_module_name(X_test, y_test, probability_features)
    rows = []
    best_score = X_test[best_name].to_numpy()
    equal_score = equal_weight_average(X_test, probability_features)

    for name, score in [
        (f"best_single_module:{best_name}", best_score),
        ("equal_weight_average", equal_score),
        ("meta_model_fusion", fusion_score),
    ]:
        metrics = compute_metrics(y_test, score)
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def save_ablation(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    probability_features: list[str],
    out_path: Path,
) -> pd.DataFrame:
    rows = []
    for feature in probability_features:
        X_ablated = X_test.copy()
        X_ablated[feature] = 0.5
        score = model.predict_proba(X_ablated)[:, 1]
        metrics = compute_metrics(y_test, score)
        rows.append(
            {
                "ablated_feature": feature,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            }
        )
    df = pd.DataFrame(rows).sort_values("f1")
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leak-safe fusion model with evaluation artifacts.")
    parser.add_argument(
        "--data_path",
        default="dataset/processed/super_model_training.csv",
        help="Path to super model training CSV.",
    )
    parser.add_argument(
        "--model_path",
        default="src/models/fusion_model_super_eval.pkl",
        help="Output path for the trained fusion model bundle.",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="dataset/artifacts/super_eval",
        help="Directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--graphs_dir",
        default="graphs",
        help="Directory for graph/image outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split and models.",
    )
    parser.add_argument(
        "--source_mode",
        choices=["common_only", "all"],
        default="common_only",
        help="Keep only source groups present in both classes, or use all rows.",
    )
    parser.add_argument(
        "--split_mode",
        choices=["random", "sender_domain_grouped"],
        default="sender_domain_grouped",
        help="Evaluation split strategy.",
    )
    parser.add_argument(
        "--modules",
        default="subject,url",
        help="Comma-separated module list to benchmark. Available modules depend on the dataset and split.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    artifacts_dir = Path(args.artifacts_dir)
    graphs_dir = Path(args.graphs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared_df, dataset_summary = prepare_dataset(df, source_mode=args.source_mode)
    if len(prepared_df) < 10:
        raise ValueError("Prepared dataset is too small after filtering.")

    train_df, test_df, split_summary = split_dataset(
        prepared_df,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    probability_features = resolve_modules(
        available_modules(train_df, y_train),
        args.modules.split(",") if args.modules else None,
    )
    if not probability_features:
        raise ValueError("No usable module features remained after filtering.")

    X_train, X_test, module_metrics = build_stacked_features(
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        module_names=probability_features,
        seed=args.seed,
    )

    model = fit_meta_model(X_train, y_train, args.seed)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_score)

    feature_order = probability_features + ["has_url", "url_count"]
    bundle = {
        "model": model,
        "feature_order": feature_order,
        "dataset_summary": dataset_summary,
        "probability_features": probability_features,
    }
    joblib.dump(bundle, model_path)

    cm = np.array(metrics["confusion_matrix"])
    pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        artifacts_dir / "fusion_confusion_matrix.csv"
    )
    plot_confusion_matrix(cm, graphs_dir / "graph_fusion_confusion_matrix.png")
    metrics["roc_auc"] = plot_roc(y_test.to_numpy(), y_score, graphs_dir / "graph_fusion_roc_curve.png")
    metrics["average_precision"] = plot_pr(
        y_test.to_numpy(), y_score, graphs_dir / "graph_fusion_precision_recall_curve.png"
    )
    metrics["rows"] = int(len(prepared_df))
    metrics["train_rows"] = int(len(train_df))
    metrics["test_rows"] = int(len(test_df))
    metrics["source_mode"] = args.source_mode
    metrics["split_mode"] = args.split_mode
    metrics["probability_features"] = probability_features
    save_metrics(metrics, artifacts_dir / "fusion_metrics.csv", artifacts_dir / "fusion_metrics.json")
    coef_df = save_coefficients(model, feature_order, artifacts_dir / "fusion_coefficients.csv")
    baseline_df = save_baselines(
        X_test,
        y_test.to_numpy(),
        y_score,
        probability_features,
        artifacts_dir / "fusion_baselines.csv",
    )
    ablation_df = save_ablation(
        model,
        X_test,
        y_test.to_numpy(),
        probability_features,
        artifacts_dir / "fusion_ablation.csv",
    )
    dataset_summary["split_summary"] = split_summary
    with (artifacts_dir / "fusion_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)
    with (artifacts_dir / "fusion_module_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(module_metrics, f, indent=2)

    print(f"Saved model bundle to {model_path}")
    print(f"Artifacts saved under {artifacts_dir}")
    print("Dataset summary:")
    print(json.dumps(dataset_summary, indent=2))
    print("Fusion metrics:")
    print(json.dumps(metrics, indent=2))
    print("Top coefficients:")
    print(coef_df.head(10).to_string(index=False))
    print("Baselines:")
    print(baseline_df.to_string(index=False))
    print("Ablation:")
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
