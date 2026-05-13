"""Generate report-friendly PNG graphs from saved super-model artifacts.

Usage:
  .venv/bin/python src/models/generate_report_graphs.py \
      --artifacts_dir dataset/artifacts/super_eval \
      --graphs_dir graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fusion_metrics(metrics_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    label_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "average_precision": "PR-AUC",
    }
    df["metric_label"] = df["metric"].map(label_map).fillna(df["metric"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(df["metric_label"], df["value"], color="#28536B")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Final Fusion Model Metrics")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, df["value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    _save(fig, out_path)


def plot_baseline_comparison(baselines_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(baselines_csv)
    metrics = ["accuracy", "f1", "roc_auc", "average_precision"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]

    x = range(len(df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.4))
    colors = ["#28536B", "#C2948A", "#5FAD56", "#D7816A"]
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        positions = [i + (idx - 1.5) * width for i in x]
        ax.bar(positions, df[metric], width=width, label=label, color=color)

    ax.set_xticks(list(x), df["model"], rotation=15)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Comparison vs Meta-Model Fusion")
    ax.legend(ncols=4, fontsize=9)
    _save(fig, out_path)


def plot_coefficients(coefficients_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(coefficients_csv).sort_values("coefficient", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = ["#D7816A" if value < 0 else "#5FAD56" for value in df["coefficient"]]
    bars = ax.barh(df["feature"], df["coefficient"], color=colors)
    ax.set_xlabel("Coefficient")
    ax.set_title("Learned Fusion Weights")
    ax.axvline(0, color="black", linewidth=1)
    for bar, value in zip(bars, df["coefficient"]):
        ax.text(value + (0.15 if value >= 0 else -0.15), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    _save(fig, out_path)


def plot_odds_ratios(coefficients_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(coefficients_csv).sort_values("odds_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(df["feature"], df["odds_ratio"], color="#28536B")
    ax.set_yscale("log")
    ax.set_ylabel("Odds Ratio (log scale)")
    ax.set_title("Fusion Feature Odds Ratios")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, df["odds_ratio"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2e}", ha="center", va="bottom", fontsize=8, rotation=90)
    _save(fig, out_path)


def plot_ablation(ablation_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(ablation_csv).sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.barh(df["ablated_feature"], df["f1"], color="#C2948A")
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("F1 Score After Removing Feature")
    ax.set_title("Ablation Study Impact")
    for bar, value in zip(bars, df["f1"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    _save(fig, out_path)


def plot_ablation_metric_drop(ablation_csv: Path, fusion_metrics_csv: Path, out_path: Path) -> None:
    ablation = pd.read_csv(ablation_csv)
    fusion_metrics = pd.read_csv(fusion_metrics_csv).set_index("metric")["value"]
    baseline_f1 = float(fusion_metrics["f1"])
    ablation["f1_drop"] = baseline_f1 - ablation["f1"]
    ablation = ablation.sort_values("f1_drop", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(ablation["ablated_feature"], ablation["f1_drop"], color="#D7816A")
    ax.set_ylabel("F1 Drop")
    ax.set_title("F1 Drop When Each Module Is Removed")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, ablation["f1_drop"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    _save(fig, out_path)


def plot_single_vs_fusion(baselines_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(baselines_csv)
    best_single = df[df["model"].str.startswith("best_single_module:")].head(1)
    selected = pd.concat(
        [best_single, df[df["model"] == "meta_model_fusion"]],
        ignore_index=True,
    ).copy()
    selected["label"] = selected["model"].map(
        lambda value: "Best Single Module" if value.startswith("best_single_module:") else "Meta-Model Fusion"
    )
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = ["#C2948A", "#5FAD56"]
    for idx, (_, row) in enumerate(selected.iterrows()):
        positions = [i + (idx - 0.5) * width for i in x]
        values = [row[m] for m in metrics]
        ax.bar(positions, values, width=width, label=row["label"], color=colors[idx])

    ax.set_xticks(list(x), metric_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Best Single Module vs Final Fusion Model")
    ax.legend()
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG graphs for the report.")
    parser.add_argument(
        "--artifacts_dir",
        default="dataset/artifacts/super_eval",
        help="Directory containing saved CSV artifacts.",
    )
    parser.add_argument(
        "--graphs_dir",
        default="graphs",
        help="Directory for output PNG files.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    plot_fusion_metrics(
        artifacts_dir / "fusion_metrics.csv",
        graphs_dir / "graph_fusion_metrics_bar_chart.png",
    )
    plot_baseline_comparison(
        artifacts_dir / "fusion_baselines.csv",
        graphs_dir / "graph_fusion_baseline_comparison.png",
    )
    plot_coefficients(
        artifacts_dir / "fusion_coefficients.csv",
        graphs_dir / "graph_fusion_learned_coefficients.png",
    )
    plot_odds_ratios(
        artifacts_dir / "fusion_coefficients.csv",
        graphs_dir / "graph_fusion_odds_ratios.png",
    )
    plot_ablation(
        artifacts_dir / "fusion_ablation.csv",
        graphs_dir / "graph_fusion_ablation_f1.png",
    )
    plot_ablation_metric_drop(
        artifacts_dir / "fusion_ablation.csv",
        artifacts_dir / "fusion_metrics.csv",
        graphs_dir / "graph_fusion_ablation_f1_drop.png",
    )
    plot_single_vs_fusion(
        artifacts_dir / "fusion_baselines.csv",
        graphs_dir / "graph_best_single_vs_fusion.png",
    )

    print(f"Graphs written to {graphs_dir}")


if __name__ == "__main__":
    main()
