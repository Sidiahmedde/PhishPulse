"""Train the super/fusion model on a prepared CSV dataset.

Usage:
  .venv/bin/python src/models/train_super_model.py \
      --data_path dataset/processed/super_model_training.csv \
      --model_path src/models/fusion_model_super_v2.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .super_module import train_fusion_model
except ImportError:
    from super_module import train_fusion_model


REQUIRED_COLUMNS = ["subject", "body", "headers", "label"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the super fusion model.")
    parser.add_argument(
        "--data_path",
        default="dataset/processed/super_model_training.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--model_path",
        default="src/models/fusion_model_super_v2.pkl",
        help="Output path for the trained fusion model.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)

    df = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {df.columns.tolist()}"
        )

    records = df.to_dict(orient="records")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_fusion_model(records, model_path=model_path)


if __name__ == "__main__":
    main()
