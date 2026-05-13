"""Preprocess raw email body data and save a cleaned CSV with feature columns.

Output CSV columns (in order):
  body_clean            – thread_clean output (with light_clean fallback)
  urgent_keyword_count  – word-boundary match count of urgency terms
  url_count             – number of http/https URLs
  exclamation_count     – number of '!' characters
  uppercase_count       – number of uppercase letters
  length_chars          – character length of raw text
  length_words          – word count of raw text
  currency_token_count  – occurrences of $, usd, eur, gbp, bitcoin, etc.
  label                 – original label

A sidecar JSON file (<out_path>.stats.json) is written alongside the CSV
with cleaning statistics (fallback_fraction, empty_fraction, dataset_size).
These stats are consumed by train_body_model.py when --use_preprocessed is set.

Usage
-----
  python -m src.fusion.preprocess_body_data \\
      --data_path src/CEAS_08.csv \\
      --out_path artifacts/body/body_preprocessed.csv
"""

import argparse
import json
import logging
import os

import pandas as pd

from src.fusion.body_model import (
    FEATURE_NAMES,
    clean_body,
    extract_body_features,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def preprocess(df: pd.DataFrame) -> tuple:
    """Clean bodies and extract features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``body`` and ``label`` columns.

    Returns
    -------
    df_out : pd.DataFrame
        Preprocessed dataframe with body_clean, feature columns, and label.
    stats : dict
        Cleaning statistics: fallback_fraction, empty_fraction, dataset_size.
    """
    bodies = df["body"].fillna("")
    n = len(bodies)

    cleaned_texts = []
    fallback_flags = []
    for raw in bodies:
        cleaned, used_fallback = clean_body(raw)
        cleaned_texts.append(cleaned)
        fallback_flags.append(used_fallback)

    fallback_count = sum(fallback_flags)
    empty_count = sum(1 for t in cleaned_texts if not t.strip())
    fallback_frac = fallback_count / max(n, 1)
    empty_frac = empty_count / max(n, 1)

    logger.info(
        "Cleaning complete — total=%d  fallback=%d (%.2f%%)  empty=%d (%.2f%%)",
        n,
        fallback_count,
        fallback_frac * 100,
        empty_count,
        empty_frac * 100,
    )

    feature_rows = [extract_body_features(raw) for raw in bodies]
    features_df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)

    df_out = pd.DataFrame({"body_clean": cleaned_texts})
    df_out = pd.concat([df_out, features_df], axis=1)
    df_out["label"] = df["label"].values

    stats = {
        "dataset_size": n,
        "fallback_count": int(fallback_count),
        "fallback_fraction": round(fallback_frac, 4),
        "empty_count": int(empty_count),
        "empty_fraction": round(empty_frac, 4),
        "feature_names": FEATURE_NAMES,
    }
    return df_out, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess email body data for BodyModel training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to raw CSV file (must contain 'body' and 'label' columns).",
    )
    parser.add_argument(
        "--out_path", required=True,
        help="Path for the output preprocessed CSV.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    logger.info("Loading raw data from %s", args.data_path)
    df = pd.read_csv(args.data_path)
    logger.info("Loaded %d rows, columns: %s", len(df), df.columns.tolist())

    missing = [c for c in ("body", "label") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df_out, stats = preprocess(df)

    out_dir = os.path.dirname(args.out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(args.out_path, index=False)
    logger.info("Preprocessed CSV saved → %s  (%d rows)", args.out_path, len(df_out))

    stats_path = args.out_path + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Cleaning stats saved → %s", stats_path)

    logger.info(
        "Summary:\n"
        "  Rows:             %d\n"
        "  Fallback used:    %d (%.2f%%)\n"
        "  Empty after clean:%d (%.2f%%)",
        stats["dataset_size"],
        stats["fallback_count"],
        stats["fallback_fraction"] * 100,
        stats["empty_count"],
        stats["empty_fraction"] * 100,
    )


if __name__ == "__main__":
    main()
