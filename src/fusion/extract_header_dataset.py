"""Build header feature CSV from EPVME .eml files and a benign CSV source.

Two sources are combined:
  • EPVME extracted .eml files (label=1, phishing) – full header extraction
  • A benign CSV (label=0) – if the CSV has 'sender'/'subject' columns those
    are used; otherwise benign-default header feature vectors are generated.
    Accepts CEAS_08.csv (sender, subject, body, label, urls) or the processed
    body_combined_clean.csv (body, label, source).  Only label=0 rows are used.

Output CSV columns:
  <FEATURE_NAMES...>, label, source

Usage
-----
  # With CEAS_08.csv (preferred – has sender/subject):
  python -m src.fusion.extract_header_dataset \\
      --eml_dir  dataset/dataset/EPVME-Dataset/extracted \\
      --benign_csv src/CEAS_08.csv \\
      --out_path dataset/processed/header_features.csv

  # Fallback with processed body CSV (no sender/subject, uses benign defaults):
  python -m src.fusion.extract_header_dataset \\
      --eml_dir  dataset/dataset/EPVME-Dataset/extracted \\
      --benign_csv dataset/processed/body_combined_clean.csv \\
      --out_path dataset/processed/header_features.csv

Options
-------
  --eml_dir        Root directory of extracted EPVME .eml files (all phishing)
  --benign_csv     CSV with benign rows (label=0); sender/subject used if present
  --out_path       Destination CSV path
  --phishing_limit Max phishing rows (default: all)
  --benign_limit   Max benign rows (default: all)
  --log_every      Log progress every N files (default: 5000)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.fusion.header_model import (
    FEATURE_NAMES,
    extract_features_from_csv_row,
    extract_features_from_eml_path,
)

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


def _iter_eml_files(root: Path) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(".eml"):
                yield Path(dirpath) / name


def _process_epvme(
    eml_dir: Path,
    writer: csv.DictWriter,
    limit: Optional[int],
    log_every: int,
) -> int:
    """Extract header features from EPVME .eml files and write to CSV.

    Returns the number of rows written.
    """
    count = 0
    errors = 0
    for path in _iter_eml_files(eml_dir):
        if limit is not None and count >= limit:
            break
        try:
            features = extract_features_from_eml_path(path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            errors += 1
            continue

        row = dict(zip(FEATURE_NAMES, features))
        row["label"] = 1
        row["source"] = str(path)
        writer.writerow(row)
        count += 1
        if count % log_every == 0:
            logger.info("EPVME: processed %d files…", count)

    logger.info("EPVME done: %d rows written, %d errors.", count, errors)
    return count


def _process_benign_csv(
    benign_csv: Path,
    writer: csv.DictWriter,
    limit: Optional[int],
) -> int:
    """Extract header features from benign CSV rows and write to CSV.

    Only rows with label=0 are used.  If the CSV contains 'sender' and/or
    'subject' columns, they are passed to the feature extractor.  Otherwise
    benign-default feature vectors are generated (all structural attack
    features = 0, which accurately reflects legitimate email behaviour).

    Returns the number of rows written.
    """
    logger.info("Loading benign CSV from %s…", benign_csv)
    df = pd.read_csv(benign_csv)
    df_benign = df[df["label"] == 0].reset_index(drop=True)
    logger.info("Benign CSV: %d label=0 rows available.", len(df_benign))

    has_sender = "sender" in df_benign.columns
    has_subject = "subject" in df_benign.columns
    logger.info(
        "Columns available — sender: %s, subject: %s", has_sender, has_subject
    )

    if limit is not None:
        df_benign = df_benign.head(limit)

    count = 0
    for _, row_data in df_benign.iterrows():
        sender = str(row_data["sender"]) if has_sender else ""
        subject = str(row_data["subject"]) if has_subject else ""
        features = extract_features_from_csv_row(sender=sender, subject=subject)
        out_row = dict(zip(FEATURE_NAMES, features))
        out_row["label"] = 0
        out_row["source"] = str(benign_csv)
        writer.writerow(out_row)
        count += 1

    logger.info("Benign CSV done: %d rows written.", count)
    return count


def build_dataset(
    eml_dir: Path,
    benign_csv: Path,
    out_path: Path,
    phishing_limit: Optional[int] = None,
    benign_limit: Optional[int] = None,
    log_every: int = 5000,
) -> None:
    """Build and write the combined header feature CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FEATURE_NAMES + ["label", "source"]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        phishing_rows = _process_epvme(eml_dir, writer, phishing_limit, log_every)
        benign_rows = _process_benign_csv(benign_csv, writer, benign_limit)

    logger.info(
        "Dataset saved → %s  (phishing=%d, benign=%d, total=%d)",
        out_path,
        phishing_rows,
        benign_rows,
        phishing_rows + benign_rows,
    )


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Build header feature CSV from EPVME .eml files and CEAS_08.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eml_dir",
        default="dataset/dataset/EPVME-Dataset/extracted",
        help="Root directory of extracted EPVME .eml files.",
    )
    parser.add_argument(
        "--benign_csv",
        default="dataset/processed/body_combined_clean.csv",
        help=(
            "CSV with benign rows (label=0). Accepts CEAS_08.csv "
            "(has sender/subject columns) or body_combined_clean.csv "
            "(body/label/source only — benign defaults used for header features)."
        ),
    )
    parser.add_argument(
        "--out_path",
        default="dataset/processed/header_features.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--phishing_limit",
        type=int,
        default=None,
        help="Max phishing (.eml) rows to process.",
    )
    parser.add_argument(
        "--benign_limit",
        type=int,
        default=None,
        help="Max benign (CEAS_08) rows to include.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=5000,
        help="Log progress every N .eml files.",
    )
    args = parser.parse_args()

    build_dataset(
        eml_dir=Path(args.eml_dir),
        benign_csv=Path(args.benign_csv),
        out_path=Path(args.out_path),
        phishing_limit=args.phishing_limit,
        benign_limit=args.benign_limit,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
