"""Merge multiple CSVs with 'body' and 'label' columns into one CSV.

Usage:
  python -m src.fusion.merge_body_datasets \\
      --inputs data/epvme_body.csv src/CEAS_08.csv \\
      --out_path data/body_combined.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def iter_rows(path: Path) -> Iterable[dict]:
    csv.field_size_limit(10**7)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("body", "label") if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        for row in reader:
            yield {
                "body": row.get("body", ""),
                "label": row.get("label", ""),
                "source": row.get("source", str(path)),
            }


def write_merged(inputs: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["body", "label", "source"])
        writer.writeheader()
        for path in inputs:
            for row in iter_rows(path):
                writer.writerow(row)
                total += 1
    print(f"Wrote {total} rows to {out_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge body datasets into one CSV.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSVs with 'body' and 'label' columns.",
    )
    parser.add_argument("--out_path", required=True, help="Destination CSV path.")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    write_merged(inputs, Path(args.out_path))


if __name__ == "__main__":
    main()
