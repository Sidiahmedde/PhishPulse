"""Clean a CSV with 'body' and 'label' columns into a normalized body CSV.

Usage:
  python -m src.fusion.clean_body_dataset \
      --in_path src/CEAS_08.csv \
      --out_path dataset/processed/ceas_body_clean.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.fusion.body_cleaning import clean_body_text


def clean_csv(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10**7)

    total = 0
    empty = 0
    with in_path.open("r", encoding="utf-8", errors="ignore", newline="") as f_in:
        reader = csv.DictReader(f_in)
        missing = [c for c in ("body", "label") if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{in_path} missing required columns: {missing}")

        with out_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["body", "label", "source"])
            writer.writeheader()
            for row in reader:
                cleaned = clean_body_text(row.get("body", ""))
                if not cleaned.strip():
                    empty += 1
                writer.writerow(
                    {
                        "body": cleaned,
                        "label": row.get("label", ""),
                        "source": row.get("source", str(in_path)),
                    }
                )
                total += 1

    print(f"Wrote {total} rows to {out_path} (empty bodies: {empty}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean body text in a CSV dataset.")
    parser.add_argument("--in_path", required=True, help="Input CSV path.")
    parser.add_argument("--out_path", required=True, help="Output CSV path.")
    args = parser.parse_args()
    clean_csv(Path(args.in_path), Path(args.out_path))


if __name__ == "__main__":
    main()
