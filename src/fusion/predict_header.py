"""CLI for classifying a single email using a trained HeaderModel.

Accepts a raw .eml file path and outputs a phishing probability + verdict.

Usage
-----
  # From a raw .eml file:
  python -m src.fusion.predict_header \\
      --model_path dataset/artifacts/header/header_model.joblib \\
      --eml path/to/email.eml

  # From sender + subject strings (CSV-row mode, for integration testing):
  python -m src.fusion.predict_header \\
      --model_path dataset/artifacts/header/header_model.joblib \\
      --sender "PayPal <noreply@evil.com>" \\
      --subject "=?utf-8?B?VXJnZW50?= Verify your account"

  # Verbose: also print extracted feature values
  python -m src.fusion.predict_header \\
      --model_path dataset/artifacts/header/header_model.joblib \\
      --eml path/to/email.eml \\
      --verbose
"""

import argparse
import sys

import numpy as np

from src.fusion.header_model import (
    FEATURE_NAMES,
    HeaderModel,
    extract_features_from_csv_row,
    extract_features_from_eml_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify an email as phishing or legitimate using header features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        default="dataset/artifacts/header/header_model.joblib",
        help="Path to a trained HeaderModel .joblib file.",
    )
    parser.add_argument(
        "--eml",
        help="Path to a raw .eml file to classify.",
    )
    parser.add_argument(
        "--sender",
        help="Sender/From string (used when --eml is not provided).",
    )
    parser.add_argument(
        "--subject",
        default="",
        help="Subject string (used when --eml is not provided).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for phishing (>= threshold → Phishing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print the extracted feature values.",
    )
    args = parser.parse_args()

    if args.eml is None and args.sender is None:
        print(
            "Error: provide either --eml <path> or --sender <string>.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract features
    if args.eml:
        features = extract_features_from_eml_path(args.eml)
        source = f"eml:{args.eml}"
    else:
        features = extract_features_from_csv_row(
            sender=args.sender or "", subject=args.subject
        )
        source = f"csv:sender={args.sender!r}"

    if args.verbose:
        print("Extracted features:")
        for name, val in zip(FEATURE_NAMES, features):
            print(f"  {name:35s} = {val}")
        print()

    # Load model and predict
    model = HeaderModel.load(args.model_path)
    X = np.array([features], dtype=float)
    proba = float(model.predict_proba(X)[0])
    label = "Phishing" if proba >= args.threshold else "Legitimate"

    print(
        f"source={source}  "
        f"score={proba:.4f}  "
        f"threshold={args.threshold:.2f}  "
        f"label={label}"
    )


if __name__ == "__main__":
    main()
