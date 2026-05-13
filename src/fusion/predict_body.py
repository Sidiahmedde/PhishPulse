"""CLI for classifying a single email body with a trained BodyModel.

Usage
-----
  # Provide text directly:
  python -m src.fusion.predict_body --model_path artifacts/body/body_model.joblib \\
      --text "Paste your email body here"

  # Provide a file:
  python -m src.fusion.predict_body --model_path artifacts/body/body_model.joblib \\
      --file path/to/email.txt

  # Or pipe via stdin:
  cat path/to/email.txt | python -m src.fusion.predict_body \\
      --model_path artifacts/body/body_model.joblib
"""

import argparse
import sys

from src.fusion.body_model import BodyModel


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --text, --file, or pipe input via stdin.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify an email body as phishing or legitimate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        default="artifacts/body/body_model.joblib",
        help="Path to a trained BodyModel .joblib file.",
    )
    parser.add_argument(
        "--text",
        help="Email body text to classify.",
    )
    parser.add_argument(
        "--file",
        help="Path to a text file containing the email body.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for phishing (>= threshold).",
    )
    args = parser.parse_args()

    body = _read_text(args)

    model = BodyModel.load(args.model_path)
    proba = float(model.predict_proba([body])[0])
    label = "Phishing" if proba >= args.threshold else "Legitimate"

    print(f"score={proba:.4f} threshold={args.threshold:.2f} label={label}")


if __name__ == "__main__":
    main()
