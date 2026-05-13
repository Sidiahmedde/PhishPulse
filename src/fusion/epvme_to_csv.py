"""Convert EPVME .eml files to a CSV with body/url/header metadata.

Usage:
  python -m src.fusion.epvme_to_csv \
      --eml_dir epvme/EPVME-Dataset/extracted \
      --out_path dataset/processed/epvme_email_fields.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, Optional

from src.fusion.body_cleaning import clean_body_text

URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)

def _iter_parts(msg: Message) -> Iterable[Message]:
    if msg.is_multipart():
        for part in msg.walk():
            yield part
    else:
        yield msg


def _get_part_text(part: Message) -> Optional[str]:
    content_type = part.get_content_type()
    if content_type not in {"text/plain", "text/html"}:
        return None
    disposition = (part.get("Content-Disposition") or "").lower()
    if "attachment" in disposition:
        return None
    try:
        text = part.get_content()
    except Exception:
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return None
            text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(text, str):
        return None
    return text


def parse_eml(path: Path) -> tuple[bytes, Message]:
    raw = path.read_bytes()
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    return raw, msg


def extract_raw_header(raw: bytes) -> str:
    for separator in (b"\r\n\r\n", b"\n\n"):
        if separator in raw:
            header_bytes = raw.split(separator, 1)[0]
            return header_bytes.decode("utf-8", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def sanitize_text(value: str) -> str:
    return value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def extract_urls(msg: Message) -> str:
    seen: list[str] = []
    for part in _iter_parts(msg):
        text = _get_part_text(part)
        if not text:
            continue
        for url in URL_RE.findall(text):
            if url not in seen:
                seen.append(url)
    return " ".join(seen)


def get_raw_header_value(msg: Message, name: str) -> str:
    for key, value in msg.raw_items():
        if key.lower() == name.lower():
            return str(value)
    return ""


def extract_body_from_msg(msg: Message) -> str:

    plain_text = None
    html_text = None

    for part in _iter_parts(msg):
        text = _get_part_text(part)
        if text is None:
            continue
        if part.get_content_type() == "text/plain" and not plain_text:
            plain_text = text
        elif part.get_content_type() == "text/html" and not html_text:
            html_text = text

    if plain_text:
        return clean_body_text(plain_text)
    if html_text:
        return clean_body_text(html_text)

    # Fallback to raw payload if no suitable part found.
    try:
        payload = msg.get_payload(decode=True)
        if payload:
            return clean_body_text(payload.decode("utf-8", errors="ignore"))
    except Exception:
        pass
    return ""


def iter_eml_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".eml"):
                yield Path(dirpath) / name


def write_csv(eml_dir: Path, out_path: Path, label: int, limit: Optional[int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    empty = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "body",
                "url",
                "sender",
                "subject",
                "header",
                "label",
                "source",
            ],
        )
        writer.writeheader()
        for path in iter_eml_files(eml_dir):
            raw, msg = parse_eml(path)
            body = extract_body_from_msg(msg)
            urls = extract_urls(msg)
            sender = get_raw_header_value(msg, "From")
            subject = get_raw_header_value(msg, "Subject")
            header = extract_raw_header(raw)
            if not body.strip():
                empty += 1
            writer.writerow(
                {
                    "body": sanitize_text(body),
                    "url": sanitize_text(urls),
                    "sender": sanitize_text(sender),
                    "subject": sanitize_text(subject),
                    "header": sanitize_text(header),
                    "label": label,
                    "source": str(path),
                }
            )
            total += 1
            if limit is not None and total >= limit:
                break

    print(f"Wrote {total} rows to {out_path} (empty bodies: {empty}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EPVME .eml files to CSV.")
    parser.add_argument(
        "--eml_dir",
        required=True,
        help="Root directory containing extracted EPVME .eml files.",
    )
    parser.add_argument(
        "--out_path",
        required=True,
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Label to assign to all rows (default: 1 for malicious).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of files to process.",
    )
    args = parser.parse_args()

    write_csv(Path(args.eml_dir), Path(args.out_path), args.label, args.limit)


if __name__ == "__main__":
    main()
