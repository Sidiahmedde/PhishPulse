"""Utilities for cleaning raw email body text."""

from __future__ import annotations

import html
import re
import quopri


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"(?is)<script[^>]*>.*?</script>")
_STYLE_RE = re.compile(r"(?is)<style[^>]*>.*?</style>")
_HEX_LINE_RE = re.compile(r"^[0-9A-Fa-f\\s]+$")
_B64_LINE_RE = re.compile(r"^[A-Za-z0-9+/=\\s]+$")
_QP_HEX_RE = re.compile(r"=[0-9A-Fa-f]{2}")


def _strip_html(raw: str) -> str:
    raw = _SCRIPT_RE.sub(" ", raw)
    raw = _STYLE_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    return text


def _decode_quoted_printable(text: str) -> str:
    if _QP_HEX_RE.search(text) is None:
        return text
    try:
        decoded = quopri.decodestring(text.encode("utf-8", errors="ignore"))
        return decoded.decode("utf-8", errors="ignore")
    except Exception:
        return text


def _remove_header_like_block(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    header_like = 0
    stop_idx = 0
    for i, line in enumerate(lines[:40]):
        if not line.strip():
            stop_idx = i + 1
            break
        if re.match(r"^[A-Za-z0-9-]{2,}:", line):
            header_like += 1
    if header_like >= 3 and stop_idx > 0:
        return "\n".join(lines[stop_idx:])
    return text


def _drop_encoded_lines(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 60:
            if _HEX_LINE_RE.match(stripped) and stripped.count(" ") < 5:
                continue
            if _B64_LINE_RE.match(stripped) and stripped.count(" ") < 5:
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _normalize(text: str) -> str:
    out = []
    for ch in text:
        if ch == "\n" or ch == "\t":
            out.append(ch)
        elif 32 <= ord(ch) <= 126:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def clean_body_text(raw: str | None) -> str:
    if raw is None:
        return ""
    text = str(raw)
    text = text.replace("\x00", " ")
    text = _remove_header_like_block(text)
    text = _decode_quoted_printable(text)
    text = _strip_html(text)
    text = _drop_encoded_lines(text)
    text = _normalize(text)
    return " ".join(text.split())
