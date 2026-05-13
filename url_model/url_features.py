import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
HEX_IPV4_RE = re.compile(r"0x[0-9a-fA-F]{1,2}")

SUSPICIOUS_WORDS = {
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "password", "bank", "billing", "invoice",
    "payment", "pay", "alert", "support", "helpdesk", "reset", "unlock",
    "suspended", "limited", "security"
}

COMMON_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "t.ly", "lnkd.in", "s.id", "rb.gy"
}

FEATURE_COLUMNS = [
    "url_len",
    "host_len",
    "path_len",
    "query_len",
    "digit_count",
    "special_count",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "at_count",
    "amp_count",
    "eq_count",
    "percent_count",
    "subdomain_parts",
    "has_https",
    "has_ip",
    "has_hex_ip",
    "is_shortener",
    "keyword_hits",
    "host_entropy",
    "path_entropy",
    "has_double_slash_in_path",
    "has_www",
    "suspicious_tld",
]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * np.log2(p)
    return float(ent)


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    return u


def looks_like_ip(hostname: str) -> int:
    if not hostname:
        return 0
    return int(bool(IPV4_RE.match(hostname)))


def has_hex_ip(url: str) -> int:
    if not url:
        return 0
    return int(bool(HEX_IPV4_RE.search(url)))


def extract_url_features(url: str) -> dict:
    u = normalize_url(url)

    hostname = ""
    path = ""
    query = ""
    scheme = ""

    try:
        parsed = urlparse(u)
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
    except ValueError:
        u_sanitized = u.replace("[", "").replace("]", "")
        try:
            parsed = urlparse(u_sanitized)
            u = u_sanitized
            scheme = (parsed.scheme or "").lower()
            hostname = (parsed.hostname or "").lower()
            path = parsed.path or ""
            query = parsed.query or ""
        except ValueError:
            path = u

    url_len = len(u)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    digit_count = sum(ch.isdigit() for ch in u)
    letter_count = sum(ch.isalpha() for ch in u)
    special_count = url_len - digit_count - letter_count

    dot_count = hostname.count(".") if hostname else 0
    hyphen_count = hostname.count("-") if hostname else 0
    underscore_count = u.count("_")
    slash_count = u.count("/")
    at_count = u.count("@")
    amp_count = u.count("&")
    eq_count = u.count("=")
    percent_count = u.count("%")

    host_parts = [p for p in hostname.split(".") if p] if hostname else []
    subdomain_parts = max(0, len(host_parts) - 2)
    tld = host_parts[-1] if len(host_parts) >= 2 else ""

    has_https = int(scheme == "https")
    has_ip = looks_like_ip(hostname)
    hex_ip = has_hex_ip(u)
    is_shortener = int(hostname in COMMON_SHORTENERS)

    token_blob = f"{hostname} {path} {query}".lower()
    keyword_hits = sum(1 for w in SUSPICIOUS_WORDS if w in token_blob)

    host_entropy = shannon_entropy(hostname)
    path_entropy = shannon_entropy(path)

    has_double_slash_in_path = int("//" in path)
    has_www = int(hostname.startswith("www."))
    suspicious_tld = int(tld in {"tk", "ml", "ga", "cf", "gq"})

    features = {
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,
        "query_len": query_len,
        "digit_count": digit_count,
        "special_count": special_count,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "underscore_count": underscore_count,
        "slash_count": slash_count,
        "at_count": at_count,
        "amp_count": amp_count,
        "eq_count": eq_count,
        "percent_count": percent_count,
        "subdomain_parts": subdomain_parts,
        "has_https": has_https,
        "has_ip": has_ip,
        "has_hex_ip": hex_ip,
        "is_shortener": is_shortener,
        "keyword_hits": keyword_hits,
        "host_entropy": host_entropy,
        "path_entropy": path_entropy,
        "has_double_slash_in_path": has_double_slash_in_path,
        "has_www": has_www,
        "suspicious_tld": suspicious_tld,
    }

    return {col: features[col] for col in FEATURE_COLUMNS}


def urls_to_feature_df(url_series: pd.Series) -> pd.DataFrame:
    rows = [extract_url_features(u) for u in url_series.fillna("")]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)