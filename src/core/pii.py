"""Lightweight regex-based PII detection & redaction.

For enterprise use, plug in `presidio-analyzer`. This module is dependency-free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns are intentionally conservative to reduce false positives on business docs.
PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL":       re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    "PHONE":       re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"),
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IPV4":        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "AWS_KEY":     re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


@dataclass
class PIIHit:
    kind: str
    start: int
    end: int
    value: str


def find(text: str) -> list[PIIHit]:
    hits: list[PIIHit] = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(text):
            hits.append(PIIHit(kind, m.start(), m.end(), m.group(0)))
    return hits


def redact(text: str, mask: str = "[REDACTED:{k}]") -> tuple[str, list[PIIHit]]:
    hits = sorted(find(text), key=lambda h: h.start, reverse=True)
    out = text
    for h in hits:
        out = out[: h.start] + mask.format(k=h.kind) + out[h.end :]
    return out, hits
