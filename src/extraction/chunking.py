"""Recursive character-based chunking with overlap."""
from __future__ import annotations

from dataclasses import dataclass

from config import settings

SPLITS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    text: str
    start: int
    end: int


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for p in parts:
        candidate = (buf + sep + p) if buf else p
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            if len(p) > size:
                out.extend(_split(p, size, rest))
                buf = ""
            else:
                buf = p
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    raw = _split(text, size, SPLITS)
    # Re-attach overlap by walking the concatenated text.
    chunks: list[Chunk] = []
    cursor = 0
    for piece in raw:
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
        start = max(0, idx - overlap) if chunks else idx
        end = idx + len(piece)
        chunks.append(Chunk(text=text[start:end], start=start, end=end))
        cursor = end
    return chunks
