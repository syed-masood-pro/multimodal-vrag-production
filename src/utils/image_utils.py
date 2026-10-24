"""Image encoding / filtering helpers."""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def image_to_data_uri(image_path: str | Path, max_side: int = 1600, quality: int = 85) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def image_dims(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # (w, h)


def is_worth_processing(
    image_path: str | Path,
    min_bytes: int,
    min_dim: int,
) -> bool:
    p = Path(image_path)
    if not p.exists() or p.stat().st_size < min_bytes:
        return False
    try:
        w, h = image_dims(p)
    except Exception:
        return False
    return w >= min_dim and h >= min_dim
