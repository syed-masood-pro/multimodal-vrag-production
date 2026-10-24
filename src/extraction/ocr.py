"""Vision OCR fallback. Page-image pixels sent externally are NOT redacted.

Extracted text is redacted by the extraction pipeline before indexing.
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from config import settings
from src.clients.groq_client import vision_chat
from src.core.exceptions import VisionError
from src.core.logger import get_logger
from src.utils.image_utils import image_to_data_uri

log = get_logger(__name__)


def render_page_to_image(pdf_path: str | Path, page_number: int, dpi: int = 180) -> Path:
    out = settings.image_dir / f"ocr_{Path(pdf_path).stem}_p{page_number}.jpg"
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img.save(out, "JPEG", quality=85)
    return out


OCR_PROMPT = (
    "Transcribe ALL readable text from this page image. Preserve line breaks and reading order. "
    "Do not summarize; output raw text only. If the page is blank, output nothing."
)


def ocr_page(pdf_path: str | Path, page_number: int) -> str:
    try:
        img_path = render_page_to_image(pdf_path, page_number)
        text = vision_chat(
            model=settings.model_vision,
            prompt=OCR_PROMPT,
            image_data_uris=[image_to_data_uri(img_path)],
            max_completion_tokens=1500,
        )
        return text.strip()
    except VisionError as e:
        log.warning("ocr_failed", page=page_number, error=str(e))
        return ""
