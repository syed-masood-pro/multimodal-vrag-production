"""Vision RAG answering."""
from __future__ import annotations

from pathlib import Path

from config import settings
from src.clients.groq_client import vision_chat
from src.core.logger import get_logger
from src.rag.prompts import VISION_RAG_PROMPT
from src.utils.image_utils import image_to_data_uri

log = get_logger(__name__)


def answer_vision(*, context: str, question: str, image_paths: list[str], max_images: int = 3) -> str:
    uris: list[str] = []
    for p in image_paths[:max_images]:
        if Path(p).exists():
            uris.append(image_to_data_uri(p))
    if not uris:
        log.warning("no_valid_images_for_vision")
    return vision_chat(
        model=settings.model_vision,
        prompt=VISION_RAG_PROMPT.format(context=context, question=question),
        image_data_uris=uris,
        max_completion_tokens=900,
    )
