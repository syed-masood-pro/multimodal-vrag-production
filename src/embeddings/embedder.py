"""HuggingFace embedding singleton."""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from src.core.exceptions import EmbeddingError
from src.core.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    try:
        emb = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info("embeddings_loaded", model=settings.embedding_model)
        return emb
    except Exception as e:
        raise EmbeddingError(f"Failed to load embeddings: {e}") from e


def embedding_dimension() -> int:
    return len(get_embeddings().embed_query("dimension check"))
