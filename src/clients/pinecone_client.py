"""Pinecone client wrapper with idempotent index management."""
from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from config import settings
from src.core.exceptions import VectorStoreError
from src.core.logger import get_logger

log = get_logger(__name__)

_pc: Pinecone | None = None


def get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def ensure_index(name: str, dimension: int, metric: str = "cosine") -> None:
    pc = get_pinecone()
    try:
        if pc.has_index(name):
            log.info("pinecone_index_exists", name=name)
            return
        log.info("pinecone_index_creating", name=name, dimension=dimension)
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if pc.describe_index(name).status.get("ready"):
                log.info("pinecone_index_ready", name=name)
                return
            time.sleep(2)
        raise VectorStoreError("Pinecone index did not become ready in time")
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"ensure_index failed: {e}") from e


def get_index(name: str):
    try:
        return get_pinecone().Index(name)
    except Exception as e:
        raise VectorStoreError(f"get_index failed: {e}") from e
