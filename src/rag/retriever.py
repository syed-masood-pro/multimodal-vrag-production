"""Retrieval facade: hybrid BM25+dense with RRF, cross-encoder rerank, filters."""
from __future__ import annotations

from langchain_core.documents import Document

from config import settings
from src.core.exceptions import RetrievalError
from src.core.db import allowed_document_ids
from src.core.logger import get_logger
from src.core.prompt_safety import wrap_untrusted
from src.rag.hybrid import bm25_store, rrf_fuse
from src.rag.query_rewrite import hyde
from src.vectorstore.pinecone_store import dense_search

log = get_logger(__name__)


def retrieve(
    question: str,
    *,
    namespace: str,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    k: int | None = None,
) -> list[Document]:
    k = k or settings.retrieval_top_k
    try:
        ids = document_ids
        if document_id is not None:
            ids = [document_id] if ids is None else [d for d in ids if d == document_id]
        ids = allowed_document_ids(namespace, ids)
        if ids == []:
            return []

        def scoped(docs):
            active = allowed_document_ids(namespace, ids)
            return docs if active is None else [d for d in docs if d.metadata.get("document_id") in active]

        dense_a = scoped(dense_search(question, namespace=namespace, k=k, document_ids=ids))
        bm25_hits = bm25_store.search(namespace, query=question, top_k=settings.bm25_top_k, document_ids=ids)
        if not dense_a and not bm25_hits:
            return []
        hyde_text = hyde(question) if settings.hyde_enabled else None
        dense_b = scoped(dense_search(hyde_text, namespace=namespace, k=k, document_ids=ids)) if hyde_text else []
        # 4) RRF fusion
        fused = rrf_fuse(dense_a, dense_b, bm25_hits, top_n=max(k, 10))
        return scoped(fused)
    except Exception as e:
        raise RetrievalError(f"Retrieval failed: {e}") from e


def format_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        modality = (d.metadata.get("modality") or "text").upper()
        parts.append(f"[Page {page} | {modality}]\n{d.page_content}")
    body = "\n\n".join(parts)
    return wrap_untrusted("CONTEXT", body)
