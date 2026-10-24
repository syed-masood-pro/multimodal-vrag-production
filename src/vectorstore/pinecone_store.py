"""Pinecone vector store with per-session namespaces and document filters."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_pinecone import PineconeVectorStore

from config import settings
from src.clients.pinecone_client import ensure_index, get_index
from src.core.exceptions import VectorStoreError
from src.core.db import allowed_document_ids, dumps, get_conn, loads, namespace_owner
from src.core.logger import get_logger
from src.embeddings.embedder import embedding_dimension, get_embeddings
from src.utils.hashing import sha256_text
from src.rag.hybrid import _key, bm25_store

log = get_logger(__name__)


def _doc_id(doc: Document) -> str:
    return sha256_text(_key(doc))


@lru_cache(maxsize=1)
def _index_ready() -> bool:
    ensure_index(settings.pinecone_index_name, embedding_dimension())
    return True


def get_vectorstore(namespace: str) -> PineconeVectorStore:
    from langchain_pinecone import PineconeVectorStore

    _index_ready()
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        namespace=namespace,
        pinecone_api_key=settings.pinecone_api_key,
    )


def upsert_documents(documents: list[Document], namespace: str | None = None) -> int:
    namespace = namespace if namespace is not None else settings.pinecone_namespace
    if not documents:
        return 0
    for document_id in {d.metadata.get("document_id", "default") for d in documents}:
        bm25_store._check_owner(namespace, document_id)
    vs = get_vectorstore(namespace)
    ids = [_doc_id(d) for d in documents]
    # Pinecone metadata cannot contain nulls, dictionaries or numeric arrays (bbox).
    # Keep filterable scalars and preserve richer citation metadata in a JSON string.
    wire_docs = []
    for doc in documents:
        metadata = {key: value for key, value in doc.metadata.items()
                    if isinstance(value, (str, bool, int, float))
                    or isinstance(value, list) and all(isinstance(item, str) for item in value)}
        metadata["_vrag_metadata_json"] = dumps(doc.metadata)
        wire_docs.append(Document(page_content=doc.page_content, metadata=metadata))
    try:
        vs.add_documents(wire_docs, ids=ids)
        # CLI ingestion intentionally has no registered document lifecycle. Preserve its
        # lexical corpus too, but only after the entire provider upsert has succeeded.
        if namespace_owner(namespace) is None:
            grouped: dict[str, list[Document]] = {}
            for doc in documents:
                grouped.setdefault(doc.metadata.get("document_id", "default"), []).append(doc)
            for document_id, docs in grouped.items():
                bm25_store.upsert(namespace, document_id, docs)
        log.info("upserted", count=len(documents), namespace=namespace)
        return len(documents)
    except Exception as e:
        raise VectorStoreError(f"upsert failed: {e}") from e


def dense_search(
    query: str, namespace: str, k: int, document_id: str | None = None,
    document_ids: list[str] | None = None,
) -> list[Document]:
    ids = document_ids
    if document_id is not None:
        ids = [document_id] if ids is None else [d for d in ids if d == document_id]
    ids = allowed_document_ids(namespace, ids)
    if ids == []:
        return []
    vs = get_vectorstore(namespace)
    filt = {"document_id": {"$in": list(dict.fromkeys(ids))}} if ids is not None else None
    try:
        docs = vs.similarity_search(query, k=k, filter=filt)
        for doc in docs:
            encoded = doc.metadata.pop("_vrag_metadata_json", None)
            if encoded:
                restored = loads(encoded)
                if isinstance(restored, dict):
                    doc.metadata.update(restored)
        return docs if ids is None else [d for d in docs if d.metadata.get("document_id") in ids]
    except Exception as e:
        raise VectorStoreError(f"dense search failed: {e}") from e


def delete_document(namespace: str, document_id: str) -> None:
    bm25_store._check_owner(namespace, document_id)
    _index_ready()
    # Pinecone serverless indexes do not support metadata-filter deletes
    # (that is a pods-only feature); delete by explicit vector ID instead.
    # Vector IDs are deterministic: sha256(chunk_id or evidence_id), so they
    # can be recomputed from the durable local BM25 corpus without needing
    # a successful prior upsert to have completed.
    rows = get_conn().execute(
        "SELECT chunk_id FROM document_chunks WHERE namespace=? AND document_id=?",
        (namespace, document_id),
    ).fetchall()
    ids = [sha256_text(r["chunk_id"]) for r in rows]
    if ids:
        try:
            index = get_index(settings.pinecone_index_name)
            # Pinecone caps delete-by-ids batch size; chunk defensively.
            for i in range(0, len(ids), 1000):
                index.delete(ids=ids[i:i + 1000], namespace=namespace)
        except Exception as e:
            raise VectorStoreError(f"Document vector deletion failed; retry required: {e}") from e
    bm25_store.delete(namespace, document_id)


def clear_namespace(namespace: str | None = None) -> None:
    namespace = namespace if namespace is not None else settings.pinecone_namespace
    if namespace_owner(namespace) == "":
        raise PermissionError("Unknown or ambiguous namespace")
    _index_ready()
    try:
        get_index(settings.pinecone_index_name).delete(delete_all=True, namespace=namespace)
        log.info("namespace_cleared", namespace=namespace)
    except Exception as e:
        raise VectorStoreError(f"Namespace vector deletion failed; retry required: {e}") from e
    bm25_store.clear(namespace)
