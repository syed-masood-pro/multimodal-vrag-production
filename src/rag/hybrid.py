"""Hybrid retrieval: BM25 (per-document) + dense (Pinecone) fused via Reciprocal Rank Fusion."""
from __future__ import annotations

import re
import hashlib
import math
from collections import Counter, defaultdict

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import settings
from src.core.logger import get_logger
from src.core.db import allowed_document_ids, dumps, get_conn, loads, namespace_owner, tx

log = get_logger(__name__)

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(t: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(t or "")]


class BM25Store:
    """SQLite-backed corpus; build scoped indexes on demand (no stale process cache)."""

    def upsert(self, namespace: str, document_id: str, docs: list[Document]) -> None:
        with tx() as conn:
            self._check_owner(namespace, document_id)
            conn.execute("DELETE FROM document_chunks WHERE namespace=? AND document_id=?", (namespace, document_id))
            for doc in docs:
                if doc.metadata.get("document_id") not in (None, document_id):
                    raise ValueError("Chunk document ownership mismatch")
                metadata = {**doc.metadata, "document_id": document_id}
                chunk = Document(page_content=doc.page_content, metadata=metadata)
                conn.execute("INSERT OR REPLACE INTO document_chunks VALUES(?,?,?,?,?)",
                             (namespace, document_id, _key(chunk), chunk.page_content, dumps(metadata)))
        log.info("bm25_upsert", namespace=namespace, doc=document_id, n=len(docs))

    def _check_owner(self, namespace: str, document_id: str) -> None:
        owner = namespace_owner(namespace)
        row = get_conn().execute("SELECT session_id FROM documents WHERE id=?", (document_id,)).fetchone()
        if owner is not None and (not row or row["session_id"] != owner):
            raise PermissionError("Document does not belong to namespace")
        if owner is None and row:
            raise PermissionError("Registered browser documents require their owner namespace")

    def delete(self, namespace: str, document_id: str) -> None:
        with tx() as conn:
            self._check_owner(namespace, document_id)
            conn.execute("DELETE FROM document_chunks WHERE namespace=? AND document_id=?", (namespace, document_id))

    delete_document = delete

    def clear(self, namespace: str) -> None:
        with tx() as conn:
            if namespace_owner(namespace) == "":
                raise PermissionError("Unknown namespace")
            conn.execute("DELETE FROM document_chunks WHERE namespace=?", (namespace,))

    def search(self, namespace: str, document_id: str | None = None, query: str = "",
               top_k: int = 20, *, document_ids: list[str] | None = None) -> list[Document]:
        ids = document_ids
        if document_id is not None:
            ids = [document_id] if ids is None else [d for d in ids if d == document_id]
        ids = allowed_document_ids(namespace, ids)
        if ids == [] or top_k <= 0 or not _tokenize(query):
            return []
        sql, args = "SELECT * FROM document_chunks WHERE namespace=?", [namespace]
        if ids is not None:
            sql += " AND document_id IN (" + ",".join("?" for _ in ids) + ")"
            args.extend(ids)
        rows = get_conn().execute(sql + " ORDER BY document_id,chunk_id", args).fetchall()
        docs = [Document(page_content=r["content"], metadata=loads(r["metadata_json"])) for r in rows if _tokenize(r["content"])]
        if not docs:
            return []
        bm25 = BM25Okapi([_tokenize(d.page_content) for d in docs])
        # Okapi's unshifted IDF can be zero/negative for tiny corpora. Positive
        # Robertson IDF preserves useful matching for a single uploaded document.
        frequencies = Counter(word for frequency_map in bm25.doc_freqs for word in frequency_map)
        bm25.idf = {word: math.log(1 + (len(docs) - count + .5) / (count + .5))
                for word, count in frequencies.items()}
        scores = bm25.get_scores(_tokenize(query))
        idxs = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in idxs if scores[i] > 0]

    def has(self, namespace: str, document_id: str) -> bool:
        if allowed_document_ids(namespace, [document_id]) == []:
            return False
        return get_conn().execute("SELECT 1 FROM document_chunks WHERE namespace=? AND document_id=? LIMIT 1",
                                  (namespace, document_id)).fetchone() is not None


bm25_store = BM25Store()


def _key(d: Document) -> str:
    m = d.metadata
    identity = m.get("chunk_id") or m.get("evidence_id")
    if not identity:
        identity = hashlib.sha256(dumps([d.page_content, m.get("page"), m.get("modality"),
                                        m.get("image_path"), m.get("image_id"), m.get("bbox"),
                                        m.get("chunk"), m.get("table_number")]).encode()).hexdigest()
    return f"{m.get('document_id', 'default')}:{identity}"


def rrf_fuse(*lists: list[Document], k: int | None = None, top_n: int = 10) -> list[Document]:
    k = k or settings.rrf_k
    scores: dict[str, float] = defaultdict(float)
    by_key: dict[str, Document] = {}
    for lst in lists:
        for rank, doc in enumerate(lst, start=1):
            key = _key(doc)
            scores[key] += 1.0 / (k + rank)
            by_key.setdefault(key, doc)
    return [by_key[k] for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
