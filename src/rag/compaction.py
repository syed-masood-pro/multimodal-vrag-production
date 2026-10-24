"""Context compaction pipeline: dedup → rerank → sentence filter → token budget."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from langchain_core.documents import Document

from config import settings
from src.core.logger import get_logger
from src.embeddings.embedder import get_embeddings

log = get_logger(__name__)


# ------------------------- tokenization -------------------------
@lru_cache(maxsize=1)
def _tokenizer():
    # Load only when token budgeting is used, not when the API is imported.
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text))


# ------------------------- dedup -------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    # Both are normalized by embedder → dot product
    return sum(x * y for x, y in zip(a, b))


def dedupe(docs: list[Document], threshold: float = 0.95) -> list[Document]:
    if len(docs) < 2:
        return docs
    emb = get_embeddings()
    vecs = emb.embed_documents([d.page_content for d in docs])
    kept: list[tuple[Document, list[float]]] = []
    for d, v in zip(docs, vecs):
        # Similar wording in different documents/pages is distinct provenance.
        scope = (d.metadata.get("document_id"), d.metadata.get("page"), d.metadata.get("modality"))
        if any(scope == (kd.metadata.get("document_id"), kd.metadata.get("page"), kd.metadata.get("modality"))
               and d.metadata.get("modality") != "visual" and _cosine(v, kv) >= threshold for kd, kv in kept):
            continue
        kept.append((d, v))
    log.debug("dedup", before=len(docs), after=len(kept))
    return [d for d, _ in kept]


# ------------------------- rerank -------------------------
@lru_cache(maxsize=1)
def _reranker():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.reranker_model)
    except Exception as e:  # pragma: no cover
        log.warning("reranker_unavailable", error=str(e))
        return None


def rerank(question: str, docs: list[Document], top_n: int | None = None) -> list[Document]:
    top_n = top_n or settings.rerank_top_n
    if not docs:
        return docs
    ce = _reranker()
    if ce is None:
        return docs[:top_n]
    pairs = [(question, d.page_content) for d in docs]
    try:
        scores = ce.predict(pairs)
    except Exception as e:
        log.warning("rerank_failed_fallback", error=str(e))
        return docs[:top_n]
    ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
    #return [d for _, d in ranked[:top_n]]
    
    floor=settings.rerank_min_score
    filtered=[d for score, d in ranked[:top_n] if 1.0/(1.0 + math.exp(-float(score))) >=floor]
    if not filtered and ranked:
        log.info("rerank_all_below_floor",top_score=float(ranked[0][0]),floor=floor)
    return filtered


# ------------------------- sentence filter -------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_filter(question: str, docs: list[Document], min_sim: float = 0.25) -> list[Document]:
    emb = get_embeddings()
    q_vec = emb.embed_query(question)
    out: list[Document] = []
    for d in docs:
        # Do not filter tables/visuals — they are already condensed
        if d.metadata.get("modality") in {"table", "visual"}:
            out.append(d)
            continue
        sents = [s.strip() for s in _SENT_SPLIT.split(d.page_content) if s.strip()]
        if not sents:
            out.append(d)
            continue
        vecs = emb.embed_documents(sents)
        kept = [s for s, v in zip(sents, vecs) if _cosine(q_vec, v) >= min_sim]
        content = " ".join(kept) if kept else d.page_content
        out.append(Document(page_content=content, metadata={**d.metadata,
                    "evidence_preview": d.metadata.get("evidence_preview") or d.page_content[:600]}))
    return out


# ------------------------- token budget -------------------------
def enforce_budget(docs: list[Document], budget: int | None = None) -> list[Document]:
    budget = budget or settings.context_token_budget
    kept: list[Document] = []
    used = 0
    for d in docs:
        t = count_tokens(d.page_content)
        if used + t > budget:
            remaining = budget - used
            if remaining > 0:
                enc = _tokenizer()
                truncated = enc.decode(enc.encode(d.page_content)[:remaining])
                kept.append(Document(page_content=truncated, metadata={**d.metadata,
                                     "evidence_preview": d.metadata.get("evidence_preview") or d.page_content[:600]}))
                used += remaining
            break
        kept.append(d)
        used += t
    log.debug("budget_enforced", kept=len(kept), tokens=used, budget=budget)
    return kept


# ------------------------- pipeline -------------------------
def compact(question: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return docs
    x = dedupe(docs)
    x = rerank(question, x)
    x = sentence_filter(question, x)
    x = enforce_budget(x)
    return x
