"""Evidence-first answering shared by sync and SSE.

SSE buffers generation and verification; token events contain finalized answer
chunks, NOT live model tokens. The verifying stage precedes generation/checks.
At most two answer attempts are allowed. Confidence off does not disable grounding.
One text grounding check per citation-valid candidate verifies the cited evidence
set, not exact per-claim entailment or image pixels. History and uncited retrieved
documents are excluded from grounding and optional confidence scoring.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from langchain_core.documents import Document

from config import settings
from src.core.exceptions import ValidationError
from src.core.logger import get_logger
from src.core.prompt_safety import looks_injected, wrap_untrusted
from src.rag.compaction import compact
from src.rag.confidence import _grounding, score_confidence, valid_component
from src.rag.query_rewrite import standalone_query, rewrite
from src.rag.retriever import retrieve
from src.rag.router import RoutingDecision, escalate, model_for, route
from src.rag.text_chain import answer_text
from src.rag.vision_chain import answer_vision

log = get_logger(__name__)
ABSTENTION = "I cannot answer this question reliably from the available document evidence."


@dataclass
class RAGResponse:
    question: str
    answer: str
    tier: str
    model: str
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_components: dict[str, float] = field(default_factory=dict)
    escalated: bool = False
    sources: list[dict[str, Any]] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    latency_ms: int = 0
    classification: dict[str, Any] = field(default_factory=dict)
    injection_flagged: bool = False
    answerable: bool = True
    abstained: bool = False
    abstention_reason: str | None = None
    status: str = "answered"
    citation_warnings: list[str] = field(default_factory=list)
    retrieval_query: str = ""
    confidence_available: bool = False
    confidence_unavailable_reason: str | None = None
    calibration: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate(question: str) -> str:
    if not question or not question.strip():
        raise ValidationError("Question cannot be empty.")
    q = question.strip()
    if len(q) > settings.max_question_chars:
        raise ValidationError(f"Question exceeds {settings.max_question_chars} chars.")
    return q


def _collect_images(docs: list[Document]) -> list[str]:
    return list(dict.fromkeys(str(d.metadata["image_path"]) for d in docs
                             if d.metadata.get("modality") == "visual"
                             and d.metadata.get("image_path")
                             and Path(d.metadata["image_path"]).exists()))


def _bbox(value):
    if (isinstance(value, (list, tuple)) and len(value) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in value)
            and value[0] < value[2] and value[1] < value[3]):
        return list(value)
    return None


def _sources(docs: list[Document]) -> list[dict[str, Any]]:
    out = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        source = m.get("source")
        out.append({"label": f"E{i}", "evidence_id": m.get("evidence_id") or m.get("chunk_id"),
                    "chunk_id": m.get("chunk_id") or m.get("evidence_id"),
                    "document_id": m.get("document_id"),
                    "filename": m.get("filename") or (str(source).replace("\\", "/").split("/")[-1] if source else None),
                    "page": m.get("page"), "modality": m.get("modality"),
                    "bbox": _bbox(m.get("bbox")) if m.get("modality") in {"table", "visual"} else None,
                    "source": source,
                    "quote": (m.get("evidence_preview") or doc.page_content)[:600],
                    "preview": (m.get("evidence_preview") or doc.page_content)[:600],
                    "cited": False})
    return out


def _history_block(history: list[dict[str, str]] | None) -> str:
    n = max(0, settings.conversation_memory_turns) * 2
    if not history or not n:
        return ""
    lines = [f"{m.get('role', 'user').upper()}: {m.get('content', '')[:2000]}" for m in history[-n:]]
    return "PRIOR CONVERSATION (NOT EVIDENCE):\n" + "\n".join(lines) + "\n\n"


def _evidence_block(doc: Document, source: dict) -> str:
    return (f"[{source['label']}] filename={source['filename']!r}; page={source['page']}; "
            f"modality={source['modality']}\n{doc.page_content}")


def _context(docs: list[Document], sources: list[dict], history) -> str:
    # Labels and identity metadata come from retrieval, never from model output.
    blocks = [_evidence_block(d, s) for d, s in zip(docs, sources)]
    # Match answer_vision's default three-image cap and unique attachment order.
    visual_order = []
    for i, path in enumerate(_collect_images(docs)[:3], 1):
        labels = [f"[{s['label']}]" for d, s in zip(docs, sources) if str(d.metadata.get("image_path", "")) == path]
        visual_order.append(f"Image {i}: {', '.join(labels)}")
    body = _history_block(history) + "RETRIEVED EVIDENCE:\n" + "\n\n".join(blocks)
    if visual_order:
        body += "\n\nVISUAL ATTACHMENT ORDER (when images are attached):\n" + "\n".join(visual_order)
    return wrap_untrusted("CONTEXT", body)


def _cited_context(docs: list[Document], sources: list[dict], cited: set[str],
                   generation_images: list[str]) -> str:
    """Text-only verification of the cited set, NOT exact claim entailment.

    Preserve retrieval labels and original generation image positions; filtering
    must not renumber either. Image references are metadata, not pixel evidence.
    Only the first three images are offered to answer_vision by default.
    """
    image_numbers = {path: i for i, path in enumerate(generation_images[:3], 1)}
    blocks = []
    for doc, source in zip(docs, sources):
        if source["label"] not in cited:
            continue
        block = _evidence_block(doc, source)
        if source["modality"] == "visual":
            number = image_numbers.get(str(doc.metadata.get("image_path", "")))
            block += (f"\nGeneration image {number} (reference only)." if number else
                      "\nNo image offered to generation for this source.")
            block += " Only the retrieved visual summary/text is available to this verifier."
        blocks.append(block)
    body = ("CITED RETRIEVED EVIDENCE ONLY:\n" + "\n\n".join(blocks))
    return ("Text-only cited-set verification, not exact per-claim entailment. "
            "No image pixels are supplied to this verifier; image references do not "
            "establish pixel-only claims. Assess support only from the cited text below.\n"
            + wrap_untrusted("CONTEXT", body))


def _citation_check(answer: str, sources: list[dict]) -> tuple[list[str], set[str]]:
    # Accept both singleton [Ei] and combined [E1, E2] / [E1; E2] / [E1 E2]
    # forms by splitting on any non-token separator inside the brackets.
    labels: set[str] = set()
    warnings: list[str] = []
    allowed = {s["label"] for s in sources}
    for raw in re.findall(r"\[([^\]\n]+)\]", answer):
        tokens = [t.strip() for t in re.split(r"[,\s;/|]+", raw) if t.strip()]
        if not tokens or not all(re.fullmatch(r"[Ee]\d+", t) for t in tokens):
            continue  # Not a citation bracket; ignore prose like [note].
        for token in tokens:
            label = token.upper()
            if label in allowed:
                labels.add(label)
            else:
                warning = f"Invalid citation [{label}]"
                if warning not in warnings:
                    warnings.append(warning)
    if not labels:
        warnings.append("Answer has no valid evidence citation.")
    return warnings, labels


def _usable(docs: list[Document]) -> list[Document]:
    return [d for d in docs if d.page_content.strip()
            and "could not summarize" not in d.page_content.lower()]


def _events(question: str, *, namespace: str, document_id: str | None = None,
            history: list[dict[str, str]] | None = None,
            document_ids: list[str] | None = None) -> Iterator[dict]:
    started = time.monotonic()
    q = _validate(question)
    retrieval_query = standalone_query(q, history)
    result = RAGResponse(q, ABSTENTION, "none", "", answerable=False,
                         abstained=True, status="abstained", retrieval_query=retrieval_query,
                         confidence_unavailable_reason="not_evaluated",
                         injection_flagged=looks_injected(q))
    yield {"type": "stage", "stage": "retrieving"}
    raw = _usable(retrieve(retrieval_query, namespace=namespace, document_id=document_id,
                           document_ids=document_ids))
    # No model calls (including rewriting/routing) before the empty evidence gate.
    if raw and history:
        rewritten = rewrite(q, history)
        if rewritten != retrieval_query:
            revised = _usable(retrieve(rewritten, namespace=namespace, document_id=document_id,
                                       document_ids=document_ids))
            if revised:
                raw, retrieval_query = revised, rewritten
                result.retrieval_query = rewritten
    if not raw:
        result.abstention_reason = "no_evidence"
    else:
        yield {"type": "stage", "stage": "compacting"}
        docs = _usable(compact(retrieval_query, raw))
        if not docs:
            result.abstention_reason = "no_evidence"
        else:
            result.sources = _sources(docs)
            result.image_paths = _collect_images(docs)
            context = _context(docs, result.sources, history)
            decision = route(q, docs)
            yield {"type": "stage", "stage": "verifying",
                   "detail": "Buffered generation and verification; not live model token streaming."}
            for attempt in range(2):
                # Never attach a previous candidate's confidence to a retry.
                result.confidence, result.confidence_band = None, None
                result.confidence_components = {}
                result.confidence_available = False
                result.confidence_unavailable_reason = "not_evaluated"
                result.calibration = None
                result.citation_warnings = []
                result.tier, result.model = decision.tier, decision.model
                result.classification = decision.classification
                vision = decision.tier == "vision" and bool(result.image_paths)
                answer = (answer_vision(context=context, question=q, image_paths=result.image_paths)
                          if vision else answer_text(model=decision.model, context=context, question=q, strict=True))
                if (not answer.strip() or "INSUFFICIENT_EVIDENCE" in answer
                    or answer.strip().startswith("I could not find that information in the document")):
                    result.abstention_reason = "insufficient_evidence"
                    break
                warnings, cited = _citation_check(answer, result.sources)
                result.citation_warnings = warnings
                if warnings:
                    result.abstention_reason = "invalid_or_missing_citations"
                else:
                    verification_context = _cited_context(
                        docs, result.sources, cited, result.image_paths if vision else [])
                    # One set-level check per candidate, never a per-claim loop.
                    grounding = _grounding(verification_context, answer)
                    if not valid_component(grounding):
                        result.abstention_reason = "grounding_unavailable"
                        result.confidence_unavailable_reason = "grounding_unavailable"
                        break
                    result.confidence_components = {"grounding": grounding}
                    if grounding < settings.confidence_low:
                        result.abstention_reason = "unsupported_answer"
                    else:
                        conf = score_confidence(model=decision.model, context=verification_context,
                                                question=q, answer=answer, grounding=grounding, vision=vision)
                        result.confidence = conf.score if conf else None
                        result.confidence_band = conf.band if conf else None
                        result.confidence_components = {"grounding": grounding, **(conf.components if conf else {})}
                        result.confidence_available = bool(conf and conf.available)
                        result.confidence_unavailable_reason = conf.unavailable_reason if conf else "disabled"
                        result.calibration = conf.calibration if conf else None
                        if conf and conf.available and conf.band == "LOW":
                            result.abstention_reason = "low_confidence"
                        elif conf and not conf.available and not vision:
                            result.abstention_reason = "confidence_unavailable"
                            break
                        else:
                            result.answer = answer
                            result.answerable, result.abstained = True, False
                            result.status, result.abstention_reason = "answered", None
                            for source in result.sources:
                                source["cited"] = source["label"] in cited
                            break
                if attempt == 0 and decision.tier in {"cheap", "mid"}:
                    tier = escalate(decision.tier)
                    decision = RoutingDecision(tier, model_for(tier), decision.classification, "verification_retry")
                    result.escalated = True
                else:
                    break
    result.latency_ms = int((time.monotonic() - started) * 1000)
    yield {"type": "meta", "tier": result.tier, "model": result.model,
           "sources": result.sources, "image_paths": result.image_paths,
           "injection_flagged": result.injection_flagged}
    # Only the finalized, verified answer (or abstention) leaves this generator.
    for start in range(0, len(result.answer), 160):
        yield {"type": "token", "text": result.answer[start:start+160]}
    yield {"type": "done", **result.to_dict()}


def ask(question: str, *, namespace: str, document_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        document_ids: list[str] | None = None) -> RAGResponse:
    for event in _events(question, namespace=namespace, document_id=document_id,
                         history=history, document_ids=document_ids):
        if event["type"] == "done":
            return RAGResponse(**{k: v for k, v in event.items() if k != "type"})
    raise RuntimeError("Answer policy did not finalize")


def ask_stream(question: str, *, namespace: str, document_id: str | None = None,
               history: list[dict[str, str]] | None = None,
               document_ids: list[str] | None = None) -> Iterator[dict]:
    """SSE finalized-answer chunks after verifying, not live model streaming."""
    try:
        yield from _events(question, namespace=namespace, document_id=document_id,
                           history=history, document_ids=document_ids)
    except Exception as exc:
        log.warning("answer_stream_failed", error=str(exc))
        yield {"type": "error", "error": str(exc)}
