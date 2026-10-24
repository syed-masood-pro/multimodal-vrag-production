"""Capability-based model router: cheap / mid / frontier / vision."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document

from config import settings
from src.clients.groq_client import chat
from src.core.logger import get_logger
from src.rag.prompts import CLASSIFIER_PROMPT

log = get_logger(__name__)

Tier = Literal["cheap", "mid", "frontier", "vision"]


@dataclass
class RoutingDecision:
    tier: Tier
    model: str
    classification: dict
    reason: str


_TIER_TO_MODEL = {
    "cheap": lambda: settings.model_cheap,
    "mid": lambda: settings.model_mid,
    "frontier": lambda: settings.model_frontier,
    "vision": lambda: settings.model_vision,
}


def model_for(tier: Tier) -> str:
    return _TIER_TO_MODEL[tier]()


def _classify(question: str) -> dict:
    try:
        raw = chat(
            model=settings.model_cheap,
            messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(question=question)}],
            max_completion_tokens=150,
        )
        # tolerate stray prose by extracting first {...}
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
    except Exception as e:
        log.warning("classify_failed_default", error=str(e))
    return {
        "needs_vision": False,
        "complexity": "medium",
        "multi_hop": False,
        "requires_math": False,
    }


def route(question: str, retrieved_docs: list[Document]) -> RoutingDecision:
    has_visual_evidence = any(d.metadata.get("modality") == "visual" for d in retrieved_docs)
    cls = _classify(question)

    if cls.get("needs_vision") or has_visual_evidence:
        return RoutingDecision("vision", model_for("vision"), cls, "visual evidence required")

    complexity = cls.get("complexity", "medium")
    multi_hop = bool(cls.get("multi_hop"))
    requires_math = bool(cls.get("requires_math"))

    if complexity == "complex" or multi_hop or requires_math:
        return RoutingDecision("frontier", model_for("frontier"), cls, "complex / multi-hop / math")
    if complexity == "simple":
        return RoutingDecision("cheap", model_for("cheap"), cls, "simple factoid")
    return RoutingDecision("mid", model_for("mid"), cls, "default mid tier")


def escalate(current: Tier) -> Tier:
    ladder: list[Tier] = ["cheap", "mid", "frontier"]
    if current not in ladder:
        return "frontier"
    i = ladder.index(current)
    return ladder[min(i + 1, len(ladder) - 1)]
