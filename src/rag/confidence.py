"""Optional heuristic confidence, not a guaranteed correctness probability.

off disables scoring (not the pipeline's independent grounding gate); fast uses
grounding/agreement; full also measures contrast. Failed components stay absent.
Feedback calibration estimates preference only and never replaces safety gates.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from config import settings
from src.clients.groq_client import chat
from src.core.logger import get_logger
from src.embeddings.embedder import get_embeddings
from src.rag.prompts import GROUNDING_PROMPT, RAG_SYSTEM_LOOSE, RAG_SYSTEM_STRICT, TEXT_RAG_PROMPT

log = get_logger(__name__)
Band = Literal["HIGH", "MED", "LOW"]
FEATURE_VERSION = "text-evidence-v2"
FEATURES = {"fast": {"grounding", "agreement"},
            "full": {"grounding", "agreement", "contrast"}}


@dataclass
class ConfidenceResult:
    score: float | None
    band: Band | None
    components: dict[str, float]
    available: bool = True
    unavailable_reason: str | None = None
    calibration: dict[str, Any] = field(default_factory=dict)


def valid_component(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def _cos(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b) or not all(math.isfinite(v) for v in [*a, *b]):
        raise ValueError("Invalid embedding")
    norm = math.sqrt(sum(v*v for v in a)) * math.sqrt(sum(v*v for v in b))
    if not norm:
        raise ValueError("Zero embedding")
    return max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b)) / norm))


def _band(score: float) -> Band:
    return "HIGH" if score >= settings.confidence_high else "MED" if score >= settings.confidence_low else "LOW"


def _grounding(context: str, answer: str) -> float | None:
    try:
        raw = chat(model=settings.model_cheap, messages=[{
            "role": "user", "content": GROUNDING_PROMPT.format(context=context, answer=answer)
        }], max_completion_tokens=60)
        start, end = raw.find("{"), raw.rfind("}")
        value = json.loads(raw[start:end+1])["score"]
        if valid_component(value):
            return float(value)
    except Exception as exc:
        log.warning("grounding_unavailable", error=str(exc))
    return None


def _sample_answers(model: str, context: str, question: str, n: int, system: str) -> list[str]:
    msgs = TEXT_RAG_PROMPT.format_messages(system=system, context=context, question=question)
    payload = [{"role": "user" if m.type == "human" else m.type, "content": m.content} for m in msgs]
    out = []
    for _ in range(n):
        try:
            value = chat(model=model, messages=payload, temperature=0.7, max_completion_tokens=500)
            if value and value.strip():
                out.append(value)
        except Exception as exc:
            log.warning("sample_failed", error=str(exc))
    return out


def _agreement(answer: str, samples: list[str]) -> float | None:
    if not samples:
        return None
    try:
        vecs = get_embeddings().embed_documents([answer] + samples)
        if len(vecs) != len(samples) + 1:
            return None
        return max(0.0, sum(_cos(vecs[0], v) for v in vecs[1:]) / len(samples))
    except Exception as exc:
        log.warning("agreement_unavailable", error=str(exc))
        return None


def _contrast(model: str, context: str, question: str, answer: str) -> float | None:
    strict = _sample_answers(model, context, question, 1, RAG_SYSTEM_STRICT)
    loose = _sample_answers(model, context, question, 1, RAG_SYSTEM_LOOSE)
    if not strict or not loose:
        return None
    try:
        a, pos, neg = get_embeddings().embed_documents([answer, strict[0], loose[0]])
        return max(0.0, min(1.0, (_cos(a, pos) - _cos(a, neg) + 1) / 2))
    except Exception as exc:
        log.warning("contrast_unavailable", error=str(exc))
        return None


def load_calibration(mode: str, features: set[str]) -> dict | None:
    """Reject legacy, nonfinite, mixed-feature, and truth-claiming artifacts."""
    try:
        artifact = json.loads((settings.cache_dir / "confidence_weights.json").read_text(encoding="utf-8"))
        if (not isinstance(artifact, dict) or not isinstance(artifact.get("keys"), list)
                or not isinstance(artifact.get("component_weights"), dict)
                or not isinstance(artifact.get("metrics"), dict)):
            return None
        if (artifact.get("schema_version") != 2
                or artifact.get("feature_version") != FEATURE_VERSION
                or artifact.get("target") != "user_preference"
                or artifact.get("mode") != mode
                or artifact.get("method") != "logistic"
                or set(artifact.get("keys", [])) != features
                or len(artifact["keys"]) != len(features)
                or set(artifact.get("component_weights", {})) != features):
            return None
        numbers = [artifact["intercept"], *artifact["component_weights"].values()]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers):
            return None
        metrics = artifact["metrics"]
        for split in ("train", "heldout"):
            stats = metrics[split]
            if (not isinstance(stats, dict)
                    or not all(type(stats[k]) is int for k in ("n", "positive", "negative"))
                    or stats["n"] < 2 or stats["positive"] < 1 or stats["negative"] < 1
                    or stats["positive"] + stats["negative"] != stats["n"]
                    or not valid_component(stats["brier"]) or not valid_component(stats["accuracy"])):
                return None
        return artifact
    except (OSError, ValueError, TypeError, KeyError):
        return None


def score_confidence(*, model: str, context: str, question: str, answer: str,
                     grounding: float | None = None, vision: bool = False) -> ConfidenceResult | None:
    mode = settings.confidence_mode
    if mode == "off":
        return None
    if vision:
        return ConfidenceResult(None, None, {}, False, "vision_not_measured",
                                {"kind": "unavailable", "reason": "Text samples cannot measure image-only support."})
    grounding = _grounding(context, answer) if grounding is None else grounding
    count = 2 if mode == "fast" else 5
    samples = _sample_answers(model, context, question, count, RAG_SYSTEM_STRICT)
    measured = {"grounding": grounding,
                "agreement": _agreement(answer, samples) if len(samples) == count else None}
    if mode == "full":
        measured["contrast"] = _contrast(model, context, question, answer)
    components = {k: float(v) for k, v in measured.items() if valid_component(v)}
    missing = FEATURES.get(mode, set()) - components.keys()
    if missing or mode not in FEATURES:
        return ConfidenceResult(None, None, components, False,
                                "measurement_unavailable:" + ",".join(sorted(missing)))
    weights = {"grounding": 0.6, "agreement": 0.4} if mode == "fast" else {
        "grounding": 0.4, "agreement": 0.4, "contrast": 0.2}
    score = sum(weights[k] * components[k] for k in weights)
    calibration: dict[str, Any] = {"kind": "heuristic", "feature_version": FEATURE_VERSION,
                                    "mode": mode, "correctness_probability": False}
    artifact = load_calibration(mode, set(components))
    if artifact:
        z = artifact["intercept"] + sum(artifact["component_weights"][k] * components[k] for k in components)
        if math.isfinite(z):
            calibration.update({"kind": "heuristic_with_preference_calibration", "schema_version": 2,
                                "target": "user_preference", "heldout": artifact["metrics"]["heldout"],
                                "preference_score": 1 / (1 + math.exp(-max(-700, min(700, z))))})
    # Preference is deliberately separate: a popular answer need not be true.
    return ConfidenceResult(round(score, 3), _band(score), components, calibration=calibration)
