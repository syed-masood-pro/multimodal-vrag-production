"""Lexical proxy evaluation, NOT Ragas or a semantic correctness benchmark.

Usage:
    python -m scripts.eval --dataset eval/eval_set.json --namespace <ns> [--document-id <id>]

Dataset format (JSON list):
    [
      {"question": "...", "reference": "...", "expected_keywords": ["..."]},
      ...
    ]

Metrics computed (all in [0,1]):
    - answer_relevancy    : token-level overlap between question and answer
    - keyword_recall      : fraction of expected_keywords present in answer
    - context_precision   : lexical overlap proxy using actual evidence previews
    - faithfulness_proxy  : reused grounding score from confidence
    - reference_overlap   : token overlap between answer and reference (if present)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.pipeline import ask  # noqa: E402

TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(s or "")}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _keyword_recall(answer: str, kws: list[str]) -> float | None:
    if not kws:
        return None
    ans = answer.lower()
    return sum(1 for k in kws if k.lower() in ans) / len(kws)


def _context_precision(answer: str, sources_texts: list[str]) -> float | None:
    sources_texts = [t for t in sources_texts if t.strip()]
    if not sources_texts:
        return None
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    used = 0
    for t in sources_texts:
        if len(_tokens(t) & ans_tokens) >= 3:
            used += 1
    return used / len(sources_texts)


def _evidence_metrics(sources: list[dict], expected: list[str] | None) -> dict:
    if expected is None:
        return {"evidence_recall": None, "evidence_precision": None}
    relevant = set(expected)
    actual = {s["evidence_id"] for s in sources if s.get("evidence_id")}
    # Legacy sources without stable IDs cannot be evaluated as non-matches.
    if any(not s.get("evidence_id") for s in sources):
        return {"evidence_recall": None, "evidence_precision": None}
    hits = len(actual & relevant)
    return {"evidence_recall": hits / len(relevant) if relevant else None,
            "evidence_precision": hits / len(actual) if actual else None}


def run(dataset: Path, namespace: str, document_id: str | None) -> dict:
    data = json.loads(dataset.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        q = item["question"]
        r = ask(q, namespace=namespace, document_id=document_id)
        src_texts = [s.get("preview") or s.get("quote") or "" for s in r.sources]
        row = {
            "question": q,
            "answer": r.answer[:400],
            "tier": r.tier,
            "confidence": r.confidence,
            "answer_relevancy": _overlap(q, r.answer),
            "keyword_recall": _keyword_recall(r.answer, item.get("expected_keywords", [])),
            "context_precision": _context_precision(r.answer, src_texts),
            "faithfulness_proxy": (r.confidence_components or {}).get("grounding") if not r.abstained else None,
            "reference_overlap": _overlap(r.answer, item["reference"]) if item.get("reference") else None,
            "answerable": r.answerable,
            "abstained": r.abstained,
            "abstention_reason": r.abstention_reason,
            "answerability_correct": (r.answerable == item["answerable"]) if "answerable" in item else None,
            **_evidence_metrics(r.sources, item.get("expected_evidence_ids")),
            "latency_ms": r.latency_ms,
        }
        rows.append(row)
        print(f"Q: {q[:80]!s}")
        for k in ("answer_relevancy", "keyword_recall", "context_precision", "faithfulness_proxy", "reference_overlap"):
            print(f"  {k}: {row[k] if row[k] is not None else 'unavailable'}")

    def mean(k):
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        return statistics.mean(vals) if vals else None

    summary = {
        "n": len(rows),
        "metric_kind": "lexical_proxies_and_labeled_evidence_overlap_not_ragas",
        "evidence_recall_mean": mean("evidence_recall"),
        "evidence_precision_mean": mean("evidence_precision"),
        "answerability_accuracy": mean("answerability_correct"),
        "answer_relevancy_mean": mean("answer_relevancy"),
        "keyword_recall_mean": mean("keyword_recall"),
        "context_precision_mean": mean("context_precision"),
        "faithfulness_proxy_mean": mean("faithfulness_proxy"),
        "reference_overlap_mean": mean("reference_overlap"),
        "latency_ms_p50": statistics.median([r["latency_ms"] for r in rows]) if rows else None,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--document-id", default=None)
    ap.add_argument("--out", type=Path, default=Path("cache/eval_report.json"))
    args = ap.parse_args()
    report = run(args.dataset, args.namespace, args.document_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport → {args.out}")


if __name__ == "__main__":
    main()
