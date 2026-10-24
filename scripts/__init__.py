"""Fit a user-preference estimator, NOT a factual correctness probability.

Owned latest votes only; exact feature sets; conversation-disjoint heldout data.
Both splits must contain both labels. Runtime keeps this estimate separate from
heuristic confidence and grounding. No LLM/network calls or new dependencies.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.db import get_conn, loads
from config import settings
from src.rag.confidence import FEATURES, FEATURE_VERSION, valid_component


def collect(mode: str | None = None) -> list[dict]:
    mode = mode or settings.confidence_mode
    keys = FEATURES.get(mode)
    if not keys:
        return []
    rows = get_conn().execute(
        """
        SELECT m.id AS mid, m.meta_json, m.conversation_id, f.vote
        FROM feedback f
        JOIN messages m ON m.id = f.message_id
        JOIN conversations c ON c.id = m.conversation_id AND c.session_id = f.session_id
        WHERE m.role = 'assistant'
          AND (f.conversation_id IS NULL OR f.conversation_id = c.id)
          AND f.vote IN (-1, 1)
          AND NOT EXISTS (
              SELECT 1 FROM feedback newer
              WHERE newer.message_id = f.message_id AND newer.session_id = f.session_id
                AND (newer.conversation_id IS NULL OR newer.conversation_id = c.id)
                AND newer.vote IN (-1, 1)
                AND (newer.created_at > f.created_at OR
                     (newer.created_at = f.created_at AND newer.id > f.id))
          )
        ORDER BY m.id
        """
    ).fetchall()
    out = []
    for r in rows:
        try:
            meta = loads(r["meta_json"]) or {}
        except (ValueError, TypeError):
            continue
        if not isinstance(meta, dict):
            continue
        comps = meta.get("confidence_components") or {}
        calibration = meta.get("calibration") or {}
        if (not isinstance(comps, dict) or set(comps) != keys
                or not all(valid_component(v) for v in comps.values())
                or meta.get("abstained") or meta.get("tier") == "vision"
                or not isinstance(calibration, dict)
                or calibration.get("feature_version") != FEATURE_VERSION
                or calibration.get("mode") != mode):
            continue
        conf = meta.get("confidence")  # Preserve a real zero; never fabricate 0.5.
        out.append({"mid": r["mid"], "group": r["conversation_id"], "comps": comps,
                    "vote": 1 if r["vote"] > 0 else 0,
                    "conf": conf if valid_component(conf) else None})
    return out


def fit_artifact(data: list[dict], mode: str) -> dict:
    keys = sorted(FEATURES[mode])
    if len(data) < 20 or len({d["mid"] for d in data}) != len(data):
        raise ValueError("Need at least 20 unique owned feedback messages")
    if any(set(d["comps"]) != set(keys) or d["vote"] not in (0, 1)
           or not all(valid_component(v) for v in d["comps"].values()) for d in data):
        raise ValueError("Invalid or incompatible features")
    if len({tuple(d["comps"][k] for k in keys) for d in data}) < 4:
        raise ValueError("Insufficient feature diversity")
    groups = sorted({d["group"] for d in data})
    if len(groups) < 4:
        raise ValueError("Need at least four distinct conversations")
    random.Random(42).shuffle(groups)
    heldout_groups = set(groups[:max(1, len(groups) // 4)])
    train = [d for d in data if d["group"] not in heldout_groups]
    heldout = [d for d in data if d["group"] in heldout_groups]
    if any({d["vote"] for d in split} != {0, 1} for split in (train, heldout)):
        raise ValueError("Both train and heldout splits require positive and negative votes")
    weights, intercept = {k: 0.0 for k in keys}, 0.0

    def predict(d):
        z = intercept + sum(weights[k] * d["comps"][k] for k in keys)
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    for _ in range(800):
        errors = [(d, predict(d) - d["vote"]) for d in train]
        intercept -= 0.15 * sum(e for _, e in errors) / len(train)
        for k in keys:
            gradient = sum(e * d["comps"][k] for d, e in errors) / len(train)
            weights[k] -= 0.15 * (gradient + 0.01 * weights[k])

    def metrics(split):
        positive = sum(d["vote"] for d in split)
        return {"n": len(split), "positive": positive, "negative": len(split) - positive,
                "brier": sum((predict(d) - d["vote"])**2 for d in split) / len(split),
                "accuracy": sum((predict(d) >= 0.5) == bool(d["vote"]) for d in split) / len(split)}

    return {"schema_version": 2, "feature_version": FEATURE_VERSION, "mode": mode,
            "target": "user_preference", "method": "logistic", "keys": keys,
            "component_weights": weights, "intercept": intercept, "n_samples": len(data),
            "split": "conversation_disjoint_seed_42", "metrics": {
                "train": metrics(train), "heldout": metrics(heldout)}}


def main():
    try:
        out = fit_artifact(collect(), settings.confidence_mode)
    except (ValueError, KeyError) as exc:
        print(f"Calibration unchanged: {exc}")
        return
    dst = settings.cache_dir / "confidence_weights.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dst.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(out, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, dst)
    finally:
        Path(name).unlink(missing_ok=True)
    print(f"Preference calibration written to {dst}. Not a correctness probability.")
    print(out["metrics"])


if __name__ == "__main__":
    main()
