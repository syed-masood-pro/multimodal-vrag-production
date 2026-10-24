"""Lightweight in-process cost / token tracking."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class CostTracker:
    _by_model: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    _lock: Lock = field(default_factory=Lock)

    def record(self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            u = self._by_model[model]
            u.calls += 1
            u.prompt_tokens += prompt_tokens
            u.completion_tokens += completion_tokens

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                m: {
                    "calls": u.calls,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                }
                for m, u in self._by_model.items()
            }


tracker = CostTracker()
