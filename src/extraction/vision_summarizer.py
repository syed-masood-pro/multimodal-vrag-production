"""Scoped redacted summaries. Image pixels sent to the provider are NOT redacted."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from config import settings
from src.clients.groq_client import vision_chat
from src.core.exceptions import VisionError
from src.core.logger import get_logger
from src.core.pii import redact
from src.utils.hashing import sha256_file
from src.utils.image_utils import image_to_data_uri

log = get_logger(__name__)

_CACHE_FILE = settings.cache_dir / "vision_summaries.json"
_LOCK = threading.RLock()
_PROMPT_VERSION = "evidence-v2"


@contextmanager
def _cache_lock():
    """Thread + process lock; bounded wait, fail closed on abandoned lock."""
    with _LOCK:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock = _CACHE_FILE.with_suffix(".lock")
        deadline = time.monotonic() + 10
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Vision cache lock unavailable")
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)


def _load_cache() -> dict[str, str]:
    if _CACHE_FILE.exists():
        try:
            value = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(value, dict) else {}
        except Exception:
            log.warning("vision_cache_corrupt_reset")
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    # Caller holds _cache_lock across read/merge/replace to avoid lost updates.
    fd, name = tempfile.mkstemp(dir=_CACHE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(cache, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, _CACHE_FILE)
    finally:
        Path(name).unlink(missing_ok=True)


_PROMPT = """This visual was extracted from page {page} of a business PDF report.

Nearby page text (for context; may be empty):
{ctx}

Describe the useful business information concisely and factually.
If chart/graph: highlight key values and the main trend.
If diagram: identify components and relationships.
If photo: describe factual content."""


def summarize_visual(
    image_path: str | Path, page_number: int, context_snippet: str = "",
    *, parent_identity: str | None = None,
) -> str:
    image_path = Path(image_path)
    context_snippet, _ = redact(context_snippet)
    scope = [parent_identity or str(image_path.resolve()), sha256_file(image_path),
             page_number, context_snippet, settings.model_vision, _PROMPT_VERSION, _PROMPT]
    key = hashlib.sha256(json.dumps(scope).encode()).hexdigest()

    cache = _load_cache()
    if key in cache:
        log.debug("vision_cache_hit", key=key[:8])
        return redact(cache[key])[0]

    try:
        summary = vision_chat(
            model=settings.model_vision,
            prompt=_PROMPT.format(page=page_number, ctx=context_snippet or "(none)"),
            image_data_uris=[image_to_data_uri(image_path)],
            max_completion_tokens=500,
        )
    except VisionError as e:
        log.warning("vision_summary_failed", page=page_number, error=str(e))
        return f"[Visual on page {page_number}: could not summarize]"

    summary, _ = redact(summary)
    try:
        with _cache_lock():
            cache = _load_cache()
            cache[key] = summary
            _save_cache(cache)
    except (OSError, TimeoutError) as e:
        log.warning("vision_cache_write_failed", error=str(e))
    return summary
