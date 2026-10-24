"""Groq client wrapper: retries, timeouts, cost tracking, unified chat/vision calls."""
from __future__ import annotations

from typing import Any

from groq import Groq

from config import settings
from src.core.cost_tracker import tracker
from src.core.exceptions import LLMError, VisionError
from src.core.logger import get_logger
from src.core.retry import network_retry

log = get_logger(__name__)

_client: Groq | None = None

def get_groq() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key, timeout=settings.request_timeout_secs)
    return _client

def _record_usage(model: str, response: Any) -> None:
    try:
        usage = getattr(response, "usage", None)
        if usage:
            inp = getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "completion_tokens", 0) or 0
            tracker.record(model, prompt_tokens=inp, completion_tokens=out)
            
            # Record non-streaming tokens to our global tenant usage bucket
            from src.core import usage as core_usage
            core_usage.add_tokens(inp, out)
    except Exception:  # pragma: no cover
        pass

@network_retry()
def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_completion_tokens: int = 800,
) -> str:
    try:
        resp = get_groq().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
    except Exception as e:
        log.error("groq_chat_failed", model=model, error=str(e))
        raise LLMError(f"Groq chat failed for model={model}: {e}") from e

    _record_usage(model, resp)
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise LLMError(f"Malformed Groq response: {e}") from e

@network_retry()
def vision_chat(
    *,
    model: str,
    prompt: str,
    image_data_uris: list[str],
    temperature: float = 0.0,
    max_completion_tokens: int = 900,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for uri in image_data_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})

    try:
        resp = get_groq().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
    except Exception as e:
        log.error("groq_vision_failed", model=model, error=str(e))
        raise VisionError(f"Groq vision failed for model={model}: {e}") from e

    _record_usage(model, resp)
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise VisionError(f"Malformed Groq vision response: {e}") from e