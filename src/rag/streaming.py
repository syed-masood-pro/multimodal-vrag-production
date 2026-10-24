"""Streaming answer generation via Groq stream=True."""
from __future__ import annotations

from typing import Iterator

from src.clients.groq_client import get_groq
from src.core.exceptions import LLMError
from src.core.logger import get_logger
from src.core import usage

log = get_logger(__name__)

def stream_chat(
    *,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_completion_tokens: int = 800,
) -> Iterator[str]:
    
    fallback_prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
    manual_output_chars = 0
    received_usage = False

    try:
        stream = get_groq().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception as e:
        raise LLMError(f"stream open failed: {e}") from e

    try:
        for chunk in stream:
            usage_data = getattr(chunk, "usage", None)
            if not usage_data and getattr(chunk, "model_extra", None):
                usage_data = chunk.model_extra.get("x_groq", {}).get("usage")

            if usage_data and not received_usage:
                received_usage = True
                if isinstance(usage_data, dict):
                    usage.add_tokens(usage_data.get("prompt_tokens", 0) or 0, usage_data.get("completion_tokens", 0) or 0)
                else:
                    usage.add_tokens(getattr(usage_data, "prompt_tokens", 0) or 0, getattr(usage_data, "completion_tokens", 0) or 0)

            try:
                delta = chunk.choices[0].delta.content if chunk.choices else ""
            except Exception:
                delta = ""
                
            if delta:
                manual_output_chars += len(delta)
                yield delta

        if not received_usage:
            usage.add_tokens(max(1, fallback_prompt_tokens), max(1, manual_output_chars // 4))

    except Exception as e:
        log.warning("stream_iter_error", error=str(e))
        raise LLMError(f"stream iteration failed: {e}") from e