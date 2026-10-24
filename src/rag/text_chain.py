"""Text-only RAG answering with model selection."""
from __future__ import annotations

from src.clients.groq_client import chat
from src.core.logger import get_logger
from src.rag.prompts import RAG_SYSTEM_LOOSE, RAG_SYSTEM_STRICT, TEXT_RAG_PROMPT

log = get_logger(__name__)


def answer_text(*, model: str, context: str, question: str, strict: bool = True) -> str:
    system = RAG_SYSTEM_STRICT if strict else RAG_SYSTEM_LOOSE
    msgs = TEXT_RAG_PROMPT.format_messages(system=system, context=context, question=question)
    # Convert LangChain messages to plain dicts
    payload = [{"role": m.type if m.type != "human" else "user", "content": m.content} for m in msgs]
    # LC uses "system"/"human"/"ai" — map "human"→"user"
    for p in payload:
        if p["role"] == "human":
            p["role"] = "user"
    return chat(model=model, messages=payload, max_completion_tokens=800)
