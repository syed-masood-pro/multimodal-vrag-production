"""Thin demo entrypoint.

Run once to ingest, then ask questions. Safe to re-run: ingestion is idempotent.

    python main.py
"""
from __future__ import annotations

from config import settings
from src.core.logger import get_logger
from src.extraction.pdf_extractor import extract_documents
from src.rag.pipeline import ask
from src.utils.hashing import sha256_file
from src.vectorstore.pinecone_store import upsert_documents

log = get_logger(__name__)


def ingest_if_needed() -> None:
    log.info("ingesting", pdf=str(settings.pdf_path))
    document_id = sha256_file(settings.pdf_path)
    docs = extract_documents(
        settings.pdf_path, document_id=document_id, original_source=settings.pdf_path.name,
    )
    upsert_documents(docs, namespace=settings.pinecone_namespace)


def demo() -> None:
    document_id = sha256_file(settings.pdf_path)
    questions = [
        "What does the report say and where is the company headquartered?",
        "According to the revenue graph, which quarter had the highest revenue?",
        "According to the supply-chain diagram, what is the critical quality-control point?",
    ]
    for q in questions:
        r = ask(q, namespace=settings.pinecone_namespace, document_id=document_id)
        print("=" * 80)
        print("Q:", r.question)
        print("A:", r.answer)
        print(f"[tier={r.tier} conf={r.confidence} band={r.confidence_band}]")


if __name__ == "__main__":
    ingest_if_needed()
    demo()
