"""Ingestion CLI: extract → embed → upsert (idempotent).

Usage:
    python -m scripts.ingest              # incremental (safe re-run)
    python -m scripts.ingest --reset      # wipe namespace first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import settings
from src.core.logger import get_logger
from src.extraction.pdf_extractor import extract_documents
from src.utils.hashing import sha256_file
from src.vectorstore.pinecone_store import clear_namespace, upsert_documents

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="Clear the namespace before upsert.")
    parser.add_argument("--pdf", type=str, default=str(settings.pdf_path))
    args = parser.parse_args()

    log.info("ingest_start", pdf=args.pdf, reset=args.reset)

    if args.reset:
        clear_namespace(namespace=settings.pinecone_namespace)

    document_id = sha256_file(args.pdf)
    docs = extract_documents(
        args.pdf, document_id=document_id, original_source=Path(args.pdf).name,
    )
    if not docs:
        log.error("no_documents_extracted")
        return 1

    n = upsert_documents(docs, namespace=settings.pinecone_namespace)
    log.info("ingest_done", upserted=n, document_id=document_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
