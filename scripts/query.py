"""Query CLI.

Usage:
    python -m scripts.query "your question"
"""
from __future__ import annotations

import argparse
import json
import sys

from config import settings
from src.core.cost_tracker import tracker
from src.core.logger import get_logger
from src.rag.pipeline import ask
from src.utils.hashing import sha256_file

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--json", action="store_true", help="Emit full JSON response.")
    parser.add_argument("--pdf", default=str(settings.pdf_path), help="PDF used to derive the same content ID as ingestion.")
    parser.add_argument("--document-id", help="Explicit ingested content ID (avoids reading the local PDF).")
    args = parser.parse_args()

    document_id = args.document_id or sha256_file(args.pdf)
    resp = ask(args.question, namespace=settings.pinecone_namespace, document_id=document_id)

    if args.json:
        print(json.dumps(resp.to_dict(), indent=2, default=str))
    else:
        print("\nQUESTION:", resp.question)
        print("\nANSWER:\n", resp.answer)
        print(
            f"\n[tier={resp.tier} model={resp.model} "
            f"confidence={resp.confidence} band={resp.confidence_band} "
            f"escalated={resp.escalated} latency_ms={resp.latency_ms}]"
        )
        print("\nSOURCES:")
        for i, s in enumerate(resp.sources, 1):
            print(f"  {i}. p.{s['page']} | {s['modality']}")
        print("\nCOST SNAPSHOT:", json.dumps(tracker.snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
