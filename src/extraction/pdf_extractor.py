"""Extraction pipeline v2:
- recursive chunking (not whole page)
- vision-LLM OCR fallback for scanned pages
- layout-aware vision (adds surrounding page text)
- PII redaction before embedding
- progress callback for background jobs
"""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Callable

import fitz
from langchain_core.documents import Document

from config import settings
from src.core.exceptions import ExtractionError
from src.core.logger import get_logger
from src.core.pii import redact
from src.extraction.chunking import chunk_text
from src.extraction.ocr import ocr_page
from src.extraction.vision_summarizer import summarize_visual
from src.utils.image_utils import is_worth_processing

log = get_logger(__name__)

ProgressCb = Callable[[int, int, str], None] | None


def _metadata(page: int, source: str, document_id: str, modality: str,
          locator: str, bbox=None) -> dict:
    identity = f"{document_id}\0{source}\0{page}\0{modality}\0{locator}"
    evidence_id = hashlib.sha256(identity.encode()).hexdigest()
    return {"evidence_id": evidence_id, "chunk_id": evidence_id,
        "document_id": document_id, "filename": Path(source).name,
        "source": source, "page": page, "modality": modality,
        "bbox": list(bbox) if bbox is not None else None,
        "location_precision": "bbox" if bbox is not None else "page"}


def _emit(cb: ProgressCb, done: int, total: int, stage: str) -> None:
    if cb:
        try:
            cb(done, total, stage)
        except Exception:
            pass


import re

def _text_docs(page, text: str, page_number: int, source: str, doc_id: str, textpage=None) -> list[Document]:
    if not text.strip():
        return []
    redacted, _hits = redact(text)
    docs: list[Document] = []
    
    for i, ck in enumerate(chunk_text(redacted), start=1):
        bbox = None
        try:
            # 1. Try a moderate snippet first (for perfect matches)
            snippet = ck.text[:35].strip()
            rects = textpage.search(snippet) if textpage else page.search_for(snippet)
            
            # 2. Foolproof Fallback: Search for the first substantial, unbroken alphanumeric word.
            # This completely ignores chopped words (like "cing"), ligatures, and line breaks.
            if not rects:
                # Extract all purely alphanumeric words longer than 4 characters
                words = [w for w in re.split(r'[^a-zA-Z0-9]+', ck.text) if len(w) > 4]
                if words:
                    # In this case, it will search for "manual" and anchor the box there
                    rects = textpage.search(words[0]) if textpage else page.search_for(words[0])

            if rects:
                bbox = list(rects[0])
        except Exception:
            pass

        docs.append(
            Document(
                page_content=ck.text,
                metadata={
                    **_metadata(page_number, source, doc_id, "text", str(i), bbox=bbox),
                    "page": page_number,
                    "modality": "text",
                    "chunk": i,
                    "source": source,
                    "document_id": doc_id,
                },
            )
        )
    return docs


def _table_docs(page, page_number: int, source: str, doc_id: str) -> list[Document]:
    docs: list[Document] = []
    try:
        tables = page.find_tables().tables
    except Exception as e:
        log.warning("table_extraction_warning", page=page_number, error=str(e))
        return docs
    for i, table in enumerate(tables, start=1):
        try:
            df = table.to_pandas()
            if df.empty:
                continue
            md, _ = redact(df.to_markdown(index=False))
            docs.append(
                Document(
                    page_content=md,
                    metadata={
                        **_metadata(page_number, source, doc_id, "table", str(i), table.bbox),
                        "page": page_number,
                        "modality": "table",
                        "table_number": i,
                        "source": source,
                        "document_id": doc_id,
                    },
                )
            )
        except Exception as e:
            log.warning("table_row_warning", page=page_number, table=i, error=str(e))
    return docs


def _image_docs(
    pdf, page, page_number: int, page_text: str, source: str, doc_id: str, seen: set[tuple[int, int]]
) -> list[Document]:
    docs: list[Document] = []
    for img_no, info in enumerate(page.get_images(full=True), start=1):
        xref = info[0]
        placement_key = (page_number, xref)
        if placement_key in seen:
            continue
        seen.add(placement_key)
        try:
            data = pdf.extract_image(xref)
            ext = data["ext"]
            path = settings.image_dir / f"{doc_id}_p{page_number}_i{img_no}.{ext}"
            path.write_bytes(data["image"])
        except Exception as e:
            log.warning("image_write_failed", page=page_number, error=str(e))
            continue
        if not is_worth_processing(path, settings.min_image_bytes, settings.min_image_dim):
            continue
        snippet, _ = redact((page_text or "").strip().replace("\n", " "))
        summary = summarize_visual(path, page_number, context_snippet=snippet[:600],
                                   parent_identity=f"{doc_id}:{source}:{page_number}:{xref}")
        summary, _ = redact(summary)
        if not summary.strip() or "could not summarize" in summary.lower():
            continue
        try:
            rects = list(page.get_image_rects(xref))
        except Exception:
            rects = []
        for placement, rect in enumerate(rects or [None], start=1):
            docs.append(Document(
                page_content=summary,
                metadata={**_metadata(page_number, source, doc_id, "visual",
                                      f"{xref}:{placement}", rect),
                          "image_path": str(path), "image_redacted": False,
                          "privacy_note": "Image pixels sent externally are not redacted."},
            ))
    return docs


def extract_documents(
    pdf_path: str | Path,
    *,
    document_id: str = "default",
    progress: ProgressCb = None,
    original_source: str | None = None,
) -> list[Document]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ExtractionError(f"PDF not found: {pdf_path}")
    try:
        pdf = fitz.open(str(pdf_path))
    except Exception as e:
        raise ExtractionError(f"Cannot open PDF: {e}") from e

    docs: list[Document] = []
    seen: set[tuple[int, int]] = set()
    total_pages = len(pdf)
    source = original_source or pdf_path.name
    if original_source is None and document_id != "default":
        # Uploads are decrypted to opaque temporary names. Resolve the registered
        # original filename without changing existing ingestion call signatures.
        try:
            from src.core.db import get_conn
            row = get_conn().execute("SELECT filename FROM documents WHERE id=?", (document_id,)).fetchone()
            if row and row["filename"]:
                source = row["filename"]
        except Exception as exc:
            log.warning("original_source_unavailable", error=str(exc))

    try:
        for idx in range(total_pages):
            page = pdf[idx]
            pn = idx + 1
            _emit(progress, pn - 1, total_pages, f"page {pn}: text")

            text = page.get_text("text").strip()
            textpage = None  # Track the layout-aware OCR page

            if len(text) < settings.ocr_min_chars_per_page:
                _emit(progress, pn - 1, total_pages, f"page {pn}: OCR")
                try:
                    # Force local Tesseract OCR on the open page
                    textpage = page.get_textpage_ocr(flags=0, dpi=180, full=True)
                    text = textpage.extractText().strip() or text
                except Exception as e:
                    log.warning("tesseract_ocr_failed", page=pn, error=str(e))
                    # Fall back to your old vision_chat if Tesseract fails
                    text = ocr_page(pdf_path, pn) or text

            # Pass the textpage down so the chunker can extract the bounding boxes
            docs.extend(_text_docs(page, text, pn, source, document_id, textpage=textpage))
            
            _emit(progress, pn - 1, total_pages, f"page {pn}: tables")
            docs.extend(_table_docs(page, pn, source, document_id))
            _emit(progress, pn - 1, total_pages, f"page {pn}: visuals")
            docs.extend(_image_docs(pdf, page, pn, text, source, document_id, seen))
            _emit(progress, pn, total_pages, f"page {pn} done")
    finally:
        pdf.close()

    log.info("extraction_complete", total_docs=len(docs), doc_id=document_id)
    return docs
