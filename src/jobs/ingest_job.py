"""Ingest job: extract → dense upsert → atomic corpus/ready publication."""
from __future__ import annotations

import uuid
import tempfile
from pathlib import Path

import fitz

from config import settings
from src.core.crypto import decrypt_file
from src.core.db import namespace_owner, now, tx
from src.core import usage
from src.core.logger import get_logger
from src.extraction.pdf_extractor import extract_documents
from src.jobs.runner import JobCancelled, checkpoint, initialize_runner, update_job
from src.rag.hybrid import bm25_store
from src.vectorstore.pinecone_store import delete_document, upsert_documents

log = get_logger(__name__)


def _write_temp_pdf(encrypted_path: Path) -> Path:
    raw = decrypt_file(encrypted_path)
    with tempfile.NamedTemporaryFile(prefix="vrag-", suffix=".pdf", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(raw)
        except BaseException:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    return tmp


def run_ingest(
    *,
    job_id: str,
    session_id: str,
    document_id: str,
    encrypted_pdf_path: str,
    namespace: str,
    filename: str,
) -> None:
    tmp_pdf: Path | None = None
    page_reservation: str | None = None
    published = False
    upsert_started = False
    page_count = 0
    current_page = 0
    owned = False

    def progress(done: int, total: int, stage: str) -> None:
        nonlocal current_page
        checkpoint(job_id)
        current_page = min(done + 1, total)
        update_job(job_id, progress=done, total=total, stage=stage)

    try:
        with tx() as conn:
            checkpoint(job_id)
            row = conn.execute("SELECT * FROM documents WHERE id=? AND session_id=?", (document_id, session_id)).fetchone()
            job = conn.execute("SELECT * FROM jobs WHERE id=? AND session_id=?", (job_id, session_id)).fetchone()
            if (not row or not job or namespace_owner(namespace) != session_id
                    or row["job_id"] != job_id or job["document_id"] != document_id):
                raise ValueError("Ingestion ownership mismatch")
            owned = True
            # Registered paths/names are authoritative, never trust worker arguments.
            encrypted_pdf_path, filename = row["storage_path"], row["filename"]
            conn.execute("UPDATE documents SET status='processing',attempt_count=attempt_count+1,"
                         "last_error=NULL,updated_at=? WHERE id=?", (now(), document_id))
        update_job(job_id, stage="decrypting")
        tmp_pdf = _write_temp_pdf(Path(encrypted_pdf_path))
        checkpoint(job_id)
        with fitz.open(str(tmp_pdf)) as pdf:
            page_count = len(pdf)
        page_reservation = usage.reserve_pages(session_id, page_count, document_id)
        checkpoint(job_id)

        update_job(job_id, stage="extracting")
        docs = extract_documents(tmp_pdf, document_id=document_id, progress=progress, original_source=filename)
        docs = [d for d in docs if d.page_content.strip()]
        checkpoint(job_id)
        if not docs:
            raise ValueError("No usable extracted content")
        missing_pages = sorted(set(range(1, page_count + 1)) - {d.metadata.get("page") for d in docs})

        update_job(job_id, stage="embedding_upserting")
        # An interrupted prior attempt can have remote vectors but no published corpus.
        upsert_started = True
        delete_document(namespace, document_id)
        checkpoint(job_id)
        n = upsert_documents(docs, namespace=namespace)
        if n != len(docs):
            raise ValueError("Incomplete vector publication")
        checkpoint(job_id)
        # Publishing corpus, ready state and successful job is one atomic local commit.
        with tx() as conn:
            checkpoint(job_id)
            bm25_store.upsert(namespace, document_id, docs)
            conn.execute("UPDATE documents SET pages=?,chunks=?,status='ready',last_error=NULL,updated_at=? "
                         "WHERE id=? AND session_id=? AND job_id=?", (page_count, n, now(), document_id, session_id, job_id))
            update_job(job_id, status="done", stage="done", progress=page_count, total=page_count,
                       result={"document_id": document_id, "filename": filename, "chunks": n,
                               "pages": page_count, "pages_without_content": missing_pages,
                               "warnings": ["Some pages produced no usable content."] if missing_pages else []})
            usage.finish_pages(page_reservation, True)
        published = True
    except (Exception, JobCancelled) as exc:
        if not owned:
            raise  # Never mutate another tenant's job/document while reporting a bad invocation.
        cancelled = isinstance(exc, JobCancelled)
        message = "Ingestion cancelled." if cancelled else "Ingestion failed. Retry or contact the administrator."
        if isinstance(exc, usage.BudgetExceeded):
            message = str(exc)
        elif isinstance(exc, ValueError) and str(exc) == "No usable extracted content":
            message = "No usable content was extracted from this PDF."
        cleanup_failed = False
        if not published:
            try:
                bm25_store.delete(namespace, document_id)
                if upsert_started:
                    delete_document(namespace, document_id)
            except Exception:
                cleanup_failed = True
                log.exception("unpublished_cleanup_failed", document_id=document_id)
            try:
                for image in settings.image_dir.iterdir():
                    if image.is_file() and image.name.startswith(f"{document_id}_p"):
                        image.unlink()
            except OSError:
                cleanup_failed = True
                log.warning("unpublished_image_cleanup_failed", document_id=document_id)
            if cleanup_failed:
                message += " Remote cleanup requires retry; document remains unavailable."
            with tx() as conn:
                status = "error" if cleanup_failed or not cancelled else "cancelled"
                conn.execute("UPDATE documents SET status=?,last_error=?,updated_at=? "
                             "WHERE id=? AND session_id=? AND job_id=? AND status!='deleting'",
                             (status, message, now(), document_id, session_id, job_id))
                update_job(job_id, status=status, stage=status, message=message,
                           result={"document_id": document_id, "failed_page": current_page or None,
                                   "cleanup_required": cleanup_failed})
                if page_reservation:
                    usage.finish_pages(page_reservation, False)
        raise
    finally:
        if tmp_pdf and tmp_pdf.exists():
            try:
                tmp_pdf.unlink()
            except Exception:
                log.warning("temporary_pdf_cleanup_failed", document_id=document_id)


def new_document_id() -> str:
    return uuid.uuid4().hex


def register_document(
    *,
    session_id: str,
    filename: str,
    storage_path: str,
    size_bytes: int,
    sha256: str,
    reservation_id: str | None = None,
    document_id: str | None = None,
) -> str:
    initialize_runner()
    doc_id = document_id or new_document_id()
    if size_bytes < 0:
        raise ValueError("size_bytes must be nonnegative")
    with tx() as conn:
        row = conn.execute("SELECT MAX(version) AS v FROM documents WHERE session_id=? AND filename=?",
                           (session_id, filename)).fetchone()
        version = int((row["v"] or 0) + 1)
        conn.execute("INSERT INTO documents(id,session_id,filename,version,size_bytes,storage_path,sha256,created_at,status,updated_at) "
                     "VALUES(?,?,?,?,?,?,?,?,'pending',?)",
                     (doc_id, session_id, filename, version, size_bytes, storage_path, sha256, now(), now()))
        if reservation_id is not None:
            usage.finish_upload(reservation_id, doc_id)
    return doc_id
