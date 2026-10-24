"""Owned PDF upload, lifecycle listing/deletion and private evidence rendering."""
from __future__ import annotations

import math
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import fitz
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from config import settings
from src.api.deps import session_ctx
from src.api.schemas import DocumentItem, SourceItem
from src.core import usage
from src.core.audit import log as audit_log
from src.core.crypto import decrypt_file, encrypt_file
from src.core.db import get_conn, loads, now, tx
from src.core.file_check import assert_pdf
from src.core.logger import get_logger
from src.core.rate_limit import limiter
from src.jobs.ingest_job import register_document, run_ingest
from src.jobs.runner import QueueFull, create_job, submit
from src.rag.hybrid import bm25_store
from src.utils.hashing import sha256_bytes
from src.vectorstore.pinecone_store import delete_document

log = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PAGE_POINTS = 14400
MAX_RENDER_DIMENSION = 4096
MAX_RENDER_PIXELS = 8_000_000
_DELETE_LOCK = threading.Lock()  # one-process serialization of delete retries


def _validate_pdf(raw: bytes) -> int:
    try:
        assert_pdf(raw[:4096])
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            if pdf.needs_pass or len(pdf) < 1:
                raise HTTPException(400, "PDF must be readable and not password protected.")
            if len(pdf) > settings.max_pdf_pages:
                raise HTTPException(413, "PDF exceeds the page limit.")
            # Bounded page-tree validation: no extraction, rasterization or providers.
            for page in pdf:
                rect = page.rect
                if not all(math.isfinite(v) for v in rect) or min(rect.width, rect.height) <= 0:
                    raise HTTPException(400, "PDF contains an invalid page.")
                if max(rect.width, rect.height) > MAX_PAGE_POINTS:
                    raise HTTPException(413, "PDF page dimensions exceed the limit.")
            return len(pdf)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Uploaded file is not a readable PDF.") from exc


@router.post("/upload", status_code=202)
@limiter.limit(settings.rate_upload)
async def upload_document(request: Request, file: UploadFile = File(...),
                          ctx=Depends(session_ctx)) -> dict[str, Any]:
    try:
        name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if (not name.lower().endswith(".pdf") or len(name) > 255
                or any(ord(c) < 32 for c in name)):
            raise HTTPException(400, "Only .pdf files with a valid filename are accepted.")
        buf = bytearray()
        while chunk := await file.read(1 << 20):
            if len(buf) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File exceeds 50 MB limit.")
            buf.extend(chunk)
        raw = bytes(buf)
        del buf
        _validate_pdf(raw)
        sid = ctx["session_id"]
        reservation = usage.reserve_upload(sid, len(raw))
        doc_id = uuid.uuid4().hex
        enc_path = settings.upload_dir / sid / f"{doc_id}.pdf.enc"
        try:
            encrypt_file(raw, enc_path)
            register_document(session_id=sid, filename=name, storage_path=str(enc_path),
                              size_bytes=len(raw), sha256=sha256_bytes(raw),
                              document_id=doc_id, reservation_id=reservation)
        except Exception as exc:
            try:
                enc_path.unlink(missing_ok=True)
            finally:
                usage.finish_upload(reservation, None)
            log.exception("upload_registration_failed")
            raise HTTPException(500, "Unable to store the uploaded PDF.") from exc

        job_id = None
        try:
            job_id = create_job(sid, "ingest", document_id=doc_id)
            # Submit before returning: QueueFull must be observable by the dashboard.
            submit(job_id, run_ingest, job_id=job_id, session_id=sid, document_id=doc_id,
                   encrypted_pdf_path=str(enc_path), namespace=ctx["namespace"], filename=name)
        except Exception as exc:
            message = ("Ingestion queue is full. Retry later." if isinstance(exc, QueueFull)
                       else "Ingestion could not start. Retry later.")
            with tx() as conn:
                conn.execute("UPDATE documents SET status='error',last_error=?,updated_at=? "
                             "WHERE id=? AND status='pending'", (message, now(), doc_id))
                if job_id:
                    conn.execute("UPDATE jobs SET status='error',stage='error',message=?,updated_at=? "
                                 "WHERE id=? AND status='queued'", (message, now(), job_id))
            if not isinstance(exc, QueueFull):
                log.exception("upload_submission_failed")
            raise HTTPException(503, {"message": message, "document_id": doc_id, "job_id": job_id},
                                headers={"Retry-After": "5"}) from exc
        audit_log("document.upload", session_id=sid, resource=doc_id,
                  ip=request.client.host if request.client else None,
                  meta={"filename": name, "size": len(raw)})
        return {"document_id": doc_id, "job_id": job_id, "filename": name,
                "size_bytes": len(raw), "status": "pending"}
    finally:
        await file.close()


@router.get("", response_model=list[DocumentItem])
def list_documents(ctx=Depends(session_ctx)) -> list[DocumentItem]:
    rows = get_conn().execute("SELECT * FROM documents WHERE session_id=? ORDER BY created_at DESC,id",
                              (ctx["session_id"],)).fetchall()
    return [DocumentItem(**dict(row)) for row in rows]


@router.delete("/{document_id}")
def delete_doc(document_id: str, ctx=Depends(session_ctx)) -> dict[str, str]:
    with _DELETE_LOCK:
        with tx() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=? AND session_id=?",
                               (document_id, ctx["session_id"])).fetchone()
            if not row:
                raise HTTPException(404, "Document not found")
            active = conn.execute("SELECT 1 FROM jobs WHERE document_id=? AND status IN ('queued','running')",
                                  (document_id,)).fetchone()
            if active or row["status"] in {"pending", "processing"}:
                raise HTTPException(409, "Ingestion is active. Cancel it and wait for a terminal state before deleting.")
            # Commit denial of retrieval BEFORE any remote operation.
            conn.execute("UPDATE documents SET status='deleting',last_error=NULL,updated_at=? WHERE id=?",
                         (now(), document_id))
        try:
            delete_document(ctx["namespace"], document_id)
            if settings.image_dir.exists():
                for image in settings.image_dir.iterdir():
                    if image.is_file() and image.name.startswith(f"{document_id}_p"):
                        image.unlink()
            bm25_store.delete(ctx["namespace"], document_id)
            Path(row["storage_path"]).unlink(missing_ok=True)
            with tx() as conn:
                conn.execute("DELETE FROM documents WHERE id=? AND session_id=?", (document_id, ctx["session_id"]))
                audit_log("document.delete", session_id=ctx["session_id"], resource=document_id)
        except Exception as exc:
            log.exception("document_delete_failed", document_id=document_id)
            with tx() as conn:
                conn.execute("UPDATE documents SET last_error=?,updated_at=? WHERE id=? AND session_id=?",
                             ("Deletion incomplete. Retry deletion.", now(), document_id, ctx["session_id"]))
            raise HTTPException(502, "Deletion incomplete. Retry deletion.") from exc
    return {"status": "deleted"}


def _ready_owned(document_id: str, ctx: dict) -> dict:
    row = get_conn().execute("SELECT * FROM documents WHERE id=? AND session_id=? AND status='ready'",
                             (document_id, ctx["session_id"])).fetchone()
    if not row:
        raise HTTPException(404, "Ready document not found")
    return dict(row)


def _source_bbox(meta: dict) -> list[float] | None:
    bbox = meta.get("bbox")
    # Text/OCR has page-level evidence; don't fabricate word-precise highlighting.
    if (meta.get("modality") in {"table", "visual", "image"}
            and isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox)
            and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        return list(bbox)
    return None


def _evidence(document: dict, evidence_id: str, ctx: dict) -> dict:
    row = get_conn().execute(
        "SELECT chunk_id,content,metadata_json FROM document_chunks WHERE namespace=? AND document_id=? "
        "AND (json_extract(metadata_json,'$.evidence_id')=? OR json_extract(metadata_json,'$.chunk_id')=? "
        "OR chunk_id=?) ORDER BY chunk_id LIMIT 1",
        (ctx["namespace"], document["id"], evidence_id, evidence_id, evidence_id),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Evidence not found")
    meta = loads(row["metadata_json"])
    page = meta.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise HTTPException(404, "Evidence page not found")
    preview = str(meta.get("evidence_preview") or row["content"])[:600]
    result = SourceItem(label=meta.get("label"), evidence_id=meta.get("evidence_id") or evidence_id,
                        chunk_id=meta.get("chunk_id") or row["chunk_id"], document_id=document["id"],
                        filename=document["filename"], source=document["filename"], page=page,
                        modality=meta.get("modality"), bbox=_source_bbox(meta), quote=preview,
                        preview=preview, cited=False).model_dump()
    result["page_image_url"] = (f"/api/documents/{quote(document['id'], safe='')}/pages/{page}.png"
                                f"?evidence_id={quote(evidence_id, safe='')}")
    return result


@router.get("/{document_id}/evidence/{evidence_id}")
def get_evidence(document_id: str, evidence_id: str, response: Response,
                 ctx=Depends(session_ctx)) -> dict:
    response.headers["Cache-Control"] = "private, no-store"
    with tx():
        return _evidence(_ready_owned(document_id, ctx), evidence_id, ctx)


@router.get("/{document_id}/pages/{page}.png")
def render_page(document_id: str, page: int, evidence_id: str | None = None,
                ctx=Depends(session_ctx)) -> Response:
    with tx():
        row = _ready_owned(document_id, ctx)
        evidence = _evidence(row, evidence_id, ctx) if evidence_id is not None else None
        if evidence and evidence["page"] != page:
            raise HTTPException(404, "Evidence is not on this page")
    if page < 1:
        raise HTTPException(404, "Page out of range")
    try:
        path = Path(row["storage_path"])
        if path.stat().st_size > MAX_UPLOAD_BYTES * 2:
            raise HTTPException(413, "Stored PDF exceeds the rendering limit")
        raw = decrypt_file(path)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Stored PDF exceeds the rendering limit")
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            if pdf.needs_pass or page > len(pdf):
                raise HTTPException(404, "Page out of range")
            pdf_page = pdf[page - 1]
            width, height = pdf_page.rect.width, pdf_page.rect.height
            if (not all(math.isfinite(v) and 0 < v <= MAX_PAGE_POINTS for v in (width, height))):
                raise HTTPException(413, "PDF page dimensions exceed the rendering limit")
            scale = min(150 / 72, (MAX_RENDER_DIMENSION - 2) / width,
                        (MAX_RENDER_DIMENSION - 2) / height,
                        math.sqrt((MAX_RENDER_PIXELS - 20000) / (width * height)))
            if math.ceil(width * scale) * math.ceil(height * scale) > MAX_RENDER_PIXELS:
                raise HTTPException(413, "PDF page exceeds the rendering pixel limit")
            if evidence and evidence["bbox"] is not None:
                rect = fitz.Rect(evidence["bbox"]) & (pdf_page.rect * pdf_page.derotation_matrix)
                if not rect.is_empty:
                    pdf_page.draw_rect(rect, color=(1, .55, 0), fill=(1, .85, 0),
                                       fill_opacity=.2, width=1.5, overlay=True)
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            png = pix.tobytes("png")
        # A concurrent deletion must not publish an image after marking unavailable.
        _ready_owned(document_id, ctx)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, no-store"})
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("render_page_failed", document_id=document_id)
        raise HTTPException(500, "Failed to render page") from exc
