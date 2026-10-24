"""Owned document lifecycle and deterministic, bounded corpus comparison.

Include router BEFORE the existing document router; include usage_router separately.
Call initialize_runner() from application lifespan before serving requests.
"""
from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import session_ctx
from src.api.schemas import (
    DocumentCancelResponse, DocumentCompareResponse, DocumentDetailResponse,
    DocumentRetryResponse, JobItem, UsageSnapshotResponse,
)
from src.core import usage
from src.core.db import get_conn, tx
from src.jobs.ingest_job import run_ingest
from src.jobs.runner import _terminal, create_job, get_job, initialize_runner, request_cancel, submit, QueueFull

router = APIRouter(prefix="/api/documents", tags=["document lifecycle"])
usage_router = APIRouter(prefix="/api", tags=["usage"])


def _owned(document_id: str, session_id: str):
    row = get_conn().execute("SELECT * FROM documents WHERE id=? AND session_id=?", (document_id, session_id)).fetchone()
    if not row:
        raise HTTPException(404, "Document not found")
    return dict(row)


def _public(doc: dict) -> dict:
    return {k: doc[k] for k in ("id", "filename", "version", "size_bytes", "pages", "chunks",
                                "status", "created_at", "updated_at", "attempt_count", "last_error", "job_id")}


def _page_text(namespace: str, document_id: str, page: int, cap: int) -> tuple[str, str, bool]:
    # Stream content while hashing the complete page: suffix-only changes remain visible
    # even when the returned preview is truncated. No full-page concatenation in memory.
    cursor = get_conn().execute(
        "SELECT content FROM document_chunks WHERE namespace=? AND document_id=? "
        "AND CAST(json_extract(metadata_json,'$.page') AS INTEGER)=? "
        "ORDER BY json_extract(metadata_json,'$.modality'),"
        "CAST(json_extract(metadata_json,'$.chunk') AS INTEGER),"
        "CAST(json_extract(metadata_json,'$.table_number') AS INTEGER),content,chunk_id", (namespace, document_id, page))
    digest = hashlib.sha256()
    parts, length = [], 0
    first = True
    for row in cursor:
        text = ("" if first else "\n\n") + row["content"]
        first = False
        digest.update(text.encode("utf-8"))
        if length < cap:
            parts.append(text[:cap - length])
        length += len(text)
    return "".join(parts), digest.hexdigest(), length > cap


@router.get("/compare", response_model=DocumentCompareResponse)
def compare_documents(left_id: str, right_id: str,
                      offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50),
                      max_chars: int = Query(40000, ge=1, le=200000), ctx=Depends(session_ctx)) -> dict:
    """offset/limit paginate union pages; max_chars bounds TOTAL returned text characters."""
    with tx() as conn:
        left, right = _owned(left_id, ctx["session_id"]), _owned(right_id, ctx["session_id"])
        if left["status"] != "ready" or right["status"] != "ready":
            raise HTTPException(409, "Both documents must be ready")
        page_rows = conn.execute(
            "SELECT DISTINCT CAST(json_extract(metadata_json,'$.page') AS INTEGER) AS page "
            "FROM document_chunks WHERE namespace=? AND document_id IN (?,?) ORDER BY page",
            (ctx["namespace"], left_id, right_id)).fetchall()
        page_numbers = [r["page"] for r in page_rows if r["page"] is not None and r["page"] >= 1]
        selected = page_numbers[offset:offset + limit]
        pages, remaining, clipped = [], max_chars, False
        for index, page in enumerate(selected):
            # Fair share per page/side avoids a long first page hiding every later page.
            allowance = remaining // max(1, 2 * (len(selected) - index))
            lt, lh, lc = _page_text(ctx["namespace"], left_id, page, allowance)
            rt, rh, rc = _page_text(ctx["namespace"], right_id, page, allowance)
            remaining -= len(lt) + len(rt)
            clipped = clipped or lc or rc
            pages.append({"page": page, "left_text": lt, "right_text": rt, "changed": lh != rh})
        next_offset = offset + len(selected) if offset + len(selected) < len(page_numbers) else None
        return {"left": {k: left[k] for k in ("id", "filename", "version")},
                "right": {k: right[k] for k in ("id", "filename", "version")},
                "pages": pages, "truncated": clipped or offset > 0 or next_offset is not None,
                "offset": offset, "limit": limit, "max_chars": max_chars,
                "total_pages": len(page_numbers), "next_offset": next_offset,
                "text_truncated": clipped}


@router.get("/{document_id}/detail", response_model=DocumentDetailResponse)
def document_detail(document_id: str, ctx=Depends(session_ctx),
                    offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=100)) -> dict:
    with tx() as conn:
        doc = _owned(document_id, ctx["session_id"])
        total = conn.execute("SELECT COUNT(*) FROM jobs WHERE document_id=? AND session_id=?",
                             (document_id, ctx["session_id"])).fetchone()[0]
        rows = conn.execute("SELECT id FROM jobs WHERE document_id=? AND session_id=? "
                            "ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                            (document_id, ctx["session_id"], limit, offset)).fetchall()
        # Use the same public projection as job polling/SSE, never raw database rows.
        jobs = [JobItem(**job) for r in rows if (job := get_job(r["id"])) is not None]
        next_offset = offset + len(rows) if offset + len(rows) < total else None
        return {"document": _public(doc), "jobs": jobs, "offset": offset, "limit": limit,
                "total_jobs": total, "next_offset": next_offset}


@router.post("/{document_id}/retry", status_code=202, response_model=DocumentRetryResponse)
def retry_document(document_id: str, ctx=Depends(session_ctx)) -> dict:
    initialize_runner()
    with tx():
        doc = _owned(document_id, ctx["session_id"])
        if doc["status"] not in {"error", "cancelled"}:
            raise HTTPException(409, "Only error or cancelled documents can be retried")
        try:
            job_id = create_job(ctx["session_id"], "ingest", document_id=document_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    try:
        submit(job_id, run_ingest, job_id=job_id, session_id=ctx["session_id"],
               document_id=document_id, encrypted_pdf_path=doc["storage_path"],
               namespace=ctx["namespace"], filename=doc["filename"])
    except QueueFull as exc:
        message = "Ingestion queue is full. Retry later."
        _terminal(job_id, "error", message)
        raise HTTPException(503, {"message": message, "job_id": job_id}, headers={"Retry-After": "5"}) from exc
    except Exception as exc:
        message = "Job executor is unavailable. Retry later."
        # The runner may already have failed, completed or cancelled this job.
        # Its guarded transition preserves terminal states and newer document jobs.
        _terminal(job_id, "error", message)
        raise HTTPException(503, {"message": message, "job_id": job_id}, headers={"Retry-After": "5"}) from exc
    return {"document_id": document_id, "job_id": job_id, "status": "pending"}


@router.post("/{document_id}/cancel", status_code=202, response_model=DocumentCancelResponse)
def cancel_document(document_id: str, ctx=Depends(session_ctx)) -> dict:
    with tx():
        doc = _owned(document_id, ctx["session_id"])
        if doc["status"] not in {"pending", "processing"}:
            raise HTTPException(409, "Document has no cancellable ingestion")
        if not doc["job_id"] or not request_cancel(doc["job_id"], ctx["session_id"]):
            raise HTTPException(409, "Document has no active job")
    return {"document_id": document_id, "job_id": doc["job_id"], "cancel_requested": True}


@usage_router.get("/usage", response_model=UsageSnapshotResponse)
def tenant_usage(ctx=Depends(session_ctx)) -> dict:
    return usage.snapshot(ctx["session_id"])