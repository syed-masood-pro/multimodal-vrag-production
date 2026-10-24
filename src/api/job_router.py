"""Job routes: status polling + SSE stream of progress."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.deps import session_ctx
from src.api.schemas import JobItem
from src.jobs.runner import get_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{jid}", response_model=JobItem)
def status(jid: str, ctx=Depends(session_ctx)) -> JobItem:
    job = get_job(jid)
    if not job or job["session_id"] != ctx["session_id"]:
        raise HTTPException(404, "Job not found")
    return JobItem(**job)


@router.get("/{jid}/stream")
async def stream(jid: str, request: Request, ctx=Depends(session_ctx)):
    # Authorization must happen before committing SSE's 200 response headers.
    status(jid, ctx)

    async def gen():
        last_upd = None
        deadline = asyncio.get_running_loop().time() + 1800
        while asyncio.get_running_loop().time() < deadline:
            if await request.is_disconnected():
                return
            job = get_job(jid)
            if not job or job["session_id"] != ctx["session_id"]:
                yield {"event": "error", "data": json.dumps({"error": "not_found"})}
                return
            if job["updated_at"] != last_upd:
                last_upd = job["updated_at"]
                yield {
                    "event": "progress",
                    "data": json.dumps(JobItem(**job).model_dump(), default=str),
                }
            if job["status"] in {"done", "error", "cancelled"}:
                yield {"event": "end", "data": json.dumps({"status": job["status"]})}
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(gen())
