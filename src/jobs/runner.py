"""Job store on top of SQLite + a thread-pool executor. Progress is polled or SSE-streamed."""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from config import settings
from src.core.db import close_connection, dumps, get_conn, loads, now, tx
from src.core.logger import get_logger

log = get_logger(__name__)

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vrag-jobs")
_LOCK = threading.RLock()
_initialized_path: str | None = None
_slots: threading.BoundedSemaphore | None = None
_submitted: set[str] = set()


class QueueFull(Exception):
    pass


class JobCancelled(BaseException):
    """Control signal bypassing extractor callbacks that swallow Exception."""


def reconcile_interrupted_jobs() -> int:
    """Startup only, BEFORE accepting requests. Never replay provider work."""
    with tx() as conn:
        rows = conn.execute("SELECT id FROM jobs WHERE status IN ('queued','running')").fetchall()
        message = "Interrupted by process restart. Retry explicitly."
        conn.execute("UPDATE documents SET status='error',last_error=?,updated_at=? "
                     "WHERE status IN ('pending','processing')", (message, now()))
        conn.execute("UPDATE jobs SET status='error',stage='interrupted',message=?,updated_at=? "
                     "WHERE status IN ('queued','running')", (message, now()))
        conn.execute("UPDATE usage_reservations SET state='released' WHERE kind='upload' AND state='pending'")
        return len(rows)


def initialize_runner() -> None:
    global _initialized_path, _slots
    if _initialized_path == str(settings.sqlite_path):
        return
    with _LOCK:
        path = str(settings.sqlite_path)
        if _initialized_path != path:
            reconcile_interrupted_jobs()
            _slots = threading.BoundedSemaphore(settings.job_max_pending)
            _initialized_path = path


def create_job(session_id: str, kind: str, document_id: str | None = None) -> str:
    initialize_runner()
    jid = uuid.uuid4().hex
    with tx() as conn:
        if document_id is not None:
            doc = conn.execute("SELECT * FROM documents WHERE id=? AND session_id=?", (document_id, session_id)).fetchone()
            if not doc or doc["status"] not in {"pending", "error", "cancelled"}:
                raise ValueError("Document is not available for ingestion")
            if conn.execute("SELECT 1 FROM jobs WHERE document_id=? AND status IN ('queued','running')", (document_id,)).fetchone():
                raise ValueError("Document already has an active job")
        conn.execute("INSERT INTO jobs(id,session_id,kind,status,created_at,updated_at,document_id) "
                     "VALUES(?,?,?,'queued',?,?,?)", (jid, session_id, kind, now(), now(), document_id))
        if document_id is not None:
            conn.execute("UPDATE documents SET job_id=?,status='pending',last_error=NULL,updated_at=? WHERE id=?",
                         (jid, now(), document_id))
    return jid


def update_job(
    jid: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    total: int | None = None,
    stage: str | None = None,
    message: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    fields, values = [], []
    if status is not None: fields.append("status=?"); values.append(status)
    if progress is not None: fields.append("progress=?"); values.append(progress)
    if total is not None: fields.append("total=?"); values.append(total)
    if stage is not None: fields.append("stage=?"); values.append(stage)
    if message is not None: fields.append("message=?"); values.append(message)
    if result is not None: fields.append("result_json=?"); values.append(dumps(result))
    fields.append("updated_at=?"); values.append(now())
    values.append(jid)
    with tx() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id=?", values)


def get_job(jid: str) -> dict[str, Any] | None:
    row = get_conn().execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["result"] = loads(d.pop("result_json", None))
    return d


def checkpoint(jid: str) -> None:
    job = get_job(jid)
    if not job or job["cancel_requested"] or job["status"] not in {"queued", "running"}:
        raise JobCancelled()
    if job["document_id"]:
        doc = get_conn().execute("SELECT status,job_id FROM documents WHERE id=?", (job["document_id"],)).fetchone()
        if not doc or doc["status"] not in {"pending", "processing"} or doc["job_id"] != jid:
            raise JobCancelled()


def _terminal(jid: str, status: str, message: str) -> None:
    with tx() as conn:
        conn.execute("UPDATE jobs SET status=?,stage=?,message=?,updated_at=? "
                     "WHERE id=? AND status IN ('queued','running')", (status, status, message, now(), jid))
        conn.execute("UPDATE documents SET status=?,last_error=?,updated_at=? "
                     "WHERE job_id=? AND status IN ('pending','processing')", (status, message, now(), jid))


def request_cancel(jid: str, session_id: str) -> bool:
    with tx() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=? AND session_id=?", (jid, session_id)).fetchone()
        if not row:
            return False
        if row["status"] not in {"queued", "running"}:
            return True
        conn.execute("UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=?", (now(), jid))
        if row["status"] == "queued":
            _terminal(jid, "cancelled", "Ingestion cancelled.")
        return True


def submit(jid: str, target: Callable[..., None], *args, **kwargs) -> None:
    initialize_runner()
    with _LOCK:
        job = get_job(jid)
        if not job or job["status"] != "queued" or jid in _submitted:
            raise ValueError("Job is not available for submission")
        doc_id = kwargs.get("document_id")
        if doc_id and not job["document_id"]:
            with tx() as conn:
                doc = conn.execute("SELECT * FROM documents WHERE id=? AND session_id=?", (doc_id, job["session_id"])).fetchone()
                if not doc or doc["status"] not in {"pending", "error", "cancelled"}:
                    raise ValueError("Document is not available for ingestion")
                if conn.execute("SELECT 1 FROM jobs WHERE document_id=? AND status IN ('queued','running')", (doc_id,)).fetchone():
                    raise ValueError("Document already has an active job")
                conn.execute("UPDATE jobs SET document_id=? WHERE id=?", (doc_id, jid))
                conn.execute("UPDATE documents SET job_id=?,status='pending',updated_at=? WHERE id=?", (jid, now(), doc_id))
        assert _slots is not None
        slots = _slots
        if not slots.acquire(blocking=False):
            _terminal(jid, "error", "Ingestion queue is full. Retry later.")
            raise QueueFull("Ingestion queue is full. Retry later.")
        _submitted.add(jid)

    def _runner():
        try:
            with tx():
                checkpoint(jid)
                update_job(jid, status="running", stage="started")
            target(*args, **kwargs)
            with tx():
                current = get_job(jid)
                if current and current["status"] == "running":
                    checkpoint(jid)
                    update_job(jid, status="done", stage="done")
        except JobCancelled:
            _terminal(jid, "cancelled", "Ingestion cancelled.")
        except Exception:
            log.exception("job_failed", jid=jid)
            _terminal(jid, "error", "Ingestion failed. Retry or contact the administrator.")
        finally:
            slots.release()
            with _LOCK:
                _submitted.discard(jid)
            close_connection()
    try:
        _pool.submit(_runner)
    except Exception:
        slots.release()
        with _LOCK:
            _submitted.discard(jid)
        _terminal(jid, "error", "Job executor is unavailable. Retry later.")
        raise


def watch(jid: str, poll_ms: int = 500, timeout_s: float = 60 * 30):
    """Generator yielding job snapshots until it reaches a terminal state or times out."""
    start = time.time()
    last_updated = 0.0
    while True:
        job = get_job(jid)
        if not job:
            return
        if job["updated_at"] != last_updated:
            last_updated = job["updated_at"]
            yield job
        if job["status"] in {"done", "error", "cancelled"}:
            return
        if time.time() - start > timeout_s:
            return
        time.sleep(poll_ms / 1000)
