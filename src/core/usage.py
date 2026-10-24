"""Atomic local quotas. Reservations are conservative: provider work is never refunded."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from config import settings
from src.core.db import now, tx


class BudgetExceeded(Exception):
    def __init__(self, resource: str):
        self.resource = resource
        super().__init__(f"Tenant {resource} limit exceeded.")


def _day() -> str:
    return datetime.now(timezone.utc).date().isoformat()

# --- GLOBAL TOKEN TRACKER ---
_token_tracker = threading.local()

def add_tokens(input_tokens: int, output_tokens: int) -> None:
    _token_tracker.input_tokens = getattr(_token_tracker, "input_tokens", 0) + input_tokens
    _token_tracker.output_tokens = getattr(_token_tracker, "output_tokens", 0) + output_tokens

def flush_tokens(session_id: str, document_id: str | None = None) -> tuple[int, int]:
    inp = getattr(_token_tracker, "input_tokens", 0)
    out = getattr(_token_tracker, "output_tokens", 0)
    if inp > 0 or out > 0:
        record_tokens(session_id, inp, out, document_id)
    _token_tracker.input_tokens = 0
    _token_tracker.output_tokens = 0
    return inp, out  # <-- Return the counts here
# ----------------------------


def _insert(conn, session_id: str, kind: str, amount: int, document_id=None) -> str:
    rid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO usage_reservations(id,session_id,kind,day,amount,state,document_id,created_at) "
        "VALUES(?,?,?,?,?,'pending',?,?)",
        (rid, session_id, kind, _day(), amount, document_id, now()),
    )
    return rid

def _daily(conn, session_id: str, kind: str) -> int:
    return conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM usage_reservations "
        "WHERE session_id=? AND kind=? AND day=? AND state!='released'",
        (session_id, kind, _day()),
    ).fetchone()[0]

def _storage(conn, session_id: str) -> tuple[int, int, int, int]:
    docs, size = conn.execute(
        "SELECT COUNT(*),COALESCE(SUM(size_bytes),0) FROM documents WHERE session_id=?", (session_id,)
    ).fetchone()
    pending, pending_size = conn.execute(
        "SELECT COUNT(*),COALESCE(SUM(amount),0) FROM usage_reservations "
        "WHERE session_id=? AND kind='upload' AND state='pending'", (session_id,)
    ).fetchone()
    return docs, size, pending, pending_size

def reserve_ask(session_id: str) -> str:
    with tx() as conn:
        if _daily(conn, session_id, "ask") >= settings.tenant_daily_asks:
            raise BudgetExceeded("daily_asks")
        return _insert(conn, session_id, "ask", 1)

def finish_ask(reservation_id: str, success: bool) -> None:
    with tx() as conn:
        conn.execute("UPDATE usage_reservations SET state='finished',success=? "
                     "WHERE id=? AND kind='ask' AND state='pending'", (int(success), reservation_id))

def reserve_upload(session_id: str, size_bytes: int) -> str:
    from src.jobs.runner import initialize_runner
    initialize_runner()
    if size_bytes < 0:
        raise ValueError("size_bytes must be nonnegative")
    with tx() as conn:
        docs, size, pending, pending_size = _storage(conn, session_id)
        if docs + pending + 1 > settings.tenant_max_documents:
            raise BudgetExceeded("documents")
        if size + pending_size + size_bytes > settings.tenant_max_storage_mb * 1024 * 1024:
            raise BudgetExceeded("storage_bytes")
        return _insert(conn, session_id, "upload", size_bytes)

def finish_upload(reservation_id: str, document_id: str | None) -> None:
    with tx() as conn:
        row = conn.execute("SELECT * FROM usage_reservations WHERE id=? AND kind='upload'",
                           (reservation_id,)).fetchone()
        if not row:
            raise ValueError("Unknown upload reservation")
        if row["state"] != "pending":
            if row["document_id"] != document_id:
                raise ValueError("Reservation already finalized")
            return
        if document_id is not None:
            doc = conn.execute("SELECT size_bytes FROM documents WHERE id=? AND session_id=?",
                               (document_id, row["session_id"])).fetchone()
            if not doc or doc["size_bytes"] != row["amount"]:
                raise ValueError("Reservation does not match the owned document")
        conn.execute("UPDATE usage_reservations SET state=?,document_id=? WHERE id=?",
                     ("finished" if document_id else "released", document_id, reservation_id))

def reserve_pages(session_id: str, pages: int, document_id: str | None = None) -> str:
    if pages < 1:
        raise ValueError("PDF has no pages")
    if pages > settings.max_pdf_pages:
        raise BudgetExceeded("pdf_pages")
    with tx() as conn:
        if _daily(conn, session_id, "pages") + pages > settings.tenant_daily_pages:
            raise BudgetExceeded("daily_pages")
        return _insert(conn, session_id, "pages", pages, document_id)

def finish_pages(reservation_id: str, success: bool) -> None:
    with tx() as conn:
        conn.execute("UPDATE usage_reservations SET state='finished',success=? "
                     "WHERE id=? AND kind='pages' AND state='pending'", (int(success), reservation_id))

def record_tokens(session_id: str, input_tokens: int, output_tokens: int, document_id: str | None = None) -> None:
    with tx() as conn:
        if input_tokens > 0:
            rid_in = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO usage_reservations(id,session_id,kind,day,amount,state,document_id,created_at) "
                "VALUES(?,?,?,?,?,'finished',?,?)",
                (rid_in, session_id, "input_tokens", _day(), input_tokens, document_id, now()),
            )
        if output_tokens > 0:
            rid_out = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO usage_reservations(id,session_id,kind,day,amount,state,document_id,created_at) "
                "VALUES(?,?,?,?,?,'finished',?,?)",
                (rid_out, session_id, "output_tokens", _day(), output_tokens, document_id, now()),
            )

def snapshot(session_id: str) -> dict:
    with tx() as conn:
        docs, size, pending, pending_size = _storage(conn, session_id)
        
        limits = {
            "documents": settings.tenant_max_documents,
            "storage_bytes": settings.tenant_max_storage_mb * 1024 * 1024,
            "daily_asks": settings.tenant_daily_asks,
            "daily_pages": settings.tenant_daily_pages,
            "input_tokens": getattr(settings, "tenant_daily_input_tokens", 100000),
            "output_tokens": getattr(settings, "tenant_daily_output_tokens", 50000)
        }
        
        usage = {
            "documents": docs + pending, 
            "storage_bytes": size + pending_size,
            "daily_asks": _daily(conn, session_id, "ask"),
            "daily_pages": _daily(conn, session_id, "pages"),
            "input_tokens": _daily(conn, session_id, "input_tokens"),
            "output_tokens": _daily(conn, session_id, "output_tokens")
        }
        
        return {
            "day": _day(), 
            "timezone": "UTC", 
            "limits": limits, 
            "usage": usage,
            "remaining": {k: max(0, limits[k] - usage[k]) for k in limits},
            "pending_uploads": pending, 
            "pending_storage_bytes": pending_size,
            "stored_documents": docs, 
            "stored_bytes": size
        }