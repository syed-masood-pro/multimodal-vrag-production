"""Append-only audit trail."""
from __future__ import annotations

from src.core.db import dumps, get_conn, now


def log(
    action: str,
    *,
    session_id: str | None = None,
    resource: str | None = None,
    ip: str | None = None,
    request_id: str | None = None,
    meta: dict | None = None,
) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit(session_id, action, resource, ip, request_id, meta_json, created_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)",
        (session_id, action, resource, ip, request_id, dumps(meta) if meta else None, now()),
    )
