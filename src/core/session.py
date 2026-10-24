"""Timed browser-session isolation, not account authentication or durable identity.

Local quotas are per browser cookie. Clearing it creates a new budget/namespace.
Keep the generated signing key private and persistent alongside the local database.
"""
from __future__ import annotations

import os
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import Request, Response
from itsdangerous import BadData, URLSafeTimedSerializer

from config import settings
from src.core.db import now, tx

SESSION_MAX_AGE = 60 * 60 * 24 * 30
_SID = re.compile(r"[A-Za-z0-9_-]{32}\Z")
_KEY_LOCK = threading.Lock()
_PLACEHOLDERS = {"", "change-me-in-prod", "change-me", "changeme", "secret",
                 "your-session-secret", "your-session-secret-here", "replace-me"}


def _signing_secret() -> str:
    configured = settings.session_secret.strip()
    if configured.lower() not in _PLACEHOLDERS:
        return configured
    path = Path(settings.sqlite_path).resolve().parent / ".session.key"
    with _KEY_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # O_EXCL prevents concurrent starters overwriting the winning key.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # A different process may still be finishing its exclusive write.
            for _ in range(50):
                key = path.read_text(encoding="ascii").strip()
                if re.fullmatch(r"[A-Za-z0-9_-]{64}", key):
                    return key
                time.sleep(.02)
            raise RuntimeError("Session signing key is unavailable")
        else:
            key = secrets.token_urlsafe(48)
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            return key


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_signing_secret(), salt="vrag-session-v2")


def _new_session_id() -> str:
    return secrets.token_urlsafe(24)


def _record_session(sid: str) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO sessions(id, created_at, last_seen) VALUES(?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen",
            (sid, now(), now()),
        )


def resolve_session(request: Request) -> str:
    existing = getattr(request.state, "session_id", None)
    if existing is not None:
        return existing
    raw = request.cookies.get(settings.session_cookie)
    serializer = _serializer()
    sid: str | None = None
    if raw:
        try:
            value = serializer.loads(raw, max_age=SESSION_MAX_AGE)
            if isinstance(value, str) and _SID.fullmatch(value):
                sid = value
        except BadData:
            pass  # Expired, malformed and legacy untimed cookies start a fresh session.
    if not sid:
        sid = _new_session_id()
        request.state.session_cookie_value = serializer.dumps(sid)
    _record_session(sid)
    request.state.session_id = sid
    return sid


def set_session_cookie(request: Request, response: Response) -> None:
    signed = getattr(request.state, "session_cookie_value", None)
    if signed:
        response.set_cookie(
            settings.session_cookie,
            signed,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
            max_age=SESSION_MAX_AGE,
            path="/",
        )


def get_or_create_session(request: Request, response: Response) -> str:
    sid = resolve_session(request)
    set_session_cookie(request, response)
    return sid


def namespace_for(session_id: str) -> str:
    """Pinecone namespace scoped to this session."""
    return f"s_{session_id[:20]}"
