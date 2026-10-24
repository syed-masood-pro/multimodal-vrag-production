"""Shared dependencies: session resolution."""
from __future__ import annotations

from fastapi import Request, Response

from src.core.session import get_or_create_session, namespace_for


def session_ctx(request: Request, response: Response) -> dict:
    # Main app middleware attaches cookies to the ACTUAL response, including SSE.
    # Fallback keeps dependencies usable in small standalone/test applications.
    sid = getattr(request.state, "session_id", None) or get_or_create_session(request, response)
    return {"session_id": sid, "namespace": namespace_for(sid)}
