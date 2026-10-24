"""Prometheus + readiness/liveness probes."""
from __future__ import annotations

from prometheus_fastapi_instrumentator import Instrumentator

from src.core.logger import get_logger

log = get_logger(__name__)

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    excluded_handlers=["/metrics", "/livez", "/readyz"],
)


def readiness_probe() -> tuple[bool, dict]:
    """Best-effort dependency checks — returns (ok, details)."""
    details: dict[str, str] = {}
    ok = True
    # DB
    try:
        from src.core.db import get_conn
        get_conn().execute("SELECT 1").fetchone()
        details["sqlite"] = "ok"
    except Exception as e:
        ok = False
        details["sqlite"] = f"fail: {e}"
    # Pinecone (optional — only if configured)
    try:
        from config import settings
        if settings.pinecone_api_key:
            from src.clients.pinecone_client import get_pinecone
            get_pinecone().list_indexes()
            details["pinecone"] = "ok"
        else:
            details["pinecone"] = "not_configured"
    except Exception as e:
        ok = False
        details["pinecone"] = f"fail: {e}"
    return ok, details
