"""API-key based auth. Header: X-API-Key."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = settings.api_key
    if not expected:
        # Auth disabled if no API_KEY configured — for local dev only.
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
