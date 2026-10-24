"""Retry decorators with exponential backoff."""
from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .logger import get_logger

log = get_logger(__name__)


def _log_retry(retry_state):  # pragma: no cover - trivial
    log.warning(
        "retrying",
        attempt=retry_state.attempt_number,
        exc=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


def network_retry(max_attempts: int = 4):
    """Decorator for network-bound calls: retries on Exception (broad by design at boundary)."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
    )
