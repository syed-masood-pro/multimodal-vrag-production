"""Structured logging setup."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

import structlog

from config import settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    handlers: list[logging.Handler]=[logging.StreamHandler(sys.stdout)]
    if str(settings.log_dir):
        settings.log_dir.mkdir(parents=True,exist_ok=True)
        file_handler=RotatingFileHandler(
            settings.log_dir/"vrag.log",
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding="utf-8",
        )
        handlers.append(file_handler)
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=handlers,
        force=True
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if settings.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)
