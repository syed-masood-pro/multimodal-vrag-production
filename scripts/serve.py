"""Serve the API + UI with uvicorn.

    python -m scripts.serve
"""
from __future__ import annotations

import uvicorn

from config import settings


def main() -> None:
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
