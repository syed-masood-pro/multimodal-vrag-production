"""UI-only server. Serves the frontend without loading the RAG pipeline.

Use this to preview the UI without installing all backend dependencies.

    python -m scripts.serve_ui

API endpoints return 503, except the empty session bootstrap. Readiness is
explicitly degraded. Use `scripts/serve.py` for the full app.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

app = FastAPI(title="Multimodal VRAG (UI only)")


@app.get("/api/session")
def session():
    """Allow the frontend to boot; no real tenant/session is created."""
    return {}


@app.get("/readyz")
def readiness():
    return {"ready": False, "details": {"mode": "UI-only preview; backend disabled"}}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
def backend_unavailable(path: str):
    """Cover documents/detail/retry/cancel/compare/evidence, jobs (including
    SSE), conversations, usage, feedback and ask/stream without fake data.
    This catch-all must follow the session route.
    """
    raise HTTPException(
        status_code=503,
        detail="UI-only preview: backend unavailable. Run python -m scripts.serve with configured credentials for documents, search and tenant usage.",
    )


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


if __name__ == "__main__":
    print("UI at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
