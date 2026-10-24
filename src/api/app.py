"""FastAPI app v2: multi-tenancy, async ingest, streaming, rate limit, observability."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from src.api.ask_router import router as ask_router
from src.api.conversations import router as conv_router
from src.api.deps import session_ctx
from src.api.documents import router as docs_router
from src.api.feedback import router as fb_router
from src.api.job_router import  router as jobs_router
from src.api.lifecycle import router as lifecycle_router, usage_router
from src.api.schemas import HealthResponse, ReadinessResponse
from src.core.db import close_connection, get_conn
from src.core.logger import get_logger
from src.core.observability import instrumentator, readiness_probe
from src.core.rate_limit import limiter
from src.core.session import resolve_session, set_session_cookie
from src.core.usage import BudgetExceeded
from src.jobs.runner import initialize_runner

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Local SQLite quotas + in-memory ingestion/stream slots require ONE process.
    # Reconcile interrupted work before accepting requests. Never generate/warm models.
    initialize_runner()
    app.state.runner_initialized = True
    try:
        yield
    finally:
        app.state.runner_initialized = False
        close_connection()


def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal VRAG API", version="2.0.0", lifespan=lifespan)
    app.state.runner_initialized = False
    # Offline health tests can disable external probes without loading a model.
    app.state.readiness_external_checks = True

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(BudgetExceeded)
    async def budget_exceeded(request: Request, exc: BudgetExceeded):
        return JSONResponse(status_code=429, content={"detail": {
            "message": "Browser session quota exceeded.", "resource": exc.resource,
            "remaining": 0,
        }})

    # Request-ID + access log
    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.time()
        try:
            if request.url.path.startswith("/api/"):
                resolve_session(request)
            resp = await call_next(request)
        except Exception:
            log.exception("unhandled", rid=rid, path=request.url.path)
            resp = JSONResponse(
                status_code=500, content={"detail": "Internal server error", "request_id": rid}
            )
        set_session_cookie(request, resp)
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "private, no-store"
        resp.headers["x-request-id"] = rid
        # Security headers
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        log.info(
            "request",
            rid=rid,
            method=request.method,
            path=request.url.path,
            status=resp.status_code,
            ms=int((time.time() - start) * 1000),
        )
        return resp

    # Prometheus
    instrumentator.instrument(app).expose(app, endpoint="/metrics")

    # Health/readiness
    @app.get("/livez", response_model=HealthResponse)
    def livez():
        return HealthResponse()

    @app.get("/readyz", response_model=ReadinessResponse)
    def readyz():
        details = {
            "groq": "configured" if settings.groq_api_key.strip() else "not_configured",
            "pinecone": "configured" if settings.pinecone_api_key.strip() else "not_configured",
            "runner": "ok" if app.state.runner_initialized else "not_initialized",
        }
        ok = all(v not in {"not_configured", "not_initialized"} for v in details.values())
        try:
            get_conn().execute("SELECT 1").fetchone()
            details["sqlite"] = "ok"
            if ok and app.state.readiness_external_checks:
                healthy, probe = readiness_probe()
                ok = ok and healthy
                details.update({k: v if v in {"ok", "not_configured"} else "unavailable"
                                for k, v in probe.items()})
        except Exception:
            ok = False
            details["dependencies"] = "unavailable"
        return JSONResponse(status_code=200 if ok else 503,
                            content=ReadinessResponse(ready=ok, details=details).model_dump())

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return HealthResponse()

    @app.get("/api/session")
    def session(ctx=Depends(session_ctx)):
        return {"session_id": ctx["session_id"]}

    # Routers
    app.include_router(lifecycle_router)
    app.include_router(usage_router)
    app.include_router(docs_router)
    app.include_router(conv_router)
    app.include_router(jobs_router)
    app.include_router(fb_router)
    app.include_router(ask_router)

    # Static
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(WEB_DIR / "index.html"))

    return app


app = create_app()
