"""Offline API contracts: temporary SQLite/PDFs, mocked pipeline and providers.

No server, model downloads, generation or external network calls are required.
Run under pytest with the project's existing dependencies.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib
import inspect
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
import uuid

import fitz
from fastapi import Request
from fastapi.testclient import TestClient
from itsdangerous import URLSafeSerializer, URLSafeTimedSerializer
import pytest

from config import settings
from src.core import db, session, usage


def _pdf(pages=1, width=300, height=300):
    with fitz.open() as pdf:
        for _ in range(pages):
            page = pdf.new_page(width=width, height=height)
            page.insert_text((30, 40), "Offline evidence")
        return pdf.tobytes()


def _events(response):
    parsed = []
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields and "data" in fields:
            parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


def _wait(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for offline worker")
        time.sleep(.005)


@pytest.fixture
def api(monkeypatch, tmp_path):
    db.close_connection()
    for name in ("cache_dir", "image_dir", "upload_dir"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(settings, name, path)
    monkeypatch.setattr(settings, "sqlite_path", tmp_path / "api.db")
    monkeypatch.setattr(settings, "session_secret", "change-me-in-prod")
    monkeypatch.setattr(settings, "tenant_daily_asks", 100)
    monkeypatch.setattr(settings, "tenant_max_documents", 100)
    monkeypatch.setattr(settings, "tenant_max_storage_mb", 10)
    monkeypatch.setattr(settings, "max_pdf_pages", 5)
    monkeypatch.setattr(settings, "groq_api_key", "offline-configured")
    monkeypatch.setattr(settings, "pinecone_api_key", "offline-configured")

    app_module = importlib.import_module("src.api.app")
    ask = importlib.import_module("src.api.ask_router")
    documents = importlib.import_module("src.api.documents")
    lifecycle = importlib.import_module("src.api.lifecycle")
    runner = importlib.import_module("src.jobs.runner")
    ingest = importlib.import_module("src.jobs.ingest_job")
    pipeline = importlib.import_module("src.rag.pipeline")
    compaction = importlib.import_module("src.rag.compaction")
    monkeypatch.setattr(compaction, "_reranker", MagicMock(side_effect=AssertionError("No model warmup")))
    monkeypatch.setattr(app_module, "readiness_probe", MagicMock(side_effect=AssertionError("No live ping")))
    # Don't register process-global Prometheus collectors repeatedly in isolated apps.
    metrics = MagicMock()
    metrics.instrument.return_value = metrics
    monkeypatch.setattr(app_module, "instrumentator", metrics)
    monkeypatch.setattr(app_module.limiter, "enabled", False)
    monkeypatch.setattr(runner, "_initialized_path", None)
    monkeypatch.setattr(ask, "_STREAM_SLOTS", threading.BoundedSemaphore(4))
    monkeypatch.setattr(documents, "submit", MagicMock())
    monkeypatch.setattr(lifecycle, "submit", MagicMock())
    monkeypatch.setattr(documents, "delete_document", MagicMock())

    def result(question, **kwargs):
        ids = kwargs["document_ids"]
        sources = [] if not ids else [{
            "label": "E1", "evidence_id": "e1", "chunk_id": "chunk1", "document_id": ids[0],
            "filename": "report.pdf", "page": 1, "bbox": [20, 20, 100, 60], "modality": "table",
            "source": "C:/private/report.pdf", "quote": "Offline evidence", "preview": "Offline evidence", "cited": True,
        }]
        return pipeline.RAGResponse(
            question=question, answer="Supported [E1]" if ids else "No evidence.", tier="cheap", model="offline",
            confidence=.91, confidence_band="HIGH", confidence_components={"grounding": .95},
            escalated=True, sources=sources, image_paths=["C:/private/image.png"], latency_ms=7,
            classification={"kind": "lookup"}, injection_flagged=False, answerable=bool(ids),
            abstained=not bool(ids), abstention_reason=None if ids else "no_evidence",
            status="answered" if ids else "abstained", citation_warnings=[], retrieval_query="standalone question",
            confidence_available=True, calibration={"version": "offline"},
        )

    def stream(question, **kwargs):
        r = result(question, **kwargs)
        yield {"type": "stage", "stage": "verifying", "detail": "Buffered verification"}
        yield {"type": "meta", "tier": r.tier, "model": r.model, "sources": r.sources, "image_paths": r.image_paths}
        for text in (r.answer[:4], r.answer[4:]):
            yield {"type": "token", "text": text}
        yield {"type": "done", **r.to_dict()}

    monkeypatch.setattr(ask, "pipeline_ask", MagicMock(side_effect=result))
    monkeypatch.setattr(ask, "ask_stream", MagicMock(side_effect=stream))
    app = app_module.create_app()
    app.state.readiness_external_checks = False
    with TestClient(app) as client:
        sid = client.get("/api/session").json()["session_id"]
        yield SimpleNamespace(app=app, client=client, sid=sid, ctx={"session_id": sid, "namespace": session.namespace_for(sid)},
                              ask=ask, documents=documents, lifecycle=lifecycle, runner=runner,
                              ingest=ingest, root=tmp_path, app_module=app_module, pipeline=pipeline)
    db.close_connection()


def _document(api, *, sid=None, filename="report.pdf", status="ready", raw=None):
    sid = sid or api.sid
    did = uuid.uuid4().hex
    raw = raw or _pdf()
    path = settings.upload_dir / sid / f"{did}.enc"
    api.documents.encrypt_file(raw, path)
    api.ingest.register_document(session_id=sid, filename=filename, storage_path=str(path),
                                 size_bytes=len(raw), sha256="offline", document_id=did)
    with db.tx() as conn:
        conn.execute("UPDATE documents SET status=?,pages=1,chunks=1 WHERE id=?", (status, did))
    return did


def _chunk(api, did, eid="e1", modality="table", bbox=None, page=1):
    meta = {"document_id": did, "evidence_id": eid, "chunk_id": "chunk1", "modality": modality,
            "page": page, "bbox": bbox, "evidence_preview": "Stored quote"}
    with db.tx() as conn:
        conn.execute("INSERT INTO document_chunks VALUES(?,?,?,?,?)",
                     (api.ctx["namespace"], did, f"{did}:chunk1", "Full stored corpus text", db.dumps(meta)))


def test_startup_session_cookie_usage_and_route_order(api):
    assert api.app.state.runner_initialized
    assert api.runner._initialized_path == str(settings.sqlite_path)
    assert api.client.get("/api/session").json()["session_id"] == api.sid
    assert api.client.get("/api/documents").json() == []
    assert api.client.get("/api/usage").json()["usage"]["daily_asks"] == 0
    paths = [getattr(route, "path", "") for route in api.app.routes]
    assert paths.index("/api/documents/compare") < paths.index("/api/documents/{document_id}")
    assert "/images" not in paths
    assert api.client.get("/images/private.png").status_code == 404
    with TestClient(api.app, base_url="https://testserver") as other:
        response = other.get("/api/session")
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie
        assert response.json()["session_id"] != api.sid


def test_generated_session_secret_persists_and_rejects_legacy_bad_and_expired_cookies(api):
    key = session._signing_secret()
    assert len(key) == 64 and key != settings.session_secret
    assert session._signing_secret() == key
    assert (api.root / ".session.key").read_text() == key
    invalid = [
        "malformed",
        URLSafeSerializer("change-me-in-prod", salt="vrag-session").dumps(api.sid),
        session._serializer().dumps({"id": api.sid}),
        session._serializer().dumps("../../not-a-session"),
    ]
    class ExpiredSerializer(URLSafeTimedSerializer):
        def make_signer(self, salt=None):
            signer = super().make_signer(salt)
            signer.get_timestamp = lambda: int(time.time()) - session.SESSION_MAX_AGE - 100
            return signer
    invalid.append(ExpiredSerializer(key, salt="vrag-session-v2").dumps(api.sid))
    for raw in invalid:
        request = Request({"type": "http", "scheme": "http", "path": "/api/session",
                           "headers": [(b"cookie", f"{settings.session_cookie}={raw}".encode())]})
        sid = session.resolve_session(request)
        assert sid != api.sid and session._SID.fullmatch(sid)
        assert request.state.session_cookie_value
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert set(pool.map(lambda _: session._signing_secret(), range(8))) == {key}


def test_readiness_offline_missing_keys_and_failed_probe_are_503(api, monkeypatch):
    assert api.client.get("/readyz").status_code == 200
    monkeypatch.setattr(settings, "groq_api_key", "")
    assert api.client.get("/readyz").status_code == 503
    assert not api.client.get("/readyz").json()["ready"]
    monkeypatch.setattr(settings, "groq_api_key", "offline")
    monkeypatch.setattr(settings, "pinecone_api_key", "")
    assert api.client.get("/readyz").status_code == 503
    monkeypatch.setattr(settings, "pinecone_api_key", "offline")
    api.app.state.readiness_external_checks = True
    monkeypatch.setattr(api.app_module, "readiness_probe", lambda: (False, {"pinecone": "fail: private-key"}))
    response = api.client.get("/readyz")
    assert response.status_code == 503 and "private-key" not in response.text


@pytest.mark.parametrize("mode", ["latest", "all"])
def test_implicit_scope_latest_ready_per_filename(api, mode):
    old, newest = _document(api), _document(api)
    _document(api, status="processing")  # newest version must not shadow READY
    another = _document(api, filename="another.pdf")
    response = api.client.post("/api/ask", json={"question": "What?", "version_mode": mode})
    assert response.status_code == 200
    ids = api.ask.pipeline_ask.call_args.kwargs["document_ids"]
    assert set(ids) == ({old, newest, another} if mode == "all" else {newest, another})
    assert api.ask.pipeline_ask.call_args.kwargs["namespace"] == api.ctx["namespace"]


def test_explicit_versions_empty_scope_and_invalid_scope_fail_closed(api):
    old, newest = _document(api), _document(api)
    response = api.client.post("/api/ask", json={"question": "What?", "document_ids": [old]})
    assert response.status_code == 200
    assert api.ask.pipeline_ask.call_args.kwargs["document_ids"] == [old]
    response = api.client.post("/api/ask", json={"question": "What?", "document_ids": []})
    assert response.status_code == 200 and response.json()["abstained"]
    assert api.ask.pipeline_ask.call_args.kwargs["document_ids"] == []
    pending = _document(api, status="pending")
    before = api.ask.pipeline_ask.call_count
    for payload in ({"document_id": "unknown"}, {"document_ids": [pending]},
                    {"document_id": old, "document_ids": [newest]}):
        assert api.client.post("/api/ask", json={"question": "What?", **payload}).status_code == 404
    assert api.ask.pipeline_ask.call_count == before


def test_conversation_scope_resolution_and_conflicting_or_foreign_cid(api):
    did, other = _document(api), _document(api, filename="other.pdf")
    cid = api.client.post("/api/conversations", json={"document_id": did}).json()["id"]
    response = api.client.post("/api/ask", json={"question": "What?", "conversation_id": cid})
    assert response.status_code == 200
    assert api.ask.pipeline_ask.call_args.kwargs["document_ids"] == [did]
    for payload in ({"conversation_id": "missing"}, {"conversation_id": cid, "document_id": other},
                    {"conversation_id": cid, "document_ids": [did, other]}):
        assert api.client.post("/api/ask", json={"question": "What?", **payload}).status_code == 404
    with TestClient(api.app) as foreign:
        assert foreign.post("/api/ask", json={"question": "What?", "conversation_id": cid}).status_code == 404
        assert foreign.post("/api/ask", json={"question": "What?", "document_ids": [did]}).status_code == 404
        assert foreign.post("/api/conversations", json={"document_id": did}).status_code == 404
        assert foreign.get(f"/api/conversations/{cid}/messages").status_code == 404
    assert len(api.client.get("/api/conversations").json()) == 1


def test_sync_and_stream_full_response_and_saved_metadata(api):
    did = _document(api)
    sync = api.client.post("/api/ask", json={"question": "What?", "document_id": did}).json()
    response = api.client.post("/api/ask/stream", json={"question": "What?", "document_ids": [did]})
    assert response.status_code == 200
    events = _events(response)
    done = next(data for event, data in events if event == "done")
    assert "".join(data["text"] for event, data in events if event == "token") == done["answer"]
    assert events[1][0] == "stage" and events[1][1]["stage"] == "verifying"
    for key in api.pipeline.RAGResponse.__dataclass_fields__:
        assert key in sync and key in done
        assert sync[key] == done[key]
    assert done["image_paths"] == [] and "image_urls" not in done
    assert "C:/private" not in response.text
    assert done["sources"][0]["label"] == "E1"
    for result in (sync, done):
        messages = api.client.get(f"/api/conversations/{result['conversation_id']}/messages").json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[-1]["id"] == result["message_id"]
        for key, value in result.items():
            assert messages[-1]["meta"][key] == value
    assert usage.snapshot(api.sid)["usage"]["daily_asks"] == 2
    assert {row[0] for row in db.get_conn().execute("SELECT success FROM usage_reservations WHERE kind='ask'")} == {1}


def test_first_request_sse_sets_cookie_on_actual_response(api):
    with TestClient(api.app) as first:
        response = first.post("/api/ask/stream", json={"question": "What?", "document_ids": []})
        assert response.status_code == 200 and "set-cookie" in response.headers
        done = next(data for event, data in _events(response) if event == "done")
        assert first.get(f"/api/conversations/{done['conversation_id']}/messages").status_code == 200
        assert first.get("/api/session").json()["session_id"] != api.sid


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/ask/stream"])
def test_validation_and_budget_rejected_before_provider_or_stream_headers(api, monkeypatch, endpoint):
    for body in ({"question": "   "}, {"question": "x", "version_mode": "unknown"}):
        assert api.client.post(endpoint, json=body).status_code == 422
    monkeypatch.setattr(settings, "tenant_daily_asks", 0)
    response = api.client.post(endpoint, json={"question": "What?"})
    assert response.status_code == 429
    assert response.json()["detail"]["resource"] == "daily_asks"
    assert "text/event-stream" not in response.headers["content-type"]
    api.ask.pipeline_ask.assert_not_called()
    api.ask.ask_stream.assert_not_called()
    assert api.client.get("/api/conversations").json() == []


def test_atomic_ask_admission_and_failure_finalization(api, monkeypatch):
    monkeypatch.setattr(settings, "tenant_daily_asks", 1)
    def reserve(_):
        try:
            return api.ask._prepare(importlib.import_module("src.api.schemas").AskRequest(question="What?"), api.ctx)
        except usage.BudgetExceeded:
            return None
        finally:
            db.close_connection()
    with ThreadPoolExecutor(max_workers=4) as pool:
        admitted = [result for result in pool.map(reserve, range(8)) if result is not None]
    assert len(admitted) == 1
    usage.finish_ask(admitted[0][3], False)
    assert usage.snapshot(api.sid)["usage"]["daily_asks"] == 1


def test_provider_errors_are_safe_and_failed_asks_remain_charged(api, monkeypatch):
    api.ask.pipeline_ask.side_effect = RuntimeError("private provider credential")
    response = api.client.post("/api/ask", json={"question": "What?"})
    assert response.status_code == 502 and "credential" not in response.text
    monkeypatch.setattr(api.ask, "ask_stream", lambda *a, **k: iter([{"type": "error", "error": "private credential"}]))
    response = api.client.post("/api/ask/stream", json={"question": "What?"})
    assert _events(response)[-1][0] == "error" and "credential" not in response.text
    assert db.get_conn().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    _wait(lambda: db.get_conn().execute("SELECT COUNT(*) FROM usage_reservations WHERE state='pending'").fetchone()[0] == 0)
    assert usage.snapshot(api.sid)["usage"]["daily_asks"] == 2


def test_message_pair_and_usage_commit_rollback_together(api, monkeypatch):
    original = usage.finish_ask
    def fail_success(reservation, success):
        if success:
            raise RuntimeError("commit failed")
        return original(reservation, success)
    monkeypatch.setattr(usage, "finish_ask", fail_success)
    assert api.client.post("/api/ask", json={"question": "What?"}).status_code == 502
    assert db.get_conn().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    row = db.get_conn().execute("SELECT state,success FROM usage_reservations WHERE kind='ask'").fetchone()
    assert tuple(row) == ("finished", 0)


def test_stream_capacity_rejected_without_spending_budget(api):
    slots = api.ask._STREAM_SLOTS
    for _ in range(4):
        assert slots.acquire(blocking=False)
    try:
        response = api.client.post("/api/ask/stream", json={"question": "What?"})
        assert response.status_code == 503 and response.headers["retry-after"] == "5"
        assert usage.snapshot(api.sid)["usage"]["daily_asks"] == 0
    finally:
        for _ in range(4):
            slots.release()


def test_worker_bounded_backpressure_disconnect_closes_generator_and_releases_slot(api, monkeypatch):
    consumed, closed = [], threading.Event()
    def many_events(*args, **kwargs):
        try:
            for number in range(10000):
                consumed.append(number)
                yield {"type": "stage", "stage": "verifying"}
        finally:
            closed.set()
    monkeypatch.setattr(api.ask, "ask_stream", many_events)
    req = importlib.import_module("src.api.schemas").AskRequest(question="What?", document_ids=[])
    assert api.ask._STREAM_SLOTS.acquire(blocking=False)
    work = api.ask._StreamWork(req, api.ctx, api.ask._prepare(req, api.ctx))
    try:
        work.start()
        _wait(lambda: work.events.full())
        assert len(consumed) <= api.ask._STREAM_QUEUE_SIZE + 1
        work.cancel()
        count = len(consumed)
        work.thread.join(timeout=5)
        assert not work.thread.is_alive() and closed.is_set()
        assert len(consumed) == count
        assert work.released
        assert work.events.empty()
        assert db.get_conn().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    finally:
        work.cancel()
        if work.thread:
            work.thread.join(timeout=5)


def test_actual_asgi_disconnect_does_not_consume_after_blocked_provider_returns(api, monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    consumed = []
    def blocked(*args, **kwargs):
        try:
            yield {"type": "stage", "stage": "verifying"}
            entered.set()
            assert release.wait(timeout=5)
            yield {"type": "token", "text": "discarded after disconnect"}
            consumed.append("must not advance")
        finally:
            closed.set()
    monkeypatch.setattr(api.ask, "ask_stream", blocked)
    req = importlib.import_module("src.api.schemas").AskRequest(question="What?", document_ids=[])
    async def exercise():
        inbox = asyncio.Queue()
        sent = []
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": "/api/ask/stream", "headers": [], "query_string": b""}
        async def receive():
            return await inbox.get()
        async def send(message):
            sent.append(message)
        request = Request(scope, receive)
        response = await inspect.unwrap(api.ask.ask_stream_endpoint)(request, req, api.ctx)
        task = asyncio.create_task(response(scope, receive, send))
        try:
            for _ in range(500):
                if entered.is_set():
                    break
                await asyncio.sleep(.005)
            assert entered.is_set()
            await inbox.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, timeout=3)
            assert response.work.cancelled.is_set()
            assert not response.work.released  # running provider still holds capacity
        finally:
            response.work.cancel()
            release.set()
            if response.work.thread:
                response.work.thread.join(timeout=5)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert response.work.released and closed.is_set()
        assert not consumed
        assert b"discarded after disconnect" not in b"".join(m.get("body", b"") for m in sent)
    asyncio.run(exercise())


def test_upload_unique_storage_atomic_usage_job_link_and_queue_full(api):
    raw = _pdf()
    responses = [api.client.post("/api/documents/upload", files={"file": ("same.pdf", raw, "application/pdf")})
                 for _ in range(2)]
    assert all(r.status_code == 202 for r in responses)
    rows = db.get_conn().execute("SELECT * FROM documents WHERE session_id=? ORDER BY version", (api.sid,)).fetchall()
    assert [r["version"] for r in rows] == [1, 2]
    assert len({r["storage_path"] for r in rows}) == 2
    assert all(Path(r["storage_path"]).exists() for r in rows)
    assert usage.snapshot(api.sid)["usage"]["storage_bytes"] == 2 * len(raw)
    assert usage.snapshot(api.sid)["pending_uploads"] == 0
    for row in rows:
        assert api.runner.get_job(row["job_id"])["document_id"] == row["id"]
    api.documents.submit.side_effect = api.runner.QueueFull("internal detail")
    response = api.client.post("/api/documents/upload", files={"file": ("same.pdf", raw)})
    assert response.status_code == 503 and "internal detail" not in response.text
    ids = response.json()["detail"]
    detail = api.client.get(f"/api/documents/{ids['document_id']}/detail").json()
    assert detail["document"]["status"] == "error" and detail["document"]["job_id"] == ids["job_id"]
    assert detail["jobs"][0]["status"] == "error"
    assert usage.snapshot(api.sid)["usage"]["documents"] == 3


def test_upload_rejects_invalid_pdf_page_limits_budget_and_cleans_failed_registration(api, monkeypatch):
    for filename, raw, code in [("wrong.txt", _pdf(), 400), ("bad.pdf", b"%PDF-broken", 400),
                                ("large.pdf", _pdf(pages=6), 413)]:
        assert api.client.post("/api/documents/upload", files={"file": (filename, raw)}).status_code == code
    assert api.client.get("/api/documents").json() == []
    monkeypatch.setattr(settings, "tenant_max_documents", 0)
    assert api.client.post("/api/documents/upload", files={"file": ("ok.pdf", _pdf())}).status_code == 429
    monkeypatch.setattr(settings, "tenant_max_documents", 100)
    monkeypatch.setattr(api.documents, "register_document", MagicMock(side_effect=RuntimeError("private db error")))
    response = api.client.post("/api/documents/upload", files={"file": ("ok.pdf", _pdf())})
    assert response.status_code == 500 and "private db" not in response.text
    assert not list(settings.upload_dir.rglob("*.enc"))
    assert usage.snapshot(api.sid)["pending_uploads"] == 0
    assert api.client.get("/api/documents").json() == []


def test_document_lifecycle_fields_and_retry_cancel_jobs_authorized_before_sse(api):
    did = _document(api, status="error")
    fields = api.client.get("/api/documents").json()[0]
    assert {"status", "updated_at", "attempt_count", "last_error", "job_id"} <= fields.keys()
    response = api.client.post(f"/api/documents/{did}/retry")
    assert response.status_code == 202
    jid = response.json()["job_id"]
    assert api.client.post(f"/api/documents/{did}/cancel").status_code == 202
    assert api.client.get(f"/api/jobs/{jid}").json()["status"] == "cancelled"
    events = _events(api.client.get(f"/api/jobs/{jid}/stream"))
    assert events[-1] == ("end", {"status": "cancelled"})
    with TestClient(api.app) as other:
        for suffix in ("", "/stream"):
            assert other.get(f"/api/jobs/{jid}{suffix}").status_code == 404
        assert other.get(f"/api/documents/{did}/detail").status_code == 404
        assert other.get(f"/api/documents/compare?left_id={did}&right_id={did}").status_code == 404


def test_evidence_owned_ready_stored_quote_highlight_and_private_cache(api):
    did = _document(api)
    _chunk(api, did, bbox=[20, 20, 120, 70])
    response = api.client.get(f"/api/documents/{did}/evidence/e1")
    assert response.status_code == 200
    evidence = response.json()
    assert evidence["quote"] == "Stored quote" and evidence["bbox"] == [20, 20, 120, 70]
    assert evidence["chunk_id"] == "chunk1" and evidence["document_id"] == did
    assert evidence["page_image_url"] == f"/api/documents/{did}/pages/1.png?evidence_id=e1"
    highlighted = api.client.get(evidence["page_image_url"])
    plain = api.client.get(f"/api/documents/{did}/pages/1.png")
    assert highlighted.status_code == plain.status_code == 200
    assert highlighted.content != plain.content
    for response in (response, highlighted, plain):
        assert response.headers["cache-control"] == "private, no-store"
    assert api.client.get(f"/api/documents/{did}/pages/2.png?evidence_id=e1").status_code == 404
    assert api.client.get(f"/api/documents/{did}/evidence/unknown").status_code == 404
    with TestClient(api.app) as other:
        assert other.get(evidence["page_image_url"]).status_code == 404
        assert other.get(f"/api/documents/{did}/evidence/e1").status_code == 404
    with db.tx() as conn:
        conn.execute("UPDATE documents SET status='deleting' WHERE id=?", (did,))
    assert api.client.get(evidence["page_image_url"]).status_code == 404


def test_text_evidence_has_no_fake_bbox_and_oversized_pages_do_not_render(api, monkeypatch):
    did = _document(api)
    _chunk(api, did, modality="text", bbox=[20, 20, 120, 70])
    evidence = api.client.get(f"/api/documents/{did}/evidence/e1").json()
    assert evidence["bbox"] is None
    assert api.client.get(evidence["page_image_url"]).content == api.client.get(f"/api/documents/{did}/pages/1.png").content
    huge = _document(api, raw=_pdf(width=15000, height=300))
    pixmap = MagicMock(side_effect=AssertionError("Must check dimensions before rasterizing"))
    monkeypatch.setattr(fitz.Page, "get_pixmap", pixmap)
    assert api.client.get(f"/api/documents/{huge}/pages/1.png").status_code == 413
    pixmap.assert_not_called()


def test_render_scales_before_allocating_pixmap(api, monkeypatch):
    did = _document(api, raw=_pdf(width=10000, height=10000))
    render = MagicMock(return_value=SimpleNamespace(tobytes=lambda format: b"offline-png"))
    monkeypatch.setattr(fitz.Page, "get_pixmap", render)
    assert api.client.get(f"/api/documents/{did}/pages/1.png").status_code == 200
    scale = render.call_args.kwargs["matrix"].a
    side = __import__("math").ceil(10000 * scale)
    assert side <= api.documents.MAX_RENDER_DIMENSION
    assert side * side <= api.documents.MAX_RENDER_PIXELS


def test_delete_active_rejected_vector_failure_stays_deleting_and_retry_cleans(api):
    active = _document(api, status="pending")
    assert api.client.delete(f"/api/documents/{active}").status_code == 409
    did = _document(api)
    _chunk(api, did)
    path = Path(db.get_conn().execute("SELECT storage_path FROM documents WHERE id=?", (did,)).fetchone()[0])
    image = settings.image_dir / f"{did}_p1_img1.png"
    image.write_bytes(b"private")
    unrelated = settings.image_dir / "other_p1.png"
    unrelated.write_bytes(b"other")
    api.documents.delete_document.side_effect = RuntimeError("private vector details")
    response = api.client.delete(f"/api/documents/{did}")
    assert response.status_code == 502 and "private vector" not in response.text
    row = db.get_conn().execute("SELECT status,last_error FROM documents WHERE id=?", (did,)).fetchone()
    assert row["status"] == "deleting" and "retry" in row["last_error"].lower()
    assert path.exists() and image.exists()
    assert api.client.post("/api/ask", json={"question": "What?", "document_id": did}).status_code == 404
    api.documents.delete_document.side_effect = None
    assert api.client.delete(f"/api/documents/{did}").json() == {"status": "deleted"}
    assert not path.exists() and not image.exists() and unrelated.exists()
    assert not db.get_conn().execute("SELECT 1 FROM document_chunks WHERE document_id=?", (did,)).fetchone()


def test_local_cleanup_failure_does_not_report_success(api, monkeypatch):
    did = _document(api)
    _chunk(api, did)
    original = Path.unlink
    def fail_encrypted_file(path, *args, **kwargs):
        if path.name == f"{did}.enc":
            raise PermissionError("private filesystem path")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_encrypted_file)
    response = api.client.delete(f"/api/documents/{did}")
    assert response.status_code == 502 and "filesystem path" not in response.text
    assert db.get_conn().execute("SELECT status FROM documents WHERE id=?", (did,)).fetchone()[0] == "deleting"


def test_legacy_message_images_are_not_publicly_returned(api):
    conversations = importlib.import_module("src.api.conversations")
    cid = api.client.post("/api/conversations", json={}).json()["id"]
    conversations.add_message(cid, "assistant", "saved", meta={
        "image_urls": ["/images/private.png"], "image_paths": ["C:/private/image.png"],
        "sources": [{"source": "C:/private/report.pdf", "image_path": "C:/private/image.png"}],
    })
    response = api.client.get(f"/api/conversations/{cid}/messages")
    assert "C:/private" not in response.text and "/images/" not in response.text
    assert response.json()[0]["meta"]["sources"][0]["source"] == "report.pdf"


def test_feedback_owned_assistant_literal_vote_dedup_latest_and_conversation_audit(api):
    result = api.client.post("/api/ask", json={"question": "What?"}).json()
    cid, mid = result["conversation_id"], result["message_id"]
    for vote in (1, -1, 1):
        assert api.client.post("/api/feedback", json={"message_id": mid, "vote": vote, "reason": str(vote)}).status_code == 200
    row = db.get_conn().execute("SELECT * FROM feedback WHERE message_id=?", (mid,)).fetchall()
    assert len(row) == 1 and row[0]["vote"] == 1 and row[0]["conversation_id"] == cid
    assert api.client.post("/api/feedback", json={"message_id": mid, "vote": 0}).status_code == 422
    uid = api.client.get(f"/api/conversations/{cid}/messages").json()[0]["id"]
    assert api.client.post("/api/feedback", json={"message_id": uid, "vote": 1}).status_code == 404
    with TestClient(api.app) as other:
        assert other.post("/api/feedback", json={"message_id": mid, "vote": 1}).status_code == 404
        assert other.delete(f"/api/conversations/{cid}").status_code == 404
    assert api.client.delete(f"/api/conversations/{cid}").status_code == 200
    assert db.get_conn().execute("SELECT 1 FROM audit WHERE action='conversation.delete' AND resource=?", (cid,)).fetchone()
    assert not db.get_conn().execute("SELECT 1 FROM messages WHERE conversation_id=?", (cid,)).fetchone()
    assert not db.get_conn().execute("SELECT 1 FROM feedback WHERE conversation_id=?", (cid,)).fetchone()


def test_add_message_rolls_back_when_conversation_timestamp_update_fails(api):
    conversations = importlib.import_module("src.api.conversations")
    cid = api.client.post("/api/conversations", json={}).json()["id"]
    db.get_conn().execute("CREATE TEMP TRIGGER reject_update BEFORE UPDATE ON conversations "
                          "BEGIN SELECT RAISE(ABORT,'offline rollback test'); END")
    with pytest.raises(sqlite3.IntegrityError):
        conversations.add_message(cid, "user", "not committed")
    assert db.get_conn().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0