"""Offline lifecycle wire contracts: isolated SQLite, mocked submissions, no providers."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from config import settings
from src.api import lifecycle, schemas
from src.core import db, usage
from src.jobs import runner


@pytest.fixture
def api(monkeypatch, tmp_path):
    db.close_connection()
    monkeypatch.setattr(settings, "sqlite_path", tmp_path / "contracts.db")
    monkeypatch.setattr(settings, "tenant_max_documents", 100)
    monkeypatch.setattr(settings, "tenant_max_storage_mb", 1)
    monkeypatch.setattr(settings, "tenant_daily_asks", 10)
    monkeypatch.setattr(settings, "tenant_daily_pages", 10)
    monkeypatch.setattr(settings, "max_pdf_pages", 5)
    monkeypatch.setattr(settings, "job_max_pending", 1)
    monkeypatch.setattr(runner, "_initialized_path", None)
    monkeypatch.setattr(runner, "_slots", None)
    monkeypatch.setattr(runner, "_submitted", set())
    monkeypatch.setattr(runner, "_pool", MagicMock())
    for sid in ("alice", "bob"):
        db.get_conn().execute("INSERT INTO sessions VALUES(?,?,?)", (sid, 1, 1))
    runner.initialize_runner()
    submission = MagicMock()
    monkeypatch.setattr(lifecycle, "submit", submission)
    app = FastAPI()
    app.include_router(lifecycle.router)
    app.include_router(lifecycle.usage_router)
    app.dependency_overrides[lifecycle.session_ctx] = lambda: {
        "session_id": "alice", "namespace": "s_alice",
    }
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(client=client, app=app, submit=submission)
    finally:
        db.close_connection()


def document(did="doc", *, sid="alice", status="error"):
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO documents(id,session_id,filename,version,size_bytes,storage_path,"
            "sha256,created_at,updated_at,status) VALUES(?,?,?,1,12,?,?,1,1,?)",
            (did, sid, "report.pdf", "C:/private/upload.enc", "private-hash", status),
        )
    return did


def history_job(jid, did, *, sid="alice", result=None):
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO jobs(id,session_id,kind,status,created_at,updated_at,document_id,result_json) "
            "VALUES(?,?,'ingest','error',1,1,?,?)", (jid, sid, did, db.dumps(result)),
        )


def chunk(did, page, text, number=0):
    with db.tx() as conn:
        conn.execute("INSERT INTO document_chunks VALUES(?,?,?,?,?)",
                     ("s_alice", did, f"{did}:{page}:{number}", text,
                      db.dumps({"page": page, "modality": "text", "chunk": number})))


def stored_document(did):
    return dict(db.get_conn().execute("SELECT * FROM documents WHERE id=?", (did,)).fetchone())


def test_openapi_registers_typed_success_responses_and_bounds(api):
    spec = api.client.get("/openapi.json").json()
    for path, method, code, model in (
        ("/api/documents/{document_id}/detail", "get", "200", "DocumentDetailResponse"),
        ("/api/documents/compare", "get", "200", "DocumentCompareResponse"),
        ("/api/usage", "get", "200", "UsageSnapshotResponse"),
        ("/api/documents/{document_id}/retry", "post", "202", "DocumentRetryResponse"),
        ("/api/documents/{document_id}/cancel", "post", "202", "DocumentCancelResponse"),
    ):
        response = spec["paths"][path][method]["responses"][code]
        assert response["content"]["application/json"]["schema"]["$ref"] == f"#/components/schemas/{model}"
    models = spec["components"]["schemas"]
    assert models["DocumentDetailResponse"]["properties"]["jobs"]["maxItems"] == 100
    compare = models["DocumentCompareResponse"]["properties"]
    assert compare["pages"]["maxItems"] == compare["limit"]["maximum"] == 50
    assert compare["max_chars"]["maximum"] == 200000
    assert models["ComparisonPageItem"]["properties"]["left_text"]["maxLength"] == 200000
    assert set(models["UsageCounters"]["properties"]) == {
        "documents", "storage_bytes", "daily_asks", "daily_pages",
    }


@pytest.mark.parametrize("result", [
    None,
    {"document_id": "doc", "filename": "report.pdf", "chunks": 3, "pages": 2,
     "pages_without_content": [2], "warnings": ["Some pages produced no usable content."]},
    {"document_id": "doc", "failed_page": None, "cleanup_required": True},
])
def test_detail_projects_jobs_like_polling_and_preserves_runner_results(api, monkeypatch, result):
    did = document()
    history_job("owned", did, result=result)
    history_job("foreign", did, sid="bob")
    original = lifecycle.get_job

    def extra_private_fields(jid):
        return {**original(jid), "namespace": "s_alice", "storage_path": "C:/private/upload.enc",
                "encrypted_pdf_path": "C:/private/upload.enc"}

    monkeypatch.setattr(lifecycle, "get_job", extra_private_fields)
    response = api.client.get(f"/api/documents/{did}/detail")
    assert response.status_code == 200
    payload = response.json()
    assert payload["jobs"] == [schemas.JobItem(**original("owned")).model_dump()]
    assert payload["jobs"][0]["result"] == result
    assert payload["jobs"][0]["cancel_requested"] is False
    assert set(payload["document"]) == set(schemas.DocumentItem.model_fields)
    for secret in ("session_id", "namespace", "storage_path", "encrypted_pdf_path", "C:/private", "private-hash"):
        assert secret not in response.text
    assert payload["total_jobs"] == 1 and payload["next_offset"] is None


def test_detail_job_history_is_bounded_deterministic_and_pageable(api):
    did = document()
    for i in range(103):
        history_job(f"job-{i:03}", did)
    first = api.client.get(f"/api/documents/{did}/detail").json()
    assert len(first["jobs"]) == first["limit"] == 100
    assert first["offset"] == 0 and first["total_jobs"] == 103 and first["next_offset"] == 100
    assert [j["id"] for j in first["jobs"]] == [f"job-{i:03}" for i in range(102, 2, -1)]
    second = api.client.get(f"/api/documents/{did}/detail", params={"offset": 100, "limit": 5}).json()
    assert [j["id"] for j in second["jobs"]] == ["job-002", "job-001", "job-000"]
    assert second["next_offset"] is None
    empty = api.client.get(f"/api/documents/{did}/detail", params={"offset": 200}).json()
    assert empty["jobs"] == [] and empty["next_offset"] is None


def test_detail_skips_missing_job_snapshot(api, monkeypatch):
    did = document()
    history_job("missing", did)
    monkeypatch.setattr(lifecycle, "get_job", MagicMock(return_value=None))
    payload = api.client.get(f"/api/documents/{did}/detail").json()
    assert payload["jobs"] == [] and payload["total_jobs"] == 1 and payload["next_offset"] is None


@pytest.mark.parametrize("budget", [1, 8, 200000])
def test_compare_hashes_suffix_beyond_preview_and_paginates_union(api, budget):
    left, right = document("left", status="ready"), document("right", status="ready")
    for did, suffix in ((left, "old"), (right, "new")):
        chunk(did, 1, suffix, number=1)  # Reverse insertion must not change chunk ordering.
        chunk(did, 1, "shared \u03bb" * 100, number=0)
    chunk(left, 2, "left only")
    chunk(right, 3, "right only")
    params = {"left_id": left, "right_id": right, "limit": 1, "max_chars": budget}
    response = api.client.get("/api/documents/compare", params=params)
    assert response.status_code == 200
    first = response.json()
    assert set(first) == set(schemas.DocumentCompareResponse.model_fields)
    assert first["total_pages"] == 3 and first["next_offset"] == 1
    assert first["pages"][0]["changed"] is True and first["truncated"] is True
    assert first["text_truncated"] is (budget < 200000)
    assert sum(len(p[side]) for p in first["pages"] for side in ("left_text", "right_text")) <= budget
    if budget < 200000:
        assert first["pages"][0]["left_text"] == first["pages"][0]["right_text"]
    last = api.client.get("/api/documents/compare", params={**params, "offset": 1, "limit": 2}).json()
    assert [p["page"] for p in last["pages"]] == [2, 3]
    assert all(p["changed"] for p in last["pages"]) and last["next_offset"] is None
    assert last["truncated"] is True
    assert sum(len(p[side]) for p in last["pages"] for side in ("left_text", "right_text")) <= budget
    empty = api.client.get("/api/documents/compare", params={**params, "offset": 10}).json()
    assert empty["pages"] == [] and empty["next_offset"] is None and not empty["text_truncated"]


def test_compare_identical_and_empty_corpus(api):
    left, right = document("left", status="ready"), document("right", status="ready")
    params = {"left_id": left, "right_id": right}
    empty = api.client.get("/api/documents/compare", params=params).json()
    assert empty["pages"] == [] and empty["total_pages"] == 0 and not empty["truncated"]
    for did in (left, right):
        chunk(did, 1, "same" * 100)
    result = api.client.get("/api/documents/compare", params={**params, "max_chars": 1}).json()
    assert not result["pages"][0]["changed"] and result["text_truncated"]


@pytest.mark.parametrize("path,params", [
    ("/api/documents/doc/detail", {"offset": -1}),
    ("/api/documents/doc/detail", {"limit": 0}),
    ("/api/documents/doc/detail", {"limit": 101}),
    ("/api/documents/compare", {"offset": -1}),
    ("/api/documents/compare", {"limit": 0}),
    ("/api/documents/compare", {"limit": 51}),
    ("/api/documents/compare", {"max_chars": 0}),
    ("/api/documents/compare", {"max_chars": 200001}),
])
def test_query_bounds_reject_invalid_values(api, path, params):
    document(status="ready")
    assert api.client.get(path, params={"left_id": "doc", "right_id": "doc", **params}).status_code == 422


def test_usage_matches_actual_snapshot_including_pending_and_failed_work(api, monkeypatch):
    document()
    document("other", sid="bob")
    usage.finish_ask(usage.reserve_ask("alice"), False)
    usage.finish_pages(usage.reserve_pages("alice", 3), False)
    usage.reserve_upload("alice", 7)
    usage.reserve_ask("bob")
    expected = usage.snapshot("alice")
    snapshot = MagicMock(return_value={**expected, "session_id": "alice",
                                      "usage": {**expected["usage"], "namespace": "s_alice"}})
    monkeypatch.setattr(lifecycle.usage, "snapshot", snapshot)
    response = api.client.get("/api/usage")
    assert response.status_code == 200 and response.json() == expected
    snapshot.assert_called_once_with("alice")
    assert expected["usage"] == {"documents": 2, "storage_bytes": 19, "daily_asks": 1, "daily_pages": 3}
    assert expected["pending_uploads"] == 1 and expected["pending_storage_bytes"] == 7
    assert expected["stored_documents"] == 1 and expected["stored_bytes"] == 12
    assert expected["remaining"]["daily_pages"] == 7 and expected["timezone"] == "UTC"


@pytest.mark.parametrize("model,payload", [
    (schemas.UsageCounters, {"documents": -1, "storage_bytes": 0, "daily_asks": 0, "daily_pages": 0}),
    (schemas.DocumentRetryResponse, {"document_id": "d", "job_id": "j", "status": "queued"}),
    (schemas.DocumentCancelResponse, {"document_id": "d", "job_id": "j", "cancel_requested": False}),
    (schemas.ComparisonPageItem, {"page": 0, "left_text": "", "right_text": "", "changed": False}),
])
def test_response_models_reject_invalid_contract_values(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_retry_and_cancel_preserve_public_payloads_and_linked_states(api, status):
    did = document(status=status)
    response = api.client.post(f"/api/documents/{did}/retry")
    assert response.status_code == 202
    jid = response.json()["job_id"]
    assert response.json() == {"document_id": did, "job_id": jid, "status": "pending"}
    api.submit.assert_called_once_with(jid, lifecycle.run_ingest, job_id=jid, session_id="alice",
                                      document_id=did, encrypted_pdf_path="C:/private/upload.enc",
                                      namespace="s_alice", filename="report.pdf")
    assert runner.get_job(jid)["status"] == "queued" and stored_document(did)["status"] == "pending"
    assert api.client.post(f"/api/documents/{did}/retry").status_code == 409
    cancelled = api.client.post(f"/api/documents/{did}/cancel")
    assert cancelled.status_code == 202
    assert cancelled.json() == {"document_id": did, "job_id": jid, "cancel_requested": True}
    assert runner.get_job(jid)["status"] == stored_document(did)["status"] == "cancelled"


@pytest.mark.parametrize("exception,message", [
    (runner.QueueFull, "Ingestion queue is full. Retry later."),
    (RuntimeError, "Job executor is unavailable. Retry later."),
    (ValueError, "Job executor is unavailable. Retry later."),
])
def test_retry_submission_failure_is_safe_503_and_terminalizes_active_work(api, exception, message):
    did = document()
    api.submit.side_effect = exception("C:/private/upload.enc namespace=s_alice SECRET")
    response = api.client.post(f"/api/documents/{did}/retry")
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    detail = response.json()["detail"]
    assert detail == {"message": message, "job_id": stored_document(did)["job_id"]}
    job = runner.get_job(detail["job_id"])
    assert job["status"] == job["stage"] == stored_document(did)["status"] == "error"
    assert job["message"] == stored_document(did)["last_error"] == message
    assert "SECRET" not in response.text and "C:/private" not in response.text and "s_alice" not in response.text
    api.submit.side_effect = None
    assert api.client.post(f"/api/documents/{did}/retry").status_code == 202


@pytest.mark.parametrize("failure", ["queue", "executor"])
def test_actual_runner_submission_failures_keep_safe_state_and_release_capacity(api, monkeypatch, failure):
    did = document()
    monkeypatch.setattr(lifecycle, "submit", runner.submit)
    if failure == "queue":
        assert runner._slots.acquire(blocking=False)
    else:
        runner._pool.submit.side_effect = RuntimeError("SECRET executor internals")
    try:
        response = api.client.post(f"/api/documents/{did}/retry")
        assert response.status_code == 503 and "SECRET" not in response.text
        jid = response.json()["detail"]["job_id"]
        assert runner.get_job(jid)["status"] == stored_document(did)["status"] == "error"
        assert runner.get_job(jid)["message"] == stored_document(did)["last_error"]
        assert jid not in runner._submitted
        if failure == "executor":
            assert runner._slots.acquire(blocking=False)
            runner._slots.release()
    finally:
        if failure == "queue":
            runner._slots.release()


@pytest.mark.parametrize("job_status,doc_status", [
    ("done", "ready"), ("cancelled", "cancelled"), ("error", "error"), ("error", "deleting"),
])
def test_submission_error_does_not_overwrite_already_terminal_states(api, job_status, doc_status):
    did = document()

    def finish_then_raise(jid, *_args, **_kwargs):
        runner.update_job(jid, status=job_status, stage="preserved", message="Existing safe outcome")
        with db.tx() as conn:
            conn.execute("UPDATE documents SET status=?,last_error=?,updated_at=7 WHERE id=?",
                         (doc_status, "Existing document outcome", did))
        before["job"] = runner.get_job(jid)
        before["document"] = stored_document(did)
        raise RuntimeError("SECRET")

    before = {}
    api.submit.side_effect = finish_then_raise
    response = api.client.post(f"/api/documents/{did}/retry")
    assert response.status_code == 503 and "SECRET" not in response.text
    assert runner.get_job(response.json()["detail"]["job_id"]) == before["job"]
    assert stored_document(did) == before["document"]


def test_failed_submission_does_not_mutate_newer_document_job(api):
    did = document()

    def replace_then_raise(jid, *_args, **_kwargs):
        runner._terminal(jid, "cancelled", "Ingestion cancelled.")
        replacement.append(runner.create_job("alice", "ingest", did))
        raise RuntimeError("SECRET")

    replacement = []
    api.submit.side_effect = replace_then_raise
    response = api.client.post(f"/api/documents/{did}/retry")
    assert response.status_code == 503
    assert runner.get_job(response.json()["detail"]["job_id"])["status"] == "cancelled"
    assert stored_document(did)["job_id"] == replacement[0]
    assert stored_document(did)["status"] == "pending" and stored_document(did)["last_error"] is None
    assert runner.get_job(replacement[0])["status"] == "queued"


def test_running_cancel_is_an_acknowledgement_not_terminal_completion(api):
    did = document()
    jid = api.client.post(f"/api/documents/{did}/retry").json()["job_id"]
    runner.update_job(jid, status="running")
    with db.tx() as conn:
        conn.execute("UPDATE documents SET status='processing' WHERE id=?", (did,))
    response = api.client.post(f"/api/documents/{did}/cancel")
    assert response.status_code == 202 and response.json()["cancel_requested"] is True
    assert runner.get_job(jid)["status"] == "running" and runner.get_job(jid)["cancel_requested"] == 1
    assert stored_document(did)["status"] == "processing"


def test_ownership_and_state_conflicts_fail_before_submission(api):
    document("other", sid="bob")
    document("ready", status="ready")
    document("pending", status="pending")
    for did in ("other", "missing"):
        assert api.client.get(f"/api/documents/{did}/detail").status_code == 404
        for action in ("retry", "cancel"):
            assert api.client.post(f"/api/documents/{did}/{action}").status_code == 404
        assert api.client.get("/api/documents/compare", params={"left_id": "ready", "right_id": did}).status_code == 404
    assert api.client.post("/api/documents/ready/retry").status_code == 409
    assert api.client.post("/api/documents/ready/cancel").status_code == 409
    assert api.client.post("/api/documents/pending/cancel").status_code == 409
    assert api.client.get("/api/documents/compare", params={"left_id": "ready", "right_id": "pending"}).status_code == 409
    api.submit.assert_not_called()