"""Lifecycle regression tests: isolated SQLite and mocked external services."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from config import settings
from src.core import db, usage
from src.jobs import ingest_job, runner
from src.rag import hybrid, retriever
from src.vectorstore import pinecone_store


@pytest.fixture
def local(monkeypatch, tmp_path):
    db.close_connection()
    monkeypatch.setattr(settings, "sqlite_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "image_dir", tmp_path / "images")
    settings.image_dir.mkdir()
    monkeypatch.setattr(settings, "tenant_max_documents", 100)
    monkeypatch.setattr(settings, "tenant_max_storage_mb", 1)
    monkeypatch.setattr(settings, "tenant_daily_asks", 2)
    monkeypatch.setattr(settings, "tenant_daily_pages", 10)
    monkeypatch.setattr(settings, "max_pdf_pages", 5)
    monkeypatch.setattr(settings, "job_max_pending", 1)
    monkeypatch.setattr(runner, "_initialized_path", None)
    monkeypatch.setattr(runner, "_submitted", set())
    for sid in ("alice", "bob"):
        db.get_conn().execute("INSERT INTO sessions VALUES(?,?,?)", (sid, db.now(), db.now()))
    runner.initialize_runner()
    yield tmp_path
    db.close_connection()


def document(sid="alice", status="ready", filename="report.pdf", size=12):
    did = ingest_job.register_document(session_id=sid, filename=filename,
                                      storage_path="unused.enc", size_bytes=size, sha256="hash")
    db.get_conn().execute("UPDATE documents SET status=? WHERE id=?", (status, did))
    return did


def chunk(did, text="alpha content", page=1, **meta):
    return Document(page_content=text, metadata={"document_id": did, "page": page, "modality": "text", **meta})


def test_transactions_rollback_and_nested_savepoints(local):
    with pytest.raises(RuntimeError):
        with db.tx() as conn:
            conn.execute("UPDATE sessions SET last_seen=1 WHERE id='alice'")
            with db.tx() as nested:
                assert nested is conn
                nested.execute("UPDATE sessions SET last_seen=2 WHERE id='bob'")
            raise RuntimeError()
    assert db.get_conn().execute("SELECT last_seen FROM sessions WHERE id='alice'").fetchone()[0] != 1


def test_additive_migration_preserves_old_documents(local, monkeypatch):
    db.close_connection()
    legacy = local / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.executescript("""
        CREATE TABLE sessions(id TEXT PRIMARY KEY,created_at REAL NOT NULL,last_seen REAL NOT NULL);
        INSERT INTO sessions VALUES('alice',1,1);
        CREATE TABLE documents(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,filename TEXT NOT NULL,
            version INTEGER NOT NULL,size_bytes INTEGER NOT NULL,storage_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,pages INTEGER,chunks INTEGER,created_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending');
        INSERT INTO documents VALUES('legacy','alice','old.pdf',1,12,'old.enc','hash',1,2,1,'ready');
    """)
    conn.close()
    monkeypatch.setattr(settings, "sqlite_path", legacy)
    row = db.get_conn().execute("SELECT * FROM documents WHERE id='legacy'").fetchone()
    assert row["status"] == "ready" and row["updated_at"] == 1 and row["attempt_count"] == 0
    assert db.get_conn().execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0] == 0


def test_connections_are_per_thread(local):
    main = db.get_conn()
    barrier = threading.Barrier(2)

    def connection():
        conn = db.get_conn()
        barrier.wait(timeout=5)
        identity = id(conn)
        assert conn is not main
        db.close_connection()
        return identity

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(connection) for _ in range(2)]
        assert len({f.result() for f in futures}) == 2


def test_corpus_survives_new_store_and_connection(local):
    did = document()
    hybrid.BM25Store().upsert("s_alice", did, [chunk(did)])
    db.close_connection()
    hits = hybrid.BM25Store().search("s_alice", did, "alpha", 5)
    assert [d.page_content for d in hits] == ["alpha content"]


def test_corpus_filters_ready_owner_multiple_and_empty(local):
    left, right, pending = document(), document(), document(status="processing")
    store = hybrid.BM25Store()
    for did in (left, right, pending):
        store.upsert("s_alice", did, [chunk(did)])
    hits = store.search("s_alice", query="alpha", document_ids=[left, right, pending])
    assert {d.metadata["document_id"] for d in hits} == {left, right}
    assert store.search("s_alice", query="alpha", document_ids=[]) == []
    assert store.search("s_bob", query="alpha") == []
    with pytest.raises(PermissionError):
        store.upsert("s_bob", left, [chunk(left)])
    with pytest.raises(PermissionError):
        store.upsert("demo", left, [chunk(left)])


def test_empty_tokenless_and_legacy_corpus(local):
    store = hybrid.BM25Store()
    store.upsert("demo", "cli", [])
    assert store.search("demo", "cli", "alpha", 5) == []
    store.upsert("demo", "cli", [chunk("cli", "!!!")])
    assert store.search("demo", "cli", "alpha", 5) == []
    store.upsert("demo", "cli", [chunk("cli")])
    assert store.search("demo", query="alpha")
    store.delete("demo", "cli")
    assert not store.has("demo", "cli")


def test_stable_identity_preserves_distinct_images():
    a, b = chunk("d", image_path="a.png"), chunk("d", image_path="b.png")
    assert hybrid._key(a) != hybrid._key(b)
    assert pinecone_store._doc_id(a) == pinecone_store._doc_id(a)
    assert pinecone_store._doc_id(a) != pinecone_store._doc_id(b)
    assert hybrid._key(chunk("d", chunk_id="x")) == hybrid._key(chunk("d", "updated", chunk_id="x"))


def test_retrieval_empty_scope_and_no_hyde_without_evidence(local, monkeypatch):
    dense, hyde = MagicMock(return_value=[]), MagicMock()
    monkeypatch.setattr(retriever, "dense_search", dense)
    monkeypatch.setattr(retriever, "hyde", hyde)
    assert retriever.retrieve("alpha", namespace="demo", document_ids=[]) == []
    dense.assert_not_called()
    assert retriever.retrieve("alpha", namespace="demo") == []
    hyde.assert_not_called()


def test_retrieval_removes_unowned_and_inactive_dense_results(local, monkeypatch):
    good, other, inactive = document(), document("bob"), document(status="error")
    dense = MagicMock(return_value=[chunk(good), chunk(other), chunk(inactive)])
    monkeypatch.setattr(retriever, "dense_search", dense)
    monkeypatch.setattr(retriever, "hyde", lambda _: None)
    result = retriever.retrieve("alpha", namespace="s_alice", document_ids=[good, other, inactive])
    assert [d.metadata["document_id"] for d in result] == [good]
    assert dense.call_args.kwargs["document_ids"] == [good]


def test_dense_multi_filter_and_empty_short_circuit(monkeypatch):
    vs = MagicMock()
    vs.similarity_search.return_value = [chunk("a"), chunk("b"), chunk("outside")]
    factory = MagicMock(return_value=vs)
    monkeypatch.setattr(pinecone_store, "get_vectorstore", factory)
    assert pinecone_store.dense_search("q", "demo", 5, document_ids=[]) == []
    factory.assert_not_called()
    result = pinecone_store.dense_search("q", "demo", 5, document_ids=["a", "b"])
    assert len(result) == 2
    assert vs.similarity_search.call_args.kwargs["filter"] == {"document_id": {"$in": ["a", "b"]}}


def test_delete_provider_failure_propagates_and_retains_corpus(local, monkeypatch):
    hybrid.bm25_store.upsert("demo", "cli", [chunk("cli")])
    index = MagicMock()
    index.delete.side_effect = RuntimeError("secret provider details")
    monkeypatch.setattr(pinecone_store, "get_index", lambda _: index)
    monkeypatch.setattr(pinecone_store, "ensure_index",lambda *a, **k: None)
    monkeypatch.setattr(pinecone_store, "embedding_dimension",lambda: 384)
    with pytest.raises(Exception, match="retry required"):
        pinecone_store.delete_document("demo", "cli")
    assert hybrid.bm25_store.has("demo", "cli")
    with pytest.raises(Exception, match="retry required"):
        pinecone_store.clear_namespace("demo")


def test_upsert_metadata_roundtrip_and_cli_corpus(local, monkeypatch):
    vs = MagicMock()
    monkeypatch.setattr(pinecone_store, "get_vectorstore", lambda _: vs)
    source = chunk("cli", bbox=[1, 2, 3, 4], optional=None)
    assert pinecone_store.upsert_documents([source], "demo") == 1
    wire = vs.add_documents.call_args.args[0][0]
    assert "bbox" not in wire.metadata and "optional" not in wire.metadata
    assert hybrid.bm25_store.has("demo", "cli")
    vs.similarity_search.return_value = [wire]
    restored = pinecone_store.dense_search("alpha", "demo", 5)[0]
    assert restored.metadata["bbox"] == [1, 2, 3, 4]


def test_corpus_delete_cascades_and_owner_prefix_collision_fails_closed(local):
    did = document()
    hybrid.bm25_store.upsert("s_alice", did, [chunk(did)])
    db.get_conn().execute("DELETE FROM documents WHERE id=?", (did,))
    assert not hybrid.bm25_store.has("s_alice", did)
    prefix = "a" * 20
    for sid in (prefix + "one", prefix + "two"):
        db.get_conn().execute("INSERT INTO sessions VALUES(?,?,?)", (sid, db.now(), db.now()))
    assert db.allowed_document_ids("s_" + prefix) == []


def test_ask_reservations_atomic_no_refunds(local):
    def reserve(_):
        try:
            rid = usage.reserve_ask("alice")
            usage.finish_ask(rid, False)
            return True
        except usage.BudgetExceeded:
            return False
        finally:
            db.close_connection()
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(reserve, range(8))) == 2
    assert usage.snapshot("alice")["remaining"]["daily_asks"] == 0


def test_upload_transfer_no_double_count_and_release(local, monkeypatch):
    monkeypatch.setattr(settings, "tenant_max_documents", 1)
    rid = usage.reserve_upload("alice", 123)
    with pytest.raises(usage.BudgetExceeded):
        usage.reserve_upload("alice", 1)
    did = ingest_job.register_document(session_id="alice", filename="a.pdf", storage_path="a.enc",
                                      size_bytes=123, sha256="hash", reservation_id=rid)
    usage.finish_upload(rid, did)
    state = usage.snapshot("alice")
    assert state["usage"]["documents"] == 1
    assert state["usage"]["storage_bytes"] == 123
    assert state["pending_uploads"] == 0
    db.get_conn().execute("DELETE FROM documents WHERE id=?", (did,))
    rid = usage.reserve_upload("alice", 2)
    usage.finish_upload(rid, None)
    assert usage.snapshot("alice")["usage"]["documents"] == 0


def test_storage_pages_and_reservation_ownership(local):
    with pytest.raises(usage.BudgetExceeded):
        usage.reserve_upload("alice", 1024 * 1024 + 1)
    for _ in range(2):
        usage.finish_pages(usage.reserve_pages("alice", 5), False)
    with pytest.raises(usage.BudgetExceeded):
        usage.reserve_pages("alice", 1)
    with pytest.raises(usage.BudgetExceeded):
        usage.reserve_pages("bob", 6)
    rid = usage.reserve_upload("alice", 12)
    with pytest.raises(ValueError):
        usage.finish_upload(rid, document("bob"))


def test_startup_reconciles_without_replay(local):
    did = document(status="pending")
    jid = runner.create_job("alice", "ingest", did)
    runner.update_job(jid, status="running")
    assert runner.reconcile_interrupted_jobs() == 1
    assert runner.get_job(jid)["status"] == "error"
    row = db.get_conn().execute("SELECT * FROM documents WHERE id=?", (did,)).fetchone()
    assert row["status"] == "error" and "restart" in row["last_error"]


def test_bounded_queue_cancel_and_observability(local, monkeypatch):
    queued = []
    pool = MagicMock()
    pool.submit.side_effect = lambda target: queued.append(target)
    monkeypatch.setattr(runner, "_pool", pool)
    a, b = runner.create_job("alice", "test"), runner.create_job("alice", "test")
    target = MagicMock()
    runner.submit(a, target)
    with pytest.raises(runner.QueueFull):
        runner.submit(b, target)
    assert runner.get_job(b)["status"] == "error"
    assert not runner.request_cancel(a, "bob")
    assert runner.request_cancel(a, "alice")
    queued.pop()()
    target.assert_not_called()
    assert runner.get_job(a)["status"] == "cancelled"
    assert list(runner.watch(a))[-1]["status"] == "cancelled"
    c = runner.create_job("alice", "test")
    runner.submit(c, target)
    queued.pop()()
    assert runner.get_job(c)["status"] == "done"


@pytest.fixture
def ingest(local, monkeypatch):
    did = document(status="pending")
    jid = runner.create_job("alice", "ingest", did)
    temp = local / "decrypted.pdf"
    temp.write_bytes(b"mock")
    monkeypatch.setattr(ingest_job, "_write_temp_pdf", lambda _: temp)
    pdf = MagicMock()
    pdf.__enter__.return_value = pdf
    pdf.__len__.return_value = 2
    monkeypatch.setattr(ingest_job.fitz, "open", lambda _: pdf)
    extract = MagicMock(return_value=[chunk(did)])
    monkeypatch.setattr(ingest_job, "extract_documents", extract)
    monkeypatch.setattr(ingest_job, "delete_document", MagicMock())
    monkeypatch.setattr(ingest_job, "upsert_documents", MagicMock(return_value=1))
    return {"job_id": jid, "session_id": "alice", "document_id": did,
            "encrypted_pdf_path": "unused.enc", "namespace": "s_alice", "filename": "report.pdf"}, temp, extract


def test_ingest_publishes_atomically_and_preserves_original_name(ingest):
    args, temp, extract = ingest
    ingest_job.run_ingest(**args)
    row = db.get_conn().execute("SELECT * FROM documents WHERE id=?", (args["document_id"],)).fetchone()
    assert row["status"] == "ready" and row["attempt_count"] == 1
    assert row["pages"] == 2
    assert extract.call_args.kwargs["original_source"] == "report.pdf"
    assert runner.get_job(args["job_id"])["result"]["pages_without_content"] == [2]
    assert hybrid.bm25_store.has("s_alice", args["document_id"])
    assert not temp.exists()


@pytest.mark.parametrize("empty", [True, False])
def test_ingest_failure_never_publishes_and_hides_provider_errors(ingest, monkeypatch, empty):
    args, temp, extract = ingest
    if empty:
        extract.return_value = []
    else:
        monkeypatch.setattr(ingest_job, "upsert_documents", MagicMock(side_effect=RuntimeError("SECRET_TOKEN")))
    with pytest.raises(Exception):
        ingest_job.run_ingest(**args)
    row = db.get_conn().execute("SELECT * FROM documents WHERE id=?", (args["document_id"],)).fetchone()
    assert row["status"] == "error"
    assert "SECRET_TOKEN" not in row["last_error"]
    assert not hybrid.bm25_store.has("s_alice", args["document_id"])
    assert not temp.exists()


def test_cancel_before_publish_cleans_up(ingest, monkeypatch):
    args, temp, _ = ingest
    runner.update_job(args["job_id"], status="running")
    def upsert(*_args, **_kwargs):
        runner.request_cancel(args["job_id"], "alice")
        return 1
    monkeypatch.setattr(ingest_job, "upsert_documents", upsert)
    with pytest.raises(runner.JobCancelled):
        ingest_job.run_ingest(**args)
    assert runner.get_job(args["job_id"])["status"] == "cancelled"
    assert not temp.exists()
    assert db.get_conn().execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0] == 0


def test_cancel_signal_escapes_extractor_exception_guard(ingest):
    from src.extraction.pdf_extractor import _emit
    args, _, extract = ingest
    runner.update_job(args["job_id"], status="running")
    def cancelled_extract(*_args, **kwargs):
        runner.request_cancel(args["job_id"], "alice")
        _emit(kwargs["progress"], 0, 2, "page 1: text")
        pytest.fail("Cancellation was swallowed by the extractor")
    extract.side_effect = cancelled_extract
    with pytest.raises(runner.JobCancelled):
        ingest_job.run_ingest(**args)
    assert runner.get_job(args["job_id"])["status"] == "cancelled"


def test_secure_temporary_paths_are_unique(local, monkeypatch):
    monkeypatch.setattr(ingest_job, "decrypt_file", lambda _: b"pdf")
    a = ingest_job._write_temp_pdf(local / "file.enc")
    b = ingest_job._write_temp_pdf(local / "file.enc")
    try:
        assert a != b and a.read_bytes() == b"pdf"
    finally:
        a.unlink()
        b.unlink()


@pytest.fixture
def client(local):
    from src.api import lifecycle
    app = FastAPI()
    app.include_router(lifecycle.router)
    app.include_router(lifecycle.usage_router)
    app.dependency_overrides[lifecycle.session_ctx] = lambda: {"session_id": "alice", "namespace": "s_alice"}
    with TestClient(app) as client:
        yield client


def test_detail_ownership_and_bounded_comparison(client):
    left, right, other = document(), document(), document("bob")
    for did, text in ((left, "alpha " * 100), (right, "alpha " * 99 + "changed")):
        hybrid.bm25_store.upsert("s_alice", did, [chunk(did, text), chunk(did, "second page", 2)])
    assert client.get(f"/api/documents/{other}/detail").status_code == 404
    detail = client.get(f"/api/documents/{left}/detail").json()
    assert "storage_path" not in detail["document"]
    payload = client.get("/api/documents/compare", params={"left_id": left, "right_id": right,
                                                          "max_chars": 20, "limit": 1}).json()
    assert payload["pages"][0]["changed"] is True
    assert sum(len(p["left_text"]) + len(p["right_text"]) for p in payload["pages"]) <= 20
    assert payload["truncated"] and payload["next_offset"] == 1
    assert client.get("/api/documents/compare", params={"left_id": left, "right_id": other}).status_code == 404
    assert client.get("/api/usage").json()["limits"]["daily_asks"] == 2


def test_retry_state_and_owned_cancel(client, monkeypatch):
    from src.api import lifecycle
    ready, failed = document(), document(status="error")
    monkeypatch.setattr(lifecycle, "submit", MagicMock())
    assert client.post(f"/api/documents/{ready}/retry").status_code == 409
    response = client.post(f"/api/documents/{failed}/retry")
    assert response.status_code == 202
    jid = response.json()["job_id"]
    assert runner.get_job(jid)["document_id"] == failed
    assert client.post(f"/api/documents/{failed}/cancel").status_code == 202
    assert runner.get_job(jid)["cancel_requested"] == 1