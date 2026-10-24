"""Focused offline answer-quality regressions. All provider calls are blocked."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from config import settings
from scripts import calibrate_confidence as calibration
from scripts import eval as evaluation
from src.extraction import pdf_extractor as extraction
from src.extraction import vision_summarizer as summaries
from src.rag import confidence, pipeline, query_rewrite
from src.rag.router import RoutingDecision


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected external/model call in offline test")

    from src.clients import groq_client
    from src.extraction import ocr
    from src.rag import router, text_chain, vision_chain
    for module, attr in [(groq_client, "chat"), (groq_client, "vision_chat"),
                         (confidence, "chat"), (query_rewrite, "chat"), (router, "chat"),
                         (text_chain, "chat"), (vision_chain, "vision_chat"),
                         (summaries, "vision_chat"), (ocr, "vision_chat")]:
        monkeypatch.setattr(module, attr, forbidden)
    monkeypatch.setattr(pipeline, "retrieve", forbidden)
    monkeypatch.setattr(pipeline, "compact", forbidden)
    monkeypatch.setattr(confidence, "get_embeddings", forbidden)
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    monkeypatch.setattr(settings, "image_dir", tmp_path)
    monkeypatch.setattr(settings, "confidence_mode", "fast")
    monkeypatch.setattr(settings, "confidence_low", 0.5)
    monkeypatch.setattr(settings, "confidence_high", 0.75)
    monkeypatch.setattr(summaries, "_CACHE_FILE", tmp_path / "summaries.json")


def evidence(**metadata):
    return Document(page_content="Revenue increased to 42 million dollars.", metadata={
        "evidence_id": "ev-real", "chunk_id": "chunk-real", "document_id": "doc-a",
        "filename": "report.pdf", "source": "report.pdf", "page": 2,
        "modality": "text", "bbox": None, **metadata})


def ready(monkeypatch, answer="Revenue increased to 42 million dollars [E1]."):
    monkeypatch.setattr(pipeline, "retrieve", Mock(return_value=[evidence()]))
    monkeypatch.setattr(pipeline, "compact", lambda question, docs: docs)
    monkeypatch.setattr(pipeline, "route", lambda *a: RoutingDecision("cheap", "cheap-model", {}, "test"))
    generate = Mock(return_value=answer)
    monkeypatch.setattr(pipeline, "answer_text", generate)
    monkeypatch.setattr(pipeline, "_grounding", Mock(return_value=0.9))
    monkeypatch.setattr(pipeline, "score_confidence", Mock(return_value=confidence.ConfidenceResult(
        0.9, "HIGH", {"grounding": 0.9, "agreement": 0.9})))
    return generate


@pytest.mark.parametrize("history", [None, [{"role": "user", "content": "Revenue?"}]])
def test_empty_evidence_abstains_without_model_calls(monkeypatch, history):
    retrieve = Mock(return_value=[])
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    rewrite = Mock(side_effect=AssertionError("No rewrite before evidence"))
    monkeypatch.setattr(pipeline, "rewrite", rewrite)
    result = pipeline.ask("What about next year?", namespace="owned", document_ids=["a"], history=history)
    assert result.abstained and not result.answerable
    assert result.abstention_reason == "no_evidence"
    assert result.status == "abstained" and result.model == ""
    assert result.confidence is None and result.sources == []
    assert retrieve.call_args.kwargs == {"namespace": "owned", "document_id": None, "document_ids": ["a"]}
    rewrite.assert_not_called()


def test_empty_after_compaction_abstains(monkeypatch):
    generate = ready(monkeypatch)
    monkeypatch.setattr(pipeline, "compact", lambda *a: [])
    assert pipeline.ask("Revenue?", namespace="n").abstention_reason == "no_evidence"
    generate.assert_not_called()


def test_sync_and_stream_share_full_verified_response(monkeypatch):
    ready(monkeypatch)
    sync = pipeline.ask("Revenue?", namespace="n").to_dict()
    events = list(pipeline.ask_stream("Revenue?", namespace="n"))
    done = events[-1]
    assert set(done) == {f.name for f in fields(pipeline.RAGResponse)} | {"type"}
    assert {k: v for k, v in done.items() if k not in {"type", "latency_ms"}} == {
        k: v for k, v in sync.items() if k != "latency_ms"}
    stages = [e.get("stage") for e in events]
    assert "verifying" in stages
    assert stages.index("verifying") < next(i for i, e in enumerate(events) if e["type"] == "token")
    assert "".join(e["text"] for e in events if e["type"] == "token") == sync["answer"]
    source = done["sources"][0]
    assert source["label"] == "E1" and source["evidence_id"] == "ev-real"
    assert source["quote"] == evidence().page_content and source["bbox"] is None and source["cited"]


@pytest.mark.parametrize("answer", ["Claim [E99].", "Claim [E1] and [E2].", "Claim without citation.", "Claim [E01]."])
def test_invalid_citations_never_stream_as_answer(monkeypatch, answer):
    generate = ready(monkeypatch, answer)
    events = list(pipeline.ask_stream("Revenue?", namespace="n"))
    done = events[-1]
    assert done["abstention_reason"] == "invalid_or_missing_citations"
    assert done["citation_warnings"] and done["abstained"]
    assert generate.call_count == 2
    assert all(e["text"] not in {answer} for e in events if e["type"] == "token")
    assert not done["sources"][0]["cited"]


@pytest.mark.parametrize("grounding,reason,attempts", [(None, "grounding_unavailable", 1),
                                                     (float("nan"), "grounding_unavailable", 1),
                                                     (0.1, "unsupported_answer", 2)])
def test_grounding_gate_is_conservative(monkeypatch, grounding, reason, attempts):
    generate = ready(monkeypatch)
    monkeypatch.setattr(pipeline, "_grounding", lambda *a: grounding)
    result = pipeline.ask("Revenue?", namespace="n")
    assert result.abstention_reason == reason and result.abstained
    assert generate.call_count == attempts


def test_low_confidence_retries_once_then_abstains(monkeypatch):
    generate = ready(monkeypatch)
    monkeypatch.setattr(pipeline, "score_confidence", lambda **k: confidence.ConfidenceResult(0.3, "LOW", {}))
    result = pipeline.ask("Revenue?", namespace="n")
    assert generate.call_count == 2 and result.escalated
    assert result.abstention_reason == "low_confidence"


def test_failed_measurement_is_not_answerable(monkeypatch):
    ready(monkeypatch)
    monkeypatch.setattr(pipeline, "score_confidence", lambda **k: confidence.ConfidenceResult(
        None, None, {}, False, "measurement_unavailable:agreement"))
    result = pipeline.ask("Revenue?", namespace="n")
    assert result.abstention_reason == "confidence_unavailable"
    assert result.confidence is None and not result.confidence_available


def test_history_rewrite_and_identical_answer_scoring_context(monkeypatch):
    generate = ready(monkeypatch)
    rewrite = Mock(return_value="Acme revenue in 2025")
    monkeypatch.setattr(pipeline, "rewrite", rewrite)
    history = [{"role": "user", "content": "Acme revenue?"}]
    result = pipeline.ask("And in 2025?", namespace="n", history=history, document_ids=["doc-a"])
    assert result.question == "And in 2025?" and result.retrieval_query == "Acme revenue in 2025"
    assert "Acme revenue?" in pipeline.retrieve.call_args_list[0].args[0]
    assert pipeline.retrieve.call_args_list[1].args[0] == result.retrieval_query
    generated = generate.call_args.kwargs
    scored = pipeline.score_confidence.call_args.kwargs
    assert generated["question"] == scored["question"] == result.question
    #assert generated["context"] == scored["context"]
    assert "[E1]" in generated["context"] and "NOT EVIDENCE" in generated["context"]
    assert "[E1]" in scored["context"] and "Text-only cited-set verification" in scored["context"]


def test_rewrite_failure_has_deterministic_history_fallback(monkeypatch):
    monkeypatch.setattr(query_rewrite, "chat", Mock(side_effect=RuntimeError("offline")))
    history = [{"role": "user", "content": "Acme"}]
    assert query_rewrite.rewrite("Its growth?", history) == query_rewrite.standalone_query("Its growth?", history)


def test_vision_stream_has_full_metadata_and_explicit_uncertainty(monkeypatch, tmp_path):
    ready(monkeypatch)
    image = tmp_path / "figure.png"
    image.write_bytes(b"offline")
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **k: [evidence(modality="visual", image_path=str(image))])
    monkeypatch.setattr(pipeline, "route", lambda *a: RoutingDecision("vision", "vision-model", {}, "test"))
    monkeypatch.setattr(pipeline, "answer_vision", lambda **k: "Revenue is 42 million dollars [E1].")
    monkeypatch.setattr(pipeline, "score_confidence", confidence.score_confidence)
    done = list(pipeline.ask_stream("Revenue?", namespace="n"))[-1]
    assert done["answerable"] and done["confidence"] is None
    assert done["confidence_unavailable_reason"] == "vision_not_measured"
    assert done["sources"][0]["cited"] and done["image_paths"] == [str(image)]


@pytest.mark.parametrize("payload", ['{}', '{"score": null}', '{"score": NaN}', '{"score": 2}', '{"score": true}', 'not json'])
def test_bad_grounding_never_defaults_to_half(monkeypatch, payload):
    monkeypatch.setattr(confidence, "chat", lambda **k: payload)
    assert confidence._grounding("ctx", "answer") is None


def test_zero_grounding_is_valid(monkeypatch):
    monkeypatch.setattr(confidence, "chat", lambda **k: '{"score": 0}')
    assert confidence._grounding("ctx", "answer") == 0


def test_modes_and_missing_samples(monkeypatch):
    monkeypatch.setattr(settings, "confidence_mode", "off")
    assert confidence.score_confidence(model="m", context="c", question="q", answer="a") is None
    monkeypatch.setattr(settings, "confidence_mode", "fast")
    monkeypatch.setattr(confidence, "_sample_answers", lambda *a: [])
    result = confidence.score_confidence(model="m", context="c", question="q", answer="a", grounding=0.8)
    assert result.score is None and not result.available
    assert result.components == {"grounding": 0.8}
    monkeypatch.setattr(settings, "confidence_mode", "full")
    monkeypatch.setattr(confidence, "_sample_answers", lambda model, context, question, n, system: ["a"] * n)
    monkeypatch.setattr(confidence, "_agreement", lambda *a: 0.8)
    monkeypatch.setattr(confidence, "_contrast", lambda *a: 0.8)
    result = confidence.score_confidence(model="m", context="c", question="q", answer="a", grounding=0.8)
    assert result.score == 0.8 and set(result.components) == confidence.FEATURES["full"]


def artifact():
    stats = {"n": 4, "positive": 2, "negative": 2, "brier": 0.2, "accuracy": 0.75}
    return {"schema_version": 2, "feature_version": confidence.FEATURE_VERSION,
            "target": "user_preference", "mode": "fast", "method": "logistic",
            "keys": ["grounding", "agreement"], "component_weights": {"grounding": 2, "agreement": 1},
            "intercept": -1, "metrics": {"train": stats, "heldout": stats}}


@pytest.mark.parametrize("change", [{"schema_version": 1}, {"intercept": float("inf")},
                                   {"mode": "full"}, {"target": "truth"}, {"keys": ["grounding"]},
                                   {"feature_version": "legacy"}, {"metrics": {}}])
def test_calibration_rejects_incompatible_or_invalid_artifacts(change):
    data = {**artifact(), **change}
    (settings.cache_dir / "confidence_weights.json").write_text(json.dumps(data), encoding="utf-8")
    assert confidence.load_calibration("fast", confidence.FEATURES["fast"]) is None


@pytest.mark.parametrize("data", [[], None, "wrong shape", {"keys": [], "metrics": []}])
def test_calibration_wrong_json_shape_is_unavailable(data):
    (settings.cache_dir / "confidence_weights.json").write_text(json.dumps(data), encoding="utf-8")
    assert confidence.load_calibration("fast", confidence.FEATURES["fast"]) is None


def test_calibration_loaded_but_preference_does_not_replace_grounding(monkeypatch):
    (settings.cache_dir / "confidence_weights.json").write_text(json.dumps(artifact()), encoding="utf-8")
    monkeypatch.setattr(confidence, "_sample_answers", lambda *a: ["a", "a"])
    monkeypatch.setattr(confidence, "_agreement", lambda *a: 0.8)
    result = confidence.score_confidence(model="m", context="c", question="q", answer="a", grounding=0.8)
    assert result.score == 0.8 and result.calibration["target"] == "user_preference"
    assert 0 < result.calibration["preference_score"] < 1
    assert result.calibration["correctness_probability"] is False


def test_extraction_text_table_metadata_and_redaction():
    first = extraction._text_docs("Contact secret@example.com", 2, "original.pdf", "doc-a")[0]
    again = extraction._text_docs("Contact secret@example.com", 2, "original.pdf", "doc-a")[0]
    other = extraction._text_docs("Contact secret@example.com", 2, "original.pdf", "doc-b")[0]
    assert first.metadata["evidence_id"] == again.metadata["evidence_id"] != other.metadata["evidence_id"]
    assert first.metadata["bbox"] is None and first.metadata["location_precision"] == "page"
    assert "secret@example.com" not in first.page_content
    table = SimpleNamespace(bbox=(1, 2, 30, 40), to_pandas=lambda: SimpleNamespace(
        empty=False, to_markdown=lambda **k: "secret@example.com | 42"))
    page = SimpleNamespace(find_tables=lambda: SimpleNamespace(tables=[table]))
    doc = extraction._table_docs(page, 2, "original.pdf", "doc-a")[0]
    assert doc.metadata["bbox"] == [1, 2, 30, 40]
    assert "secret@example.com" not in doc.page_content
    assert doc.metadata["evidence_id"] != first.metadata["evidence_id"]


def test_registered_original_filename_not_temporary_decryption_name(monkeypatch, tmp_path):
    from src.core import db
    pdf_path = tmp_path / "opaque.dec.pdf"
    pdf_path.write_bytes(b"mock pdf")
    page = SimpleNamespace(get_text=lambda *a: "Revenue increased to 42 million dollars.",
                           find_tables=lambda: SimpleNamespace(tables=[]), get_images=lambda **k: [])

    class PDF:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return page

        def close(self):
            pass

    monkeypatch.setattr(extraction.fitz, "open", lambda *a: PDF())
    monkeypatch.setattr(settings, "ocr_min_chars_per_page", 0)
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = {"filename": "Original Report.pdf"}
    monkeypatch.setattr(db, "get_conn", lambda: conn)
    docs = extraction.extract_documents(pdf_path, document_id="doc-a")
    assert docs[0].metadata["source"] == "Original Report.pdf"
    assert docs[0].metadata["filename"] == "Original Report.pdf"
    assert docs[0].metadata["bbox"] is None


def test_visual_placements_have_distinct_ids_boxes_and_redacted_text(monkeypatch):
    pdf = SimpleNamespace(extract_image=lambda x: {"ext": "png", "image": b"fake"})
    page = SimpleNamespace(get_images=lambda **k: [(7,)], get_image_rects=lambda x: [(0, 0, 20, 20), (20, 0, 40, 20)])
    monkeypatch.setattr(extraction, "is_worth_processing", lambda *a: True)
    summarize = Mock(return_value="Contact secret@example.com")
    monkeypatch.setattr(extraction, "summarize_visual", summarize)
    seen = set()
    docs = extraction._image_docs(pdf, page, 1, "secret@example.com", "r.pdf", "doc", seen)
    docs += extraction._image_docs(pdf, page, 2, "secret@example.com", "r.pdf", "doc", seen)
    assert len({d.metadata["evidence_id"] for d in docs}) == 4
    assert docs[0].metadata["bbox"] == [0, 0, 20, 20]
    assert all("secret@example.com" not in d.page_content for d in docs)
    assert all(d.metadata["image_redacted"] is False for d in docs)
    assert "secret@example.com" not in summarize.call_args.kwargs["context_snippet"]


def test_cache_scope_context_model_version_and_redaction(monkeypatch, tmp_path):
    image = tmp_path / "fake.png"
    image.write_bytes(b"same bytes")
    generate = Mock(return_value="Contact secret@example.com")
    monkeypatch.setattr(summaries, "vision_chat", generate)
    monkeypatch.setattr(summaries, "image_to_data_uri", lambda p: "data:offline")
    for parent, context in [("doc-a", "ctx"), ("doc-a", "ctx"), ("doc-b", "ctx"), ("doc-a", "other")]:
        assert "secret@example.com" not in summaries.summarize_visual(image, 1, context, parent_identity=parent)
    assert generate.call_count == 3
    monkeypatch.setattr(settings, "model_vision", "different-model")
    summaries.summarize_visual(image, 1, "ctx", parent_identity="doc-a")
    monkeypatch.setattr(summaries, "_PROMPT_VERSION", "different-version")
    summaries.summarize_visual(image, 1, "ctx", parent_identity="doc-a")
    assert generate.call_count == 5
    assert len(json.loads(summaries._CACHE_FILE.read_text())) == 5
    assert "secret@example.com" not in summaries._CACHE_FILE.read_text()
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))


def test_concurrent_cache_updates_do_not_lose_keys(monkeypatch, tmp_path):
    image = tmp_path / "fake.png"
    image.write_bytes(b"same")
    monkeypatch.setattr(summaries, "vision_chat", lambda **k: "Summary")
    monkeypatch.setattr(summaries, "image_to_data_uri", lambda p: "data:offline")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: summaries.summarize_visual(image, 1, parent_identity=str(i)), range(8)))
    assert results == ["Summary"] * 8 and len(summaries._load_cache()) == 8


def test_failed_atomic_cache_replace_preserves_existing_cache(monkeypatch):
    summaries._CACHE_FILE.write_text('{"existing": "safe"}', encoding="utf-8")
    monkeypatch.setattr(summaries.os, "replace", Mock(side_effect=OSError("simulated")))
    with pytest.raises(OSError):
        with summaries._cache_lock():
            summaries._save_cache({"new": "summary"})
    assert summaries._load_cache() == {"existing": "safe"}
    assert not list(settings.cache_dir.glob("*.tmp"))
    assert not list(settings.cache_dir.glob("*.lock"))


def test_collect_owned_latest_votes_preserves_zero(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE conversations(id TEXT, session_id TEXT);
        CREATE TABLE messages(id INTEGER, conversation_id TEXT, role TEXT, meta_json TEXT);
        CREATE TABLE feedback(id INTEGER, session_id TEXT, conversation_id TEXT, message_id INTEGER, vote INTEGER, created_at REAL);
        INSERT INTO conversations VALUES ('c', 'owner');
        INSERT INTO feedback VALUES (1, 'owner', NULL, 1, -1, 1), (2, 'owner', NULL, 1, 1, 2),
                                    (3, 'intruder', NULL, 1, -1, 3), (4, 'owner', 'other', 1, -1, 4);
    """)
    meta = {"confidence": 0, "confidence_components": {"grounding": 0, "agreement": 0},
            "calibration": {"feature_version": confidence.FEATURE_VERSION, "mode": "fast"}}
    conn.execute("INSERT INTO messages VALUES (1, 'c', 'assistant', ?)", (json.dumps(meta),))
    monkeypatch.setattr(calibration, "get_conn", lambda: conn)
    rows = calibration.collect("fast")
    assert len(rows) == 1 and rows[0]["vote"] == 1 and rows[0]["conf"] == 0
    conn.close()


def training_data():
    return [{"mid": group * 6 + i, "group": str(group), "vote": i % 2,
             "comps": {"grounding": (i + 1) / 8, "agreement": (group + i) / 12}}
            for group in range(4) for i in range(6)]


def test_calibration_has_heldout_metrics_and_rejects_one_class():
    data = training_data()
    result = calibration.fit_artifact(data, "fast")
    assert result["target"] == "user_preference" and result["schema_version"] == 2
    assert result["metrics"]["train"]["n"] + result["metrics"]["heldout"]["n"] == 24
    for row in data:
        row["vote"] = 1
    with pytest.raises(ValueError, match="positive and negative"):
        calibration.fit_artifact(data, "fast")


def test_eval_missing_metrics_and_actual_previews(monkeypatch, tmp_path):
    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps([{"question": "Revenue?", "expected_evidence_ids": ["ev-real", "other"]}]))
    result = pipeline.RAGResponse("Revenue?", "Revenue increased to 42 million dollars.", "cheap", "m",
                                  sources=pipeline._sources([evidence()]))
    monkeypatch.setattr(evaluation, "ask", lambda *a, **k: result)
    report = evaluation.run(dataset, "n", None)
    row = report["rows"][0]
    assert row["context_precision"] == 1
    assert row["evidence_precision"] == 1 and row["evidence_recall"] == 0.5
    assert row["faithfulness_proxy"] is None and row["reference_overlap"] is None
    assert row["keyword_recall"] is None
    assert evaluation._evidence_metrics([], None)["evidence_recall"] is None
    assert evaluation._context_precision("answer", [""]) is None
    assert evaluation._evidence_metrics([{}], ["known"])["evidence_precision"] is None