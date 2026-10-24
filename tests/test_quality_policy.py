"""Isolated pipeline policy tests: dependency doubles, no providers or model loads.

Load the real pipeline under a private name with import-boundary stubs. These
tests exercise orchestration, not confidence algorithms or SDK integration.
"""
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@dataclass
class Evidence:
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Decision:
    tier: str
    model: str
    classification: dict
    reason: str


@pytest.fixture
def policy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected dependency call in isolated policy test")

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    settings = SimpleNamespace(max_question_chars=4000, conversation_memory_turns=3,
                               confidence_mode="off", confidence_low=0.5)
    stub("config", settings=settings)
    stub("langchain_core.documents", Document=Evidence)
    stub("src.core.exceptions", ValidationError=ValueError)
    stub("src.core.logger", get_logger=lambda name: Mock())
    stub("src.core.prompt_safety", looks_injected=lambda text: False,
         wrap_untrusted=lambda label, text: f"<<<{label}_BEGIN>>>\n{text}\n<<<{label}_END>>>")
    stub("src.rag.compaction", compact=Mock(side_effect=forbidden))
    stub("src.rag.confidence", _grounding=Mock(side_effect=forbidden),
         score_confidence=Mock(side_effect=forbidden),
         valid_component=lambda value: isinstance(value, (int, float))
         and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1)
    stub("src.rag.query_rewrite", standalone_query=lambda question, history: question,
         rewrite=Mock(side_effect=forbidden))
    stub("src.rag.retriever", retrieve=Mock(side_effect=forbidden))
    stub("src.rag.router", RoutingDecision=Decision, escalate=lambda tier: "mid",
         model_for=lambda tier: f"{tier}-model", route=Mock(side_effect=forbidden))
    stub("src.rag.text_chain", answer_text=Mock(side_effect=forbidden))
    stub("src.rag.vision_chain", answer_vision=Mock(side_effect=forbidden))
    name = "_isolated_quality_pipeline"
    path = Path(__file__).resolve().parents[1] / "src" / "rag" / "pipeline.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def evidence(text="Cited revenue is 42 million.", **metadata):
    return Evidence(text, {"filename": "report.pdf", "page": 2,
                           "modality": "text", **metadata})


def ready(policy, docs, answer="Revenue is 42 million [E2]."):
    policy.retrieve = Mock(return_value=docs)
    policy.compact = Mock(side_effect=lambda question, items: items)
    policy.rewrite = Mock(side_effect=lambda question, history: question)
    policy.route = Mock(return_value=Decision("cheap", "cheap-model", {}, "test"))
    policy.answer_text = Mock(return_value=answer)
    policy._grounding = Mock(return_value=0.9)
    # Represents disabled scoring, independently of the mandatory grounding gate.
    policy.score_confidence = Mock(return_value=None)


def response(policy, stream=False, **kwargs):
    if not stream:
        return policy.ask("Revenue?", namespace="owned", **kwargs).to_dict()
    events = list(policy.ask_stream("Revenue?", namespace="owned", **kwargs))
    assert events[-1]["type"] == "done"
    done = {key: value for key, value in events[-1].items() if key != "type"}
    assert "".join(event["text"] for event in events if event["type"] == "token") == done["answer"]
    return done


@pytest.mark.parametrize("stream", [False, True])
def test_verification_uses_only_cited_subset_without_history(policy, stream):
    docs = [evidence("UNCITED_ONE", filename="irrelevant-one.pdf"), evidence(),
            evidence("UNCITED_THREE", filename="irrelevant-three.pdf"),
            evidence("Cited currency is USD.", filename="currency.pdf")]
    ready(policy, docs, "Revenue is 42 million USD [E4] [E2].")
    history = [{"role": "user", "content": "HISTORY_USER"},
               {"role": "assistant", "content": "HISTORY_ASSISTANT [E1]"}]
    result = response(policy, stream, history=history)
    generated = policy.answer_text.call_args.kwargs["context"]
    checked, candidate = policy._grounding.call_args.args
    scored = policy.score_confidence.call_args.kwargs
    assert result["answerable"] and candidate == result["answer"]
    assert "HISTORY_USER" in generated and "UNCITED_ONE" in generated
    for excluded in ("UNCITED_ONE", "UNCITED_THREE", "irrelevant-one.pdf",
                     "irrelevant-three.pdf", "HISTORY_USER", "HISTORY_ASSISTANT", "[E1]", "[E3]"):
        assert excluded not in checked
    assert "[E2]" in checked and "[E4]" in checked
    assert checked.index("[E2]") < checked.index("[E4]")
    assert docs[1].page_content in checked and docs[3].page_content in checked
    assert scored["context"] == checked and scored["grounding"] == 0.9
    assert [source["label"] for source in result["sources"] if source["cited"]] == ["E2", "E4"]
    policy._grounding.assert_called_once()


def test_uncited_or_history_support_cannot_rescue_candidate(policy):
    ready(policy, [evidence("SUPPORT_ONLY_ELSEWHERE"), evidence("No revenue given.")])
    policy._grounding.side_effect = lambda context, answer: 0.9 if "SUPPORT_ONLY_ELSEWHERE" in context else 0.1
    result = response(policy, history=[{"role": "assistant", "content": "SUPPORT_ONLY_ELSEWHERE"}])
    assert result["abstention_reason"] == "unsupported_answer"
    assert policy.answer_text.call_count == policy._grounding.call_count == 2
    policy.score_confidence.assert_not_called()


def test_retry_rebuilds_cited_set_and_streams_only_final_answer(policy):
    ready(policy, [evidence("FIRST_ONLY"), evidence("SECOND_ONLY")])
    policy.answer_text.side_effect = ["Rejected [E1].", "Accepted [E2]."]
    policy._grounding.side_effect = [0.1, 0.9]
    events = list(policy.ask_stream("Revenue?", namespace="owned"))
    assert events[-1]["answer"] == "Accepted [E2]."
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Accepted [E2]."
    first, second = [call.args[0] for call in policy._grounding.call_args_list]
    assert "FIRST_ONLY" in first and "SECOND_ONLY" not in first
    assert "SECOND_ONLY" in second and "FIRST_ONLY" not in second
    assert [source["cited"] for source in events[-1]["sources"]] == [False, True]
    assert policy.answer_text.call_count == policy._grounding.call_count == 2
    policy.score_confidence.assert_called_once()


@pytest.mark.parametrize("grounding,reason,calls", [(0.9, None, 1), (0.1, "unsupported_answer", 2),
                                                   (None, "grounding_unavailable", 1)])
@pytest.mark.parametrize("stream", [False, True])
def test_confidence_off_keeps_bounded_grounding(policy, grounding, reason, calls, stream):
    ready(policy, [evidence()], "One [E1]. Two [E1]. Three [E1].")
    policy._grounding.return_value = grounding
    result = response(policy, stream)
    assert policy.settings.confidence_mode == "off"
    assert result["abstention_reason"] == reason
    assert result["answerable"] == (reason is None)
    assert result["confidence"] is None
    assert policy.answer_text.call_count == policy._grounding.call_count == calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("docs", [[], [evidence("   ")], [evidence("Could not summarize image")]])
def test_empty_evidence_does_no_downstream_work(policy, stream, docs):
    policy.retrieve = Mock(return_value=docs)
    result = response(policy, stream, history=[{"role": "user", "content": "Prior topic"}])
    assert result["abstention_reason"] == "no_evidence"
    for name in ("rewrite", "compact", "route", "answer_text", "answer_vision", "_grounding", "score_confidence"):
        getattr(policy, name).assert_not_called()


def test_empty_compaction_does_not_route_or_generate(policy):
    policy.retrieve = Mock(return_value=[evidence()])
    policy.compact = Mock(return_value=[])
    assert response(policy)["abstention_reason"] == "no_evidence"
    for name in ("route", "answer_text", "answer_vision", "_grounding", "score_confidence"):
        getattr(policy, name).assert_not_called()


@pytest.mark.parametrize("answer", ["Missing citation.", "Bad [E99].", "Mixed [E1] [E99]."])
def test_invalid_citations_skip_verifier_and_stop_after_two_candidates(policy, answer):
    ready(policy, [evidence()], answer)
    result = response(policy, stream=True)
    assert result["abstention_reason"] == "invalid_or_missing_citations"
    assert policy.answer_text.call_count == 2
    policy._grounding.assert_not_called()
    policy.score_confidence.assert_not_called()


def test_sync_and_sse_share_result_and_cited_context(policy):
    ready(policy, [evidence("IRRELEVANT"), evidence()])
    sync = response(policy)
    streamed = response(policy, stream=True)
    sync.pop("latency_ms")
    streamed.pop("latency_ms")
    assert sync == streamed
    assert policy._grounding.call_args_list[0] == policy._grounding.call_args_list[1]


@pytest.mark.parametrize("vision", [False, True])
def test_visual_references_preserve_original_order_and_text_only_limits(policy, tmp_path, vision):
    paths = [tmp_path / f"image-{i}.png" for i in range(4)]
    for path in paths:
        path.write_bytes(b"placeholder; never decoded")
    docs = [evidence(f"SUMMARY_{i}", modality="visual", image_path=str(path))
            for i, path in enumerate(paths)]
    # Duplicate placement shares image 2, not a newly numbered attachment.
    docs.append(evidence("DUPLICATE_PLACEMENT", modality="visual", image_path=str(paths[1])))
    ready(policy, docs, "Visual summary [E2] [E4] [E5].")
    if vision:
        policy.route.return_value = Decision("vision", "vision-model", {}, "test")
        policy.answer_vision = Mock(return_value="Visual summary [E2] [E4] [E5].")
    result = response(policy, stream=True)
    checked = policy._grounding.call_args.args[0]
    assert result["answerable"]
    assert "SUMMARY_0" not in checked and "SUMMARY_2" not in checked
    assert "[E1]" not in checked and "[E3]" not in checked
    assert all(label in checked for label in ("[E2]", "[E4]", "[E5]"))
    assert "No image pixels are supplied" in checked
    assert "not exact per-claim entailment" in checked
    assert "Generation image 1" not in checked and "Generation image 4" not in checked
    assert checked.count("No image offered to generation") == (1 if vision else 3)
    assert checked.count("Generation image 2 (reference only)") == (2 if vision else 0)
    assert policy.score_confidence.call_args.kwargs["context"] == checked
    if vision:
        assert policy.answer_vision.call_args.kwargs["image_paths"] == [str(path) for path in paths]
        policy.answer_text.assert_not_called()


@pytest.mark.parametrize("bbox", [None, [], [0, 0, 1], [0, 0, 1, 2, 3], "0,0,1,2",
                                  [False, 0, 1, 2], [0, True, 2, 3], [0, 0, True, 2],
                                  [0, 0, 2, True], [0, 0, 0, 2], [0, 1, 2, 1],
                                  [-1, 0, 2, 3], [0, -1, 2, 3], [2, 0, 1, 3],
                                  [0, 2, 3, 1], [0, 0, float("inf"), 2],
                                  [float("nan"), 0, 2, 3], [0, 0, "2", 3]])
@pytest.mark.parametrize("modality", ["table", "visual"])
def test_invalid_bbox_is_not_exposed(policy, bbox, modality):
    doc = evidence(modality=modality, bbox=bbox)
    assert policy._bbox(bbox) is None
    assert policy._sources([doc])[0]["bbox"] is None


@pytest.mark.parametrize("modality", ["table", "visual"])
@pytest.mark.parametrize("bbox", [[0, 0, 2, 3], (0.25, 1.5, 2.75, 4.0)])
def test_positive_area_bbox_is_preserved_without_mutating_metadata(policy, modality, bbox):
    doc = evidence(modality=modality, bbox=bbox)
    source = policy._sources([doc])[0]
    assert source["bbox"] == list(bbox) and source["bbox"] is not bbox
    assert doc.metadata["bbox"] is bbox


@pytest.mark.parametrize("modality", ["text", "ocr", "image", None, "unknown"])
def test_non_table_visual_metadata_is_page_level_only(policy, modality):
    doc = evidence(modality=modality, bbox=[0, 0, 2, 3], evidence_id="original-id")
    source = policy._sources([doc])[0]
    assert source["bbox"] is None
    assert source["page"] == 2 and source["evidence_id"] == "original-id"
    assert source["quote"] == doc.page_content
    assert doc.metadata["bbox"] == [0, 0, 2, 3]