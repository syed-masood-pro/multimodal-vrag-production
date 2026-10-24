from langchain_core.documents import Document

from src.rag.compaction import count_tokens, enforce_budget


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0


def test_enforce_budget_truncates():
    big = Document(page_content="word " * 5000, metadata={"page": 1, "modality": "text"})
    kept = enforce_budget([big], budget=100)
    assert count_tokens(kept[0].page_content) <= 100
