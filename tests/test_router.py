from langchain_core.documents import Document

from src.rag import router as router_mod


def test_route_vision_when_visual_present(monkeypatch):
    monkeypatch.setattr(router_mod, "_classify", lambda q: {"needs_vision": False, "complexity": "simple"})
    docs = [Document(page_content="x", metadata={"modality": "visual", "image_path": "/tmp/x.png"})]
    d = router_mod.route("q", docs)
    assert d.tier == "vision"


def test_route_frontier_for_complex(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify",
        lambda q: {"needs_vision": False, "complexity": "complex", "multi_hop": True},
    )
    d = router_mod.route("q", [])
    assert d.tier == "frontier"


def test_route_cheap_for_simple(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify",
        lambda q: {"needs_vision": False, "complexity": "simple", "multi_hop": False},
    )
    d = router_mod.route("q", [])
    assert d.tier == "cheap"


def test_escalate_ladder():
    assert router_mod.escalate("cheap") == "mid"
    assert router_mod.escalate("mid") == "frontier"
    assert router_mod.escalate("frontier") == "frontier"
