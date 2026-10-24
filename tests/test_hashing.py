from src.utils.hashing import sha256_bytes, sha256_text


def test_sha256_deterministic():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")


def test_sha256_bytes():
    assert len(sha256_bytes(b"abc")) == 64
