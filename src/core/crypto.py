"""Fernet-based encryption at rest for uploaded PDFs."""
from __future__ import annotations

import base64
from pathlib import Path

from cryptography.fernet import Fernet

from config import settings

_KEY_FILE = Path(settings.cache_dir) / ".fernet.key"


def _load_key() -> bytes:
    if settings.encryption_key:
        return settings.encryption_key.encode()
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes()
    k = Fernet.generate_key()
    _KEY_FILE.write_bytes(k)
    try:
        # Best-effort permissions on POSIX; on Windows this is a no-op.
        _KEY_FILE.chmod(0o600)
    except Exception:
        pass
    return k


_fernet = Fernet(_load_key())


def encrypt_bytes(data: bytes) -> bytes:
    return _fernet.encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    return _fernet.decrypt(data)


def encrypt_file(src_bytes: bytes, dest: Path) -> int:
    enc = encrypt_bytes(src_bytes)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(enc)
    return len(enc)


def decrypt_file(src: Path) -> bytes:
    return decrypt_bytes(src.read_bytes())


def key_fingerprint() -> str:
    """Short public fingerprint for the current key (no secret revealed)."""
    import hashlib
    return hashlib.sha256(_load_key()).hexdigest()[:12]


__all__ = ["encrypt_bytes", "decrypt_bytes", "encrypt_file", "decrypt_file", "key_fingerprint"]

# Silence lint on unused base64 (kept for potential external key handling)
_ = base64
