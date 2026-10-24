"""SQLite persistence layer for users, docs, conversations, jobs, audit, feedback."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from config import settings

_LOCK = threading.RLock()
_LOCAL = threading.local()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(settings.sqlite_path), timeout=30, isolation_level=None
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def get_conn() -> sqlite3.Connection:
    path = str(settings.sqlite_path)
    if getattr(_LOCAL, "path", None) != path or getattr(_LOCAL, "conn", None) is None:
        close_connection()
        with _LOCK:
            conn = _connect()
            try:
                _init_schema(conn)
            except BaseException:
                conn.close()
                raise
            _LOCAL.conn, _LOCAL.path = conn, path
    return _LOCAL.conn


def close_connection() -> None:
    """Close only the calling thread's connection."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is not None:
        conn.close()
    _LOCAL.conn = None


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    nested = conn.in_transaction
    savepoint = f"sp_{getattr(_LOCAL, 'depth', 0)}"
    _LOCAL.depth = getattr(_LOCAL, "depth", 0) + 1
    conn.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute(f"RELEASE SAVEPOINT {savepoint}" if nested else "COMMIT")
    except BaseException:
        if nested:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        else:
            conn.execute("ROLLBACK")
        raise
    finally:
        _LOCAL.depth -= 1


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            last_seen  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            version INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            storage_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            pages INTEGER,
            chunks INTEGER,
            created_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            document_id TEXT,
            title TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,           -- user | assistant
            content TEXT NOT NULL,
            meta_json TEXT,               -- tier/confidence/sources/etc
            created_at REAL NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,           -- ingest
            status TEXT NOT NULL,         -- queued | running | done | error
            progress INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL DEFAULT 0,
            stage TEXT,
            message TEXT,
            result_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            action TEXT NOT NULL,
            resource TEXT,
            ip TEXT,
            request_id TEXT,
            meta_json TEXT,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            conversation_id TEXT,
            message_id INTEGER,
            vote INTEGER NOT NULL,        -- 1 up, -1 down
            reason TEXT,
            created_at REAL NOT NULL
        );
        """
    )

    # Additive migrations preserve existing installations and observable jobs.
    conn.execute("BEGIN IMMEDIATE")
    try:
        for table, columns in {
            "documents": {"attempt_count": "INTEGER NOT NULL DEFAULT 0", "last_error": "TEXT",
                          "updated_at": "REAL", "job_id": "TEXT"},
            "jobs": {"document_id": "TEXT", "cancel_requested": "INTEGER NOT NULL DEFAULT 0"},
        }.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
        conn.execute("UPDATE documents SET updated_at=created_at WHERE updated_at IS NULL")
        conn.execute("""CREATE TABLE IF NOT EXISTS document_chunks (
            namespace TEXT NOT NULL, document_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
            content TEXT NOT NULL, metadata_json TEXT NOT NULL,
            PRIMARY KEY(namespace, document_id, chunk_id))""")
        conn.execute("""CREATE TRIGGER IF NOT EXISTS delete_document_chunks
            AFTER DELETE ON documents BEGIN
            DELETE FROM document_chunks WHERE document_id=OLD.id; END""")
        conn.execute("""CREATE TABLE IF NOT EXISTS usage_reservations (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT NOT NULL,
            day TEXT NOT NULL, amount INTEGER NOT NULL, state TEXT NOT NULL,
            document_id TEXT, success INTEGER, created_at REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_reservations(session_id, kind, day, state)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id, created_at DESC)")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def namespace_owner(namespace: str) -> str | None:
    """Reserve s_ namespaces for browser sessions. Ambiguous prefixes fail closed."""
    if not namespace.startswith("s_"):
        return None  # Intentional legacy CLI namespace, no document registry required.
    rows = get_conn().execute(
        "SELECT id FROM sessions WHERE substr(id,1,20)=?", (namespace[2:],)
    ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else ""


def allowed_document_ids(namespace: str, document_ids: list[str] | None = None) -> list[str] | None:
    owner = namespace_owner(namespace)
    if owner is None:
        return document_ids
    rows = get_conn().execute(
        "SELECT id FROM documents WHERE session_id=? AND status='ready' ORDER BY id", (owner,)
    ).fetchall()
    allowed = [r["id"] for r in rows]
    allowed_set = set(allowed)
    return allowed if document_ids is None else [d for d in document_ids if d in allowed_set]


# ---------------- helpers ----------------
def now() -> float:
    return time.time()


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def loads(s: str | None) -> Any:
    return json.loads(s) if s else None
