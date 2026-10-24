"""Conversation CRUD."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import session_ctx
from src.api.schemas import ConversationItem, ConversationRequest, MessageItem
from src.core.audit import log as audit_log
from src.core.db import dumps, get_conn, loads, now, tx

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationItem)
def create_conversation(
    payload: ConversationRequest | None = None, ctx=Depends(session_ctx)
) -> ConversationItem:
    payload = payload or ConversationRequest()
    cid = uuid.uuid4().hex[:16]
    doc_id = payload.document_id
    title = payload.title
    t = now()
    with tx() as conn:
        if doc_id is not None:
            assert_ready_document(doc_id, ctx["session_id"])
        conn.execute(
            "INSERT INTO conversations(id, session_id, document_id, title, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (cid, ctx["session_id"], doc_id, title, t, t),
        )
    return ConversationItem(id=cid, title=title, document_id=doc_id, created_at=t, updated_at=t)


@router.get("", response_model=list[ConversationItem])
def list_conversations(ctx=Depends(session_ctx)) -> list[ConversationItem]:
    rows = get_conn().execute(
        "SELECT id, title, document_id, created_at, updated_at FROM conversations "
        "WHERE session_id=? ORDER BY updated_at DESC",
        (ctx["session_id"],),
    ).fetchall()
    return [ConversationItem(**dict(r)) for r in rows]


@router.get("/{cid}/messages", response_model=list[MessageItem])
def list_messages(cid: str, ctx=Depends(session_ctx)) -> list[MessageItem]:
    _assert_owned(cid, ctx["session_id"])
    rows = get_conn().execute(
        "SELECT id, role, content, meta_json, created_at FROM messages "
        "WHERE conversation_id=? ORDER BY id ASC",
        (cid,),
    ).fetchall()
    return [
        MessageItem(
            id=r["id"], role=r["role"], content=r["content"],
            meta=_public_meta(loads(r["meta_json"])), created_at=r["created_at"],
        )
        for r in rows
    ]


def _public_meta(meta: dict | None) -> dict | None:
    if meta is None:
        return None
    # Older saved chats may contain public image URLs or private local paths.
    result = {k: v for k, v in meta.items() if k not in {"image_urls", "image_paths"}}
    if "image_paths" in meta:
        result["image_paths"] = []
    if "sources" in result:
        sources = []
        for source in result["sources"]:
            item = {k: v for k, v in source.items() if k not in {"image_path", "image_url"}}
            if item.get("source"):
                item["source"] = str(item["source"]).replace("\\", "/").rsplit("/", 1)[-1]
            sources.append(item)
        result["sources"] = sources
    return result


@router.delete("/{cid}")
def delete_conversation(cid: str, ctx=Depends(session_ctx)) -> dict[str, str]:
    with tx() as conn:
        _assert_owned(cid, ctx["session_id"])
        conn.execute("DELETE FROM feedback WHERE conversation_id=? AND session_id=?", (cid, ctx["session_id"]))
        conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
        audit_log("conversation.delete", session_id=ctx["session_id"], resource=cid)
    return {"status": "deleted"}


def assert_ready_document(document_id: str, session_id: str) -> None:
    if not get_conn().execute(
        "SELECT 1 FROM documents WHERE id=? AND session_id=? AND status='ready'",
        (document_id, session_id),
    ).fetchone():
        raise HTTPException(404, "Ready document not found")


def _assert_owned(cid: str, session_id: str) -> None:
    row = get_conn().execute(
        "SELECT id FROM conversations WHERE id=? AND session_id=?", (cid, session_id)
    ).fetchone()
    if not row:
        raise HTTPException(404, "Conversation not found")


def add_message(cid: str, role: str, content: str, meta: dict | None = None) -> int:
    t = now()
    with tx() as conn:
        cur = conn.execute(
            "INSERT INTO messages(conversation_id, role, content, meta_json, created_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (cid, role, content, dumps(meta) if meta is not None else None, t),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (t, cid))
        return int(cur.lastrowid)


def load_history(cid: str, turns: int) -> list[dict[str, str]]:
    rows = get_conn().execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
        (cid, 2 * max(0, turns)),
    ).fetchall()
    return list(reversed([{"role": r["role"], "content": r["content"]} for r in rows]))
