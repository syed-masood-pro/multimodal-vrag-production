"""Feedback endpoint (thumbs up/down + reason)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import session_ctx
from src.api.schemas import FeedbackRequest
from src.core.audit import log as audit_log
from src.core.db import now, tx

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("")
def submit_feedback(req: FeedbackRequest, ctx=Depends(session_ctx)) -> dict[str, str]:
    with tx() as conn:
        message = conn.execute(
            "SELECT m.conversation_id FROM messages m JOIN conversations c ON c.id=m.conversation_id "
            "WHERE m.id=? AND m.role='assistant' AND c.session_id=?",
            (req.message_id, ctx["session_id"]),
        ).fetchone()
        if not message:
            raise HTTPException(404, "Assistant message not found")
        # BEGIN IMMEDIATE serializes writers, including dedup of legacy repeated votes.
        rows = conn.execute("SELECT id FROM feedback WHERE session_id=? AND message_id=? ORDER BY id",
                            (ctx["session_id"], req.message_id)).fetchall()
        if rows:
            fid = rows[0]["id"]
            conn.execute("DELETE FROM feedback WHERE session_id=? AND message_id=? AND id!=?",
                         (ctx["session_id"], req.message_id, fid))
            conn.execute("UPDATE feedback SET conversation_id=?,vote=?,reason=?,created_at=? WHERE id=?",
                         (message["conversation_id"], req.vote, req.reason, now(), fid))
        else:
            conn.execute(
                "INSERT INTO feedback(session_id,conversation_id,message_id,vote,reason,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (ctx["session_id"], message["conversation_id"], req.message_id, req.vote, req.reason, now()),
            )
        audit_log("feedback.submit", session_id=ctx["session_id"], resource=str(req.message_id))
    return {"status": "ok"}
