"""Evidence-first JSON and bounded SSE integration (one server process only).

SSE forwards finalized answer chunks, not live/unverified provider deltas. A
disconnect stops iteration at the next pipeline yield. A running provider call
cannot be forcibly interrupted; its slot remains occupied until it returns.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from config import settings
from src.api.conversations import add_message, assert_ready_document, load_history
from src.api.deps import session_ctx
from src.api.schemas import AskRequest, AskResponse, SourceItem
from src.core import usage
from src.core.audit import log as audit_log
from src.core.db import close_connection, dumps, now, tx
from src.core.exceptions import ValidationError
from src.core.logger import get_logger
from src.core.rate_limit import limiter
from src.rag.pipeline import ask as pipeline_ask, ask_stream
from src.clients.groq_client import get_groq

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ask"])
_STREAM_SLOTS = threading.BoundedSemaphore(4)
_STREAM_QUEUE_SIZE = 8
_SAFE_ERROR = "Unable to complete the answer. Please retry."


def _prepare(req: AskRequest, ctx: dict) -> tuple[str, list[str], list[dict], str]:
    sid = ctx["session_id"]
    with tx() as conn:
        cid, bound = req.conversation_id, None
        if cid is not None:
            row = conn.execute("SELECT document_id FROM conversations WHERE id=? AND session_id=?",
                               (cid, sid)).fetchone()
            if not row:
                raise HTTPException(404, "Conversation not found")
            bound = row["document_id"]
            if bound and req.document_id is not None and req.document_id != bound:
                raise HTTPException(404, "Conversation scope not found")
            if bound and req.document_ids and set(req.document_ids) != {bound}:
                raise HTTPException(404, "Conversation scope not found")
        document_id = req.document_id if req.document_id is not None else bound
        if document_id is not None:
            assert_ready_document(document_id, sid)
        if req.document_ids is not None:
            eligible = list(dict.fromkeys(req.document_ids))
            for did in eligible:
                assert_ready_document(did, sid)
            if document_id is not None and eligible and set(eligible) != {document_id}:
                raise HTTPException(404, "Document scope not found")
        elif document_id is not None:
            eligible = [document_id]
        else:
            rows = conn.execute(
                "SELECT id,filename FROM documents WHERE session_id=? AND status='ready' "
                "ORDER BY filename,version DESC,created_at DESC,id", (sid,),
            ).fetchall()
            seen: set[str] = set()
            eligible = []
            for row in rows:
                if req.version_mode == "all" or row["filename"] not in seen:
                    eligible.append(row["id"])
                    seen.add(row["filename"])
        reservation = usage.reserve_ask(sid)
        if cid is None:
            cid = uuid.uuid4().hex[:16]
            selected = document_id or (eligible[0] if req.document_ids and len(eligible) == 1 else None)
            conn.execute("INSERT INTO conversations(id,session_id,document_id,title,created_at,updated_at) "
                         "VALUES(?,?,?,?,?,?)", (cid, sid, selected, None, now(), now()))
        history = load_history(cid, settings.conversation_memory_turns)
    return cid, eligible, history, reservation


def _public_payload(payload: dict) -> dict:
    data = {k: v for k, v in payload.items() if k not in {"type", "image_urls"}}
    data["image_paths"] = []
    sources = []
    for source in data.get("sources", []):
        item = SourceItem(**source).model_dump()
        if item["source"]:
            item["source"] = str(item["source"]).replace("\\", "/").rsplit("/", 1)[-1]
        sources.append(item)
    data["sources"] = sources
    return data


def _save_answer(payload: dict, req: AskRequest, ctx: dict, cid: str,
                 eligible: list[str], reservation: str,
                 prompt_tokens: int = 0, completion_tokens: int = 0) -> dict:
    
    result = AskResponse(**_public_payload(payload)).model_dump()
    result["conversation_id"] = cid
    
    with tx() as conn:
        row = conn.execute("SELECT document_id, title FROM conversations WHERE id=? AND session_id=?",
                           (cid, ctx["session_id"])).fetchone()
        if not row:
            raise HTTPException(404, "Conversation not found")
        
        new_title = _generate_title(req.question)
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (new_title, cid))

        for did in eligible:
            assert_ready_document(did, ctx["session_id"])
        add_message(cid, "user", req.question)
        mid = add_message(cid, "assistant", result["answer"])
        result["message_id"] = mid
        meta = {**result, "scope": {"document_id": req.document_id,
                "document_ids": eligible, "version_mode": req.version_mode}}
        conn.execute("UPDATE messages SET meta_json=? WHERE id=?", (dumps(meta), mid))
        usage.finish_ask(reservation, True)
        audit_log("ask", session_id=ctx["session_id"], resource=cid,
                  meta={"tier": result["tier"], "document_ids": eligible})
                  
    return result
from src.clients.groq_client import chat

def _generate_title(question: str) -> str:
    try:
        # Use a fast, natively supported Groq model
        cheap_model = getattr(settings, "cheap_model", "qwen/qwen3.8-27b")
        prompt = f"Summarize this user question into a very short, concise title of 3 to 5 words maximum. Do not use quotation marks or punctuation: {question}"
        title = chat(
            model=cheap_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=15,
            temperature=0.3
        )
        return title.strip().strip('"')[:40] or "Untitled conversation"
    except Exception:
        return "Untitled conversation"

@router.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_ask)
def ask_endpoint(request: Request, req: AskRequest, ctx=Depends(session_ctx)) -> AskResponse:
    cid, eligible, history, reservation = _prepare(req, ctx)
    try:
        result = pipeline_ask(req.question, namespace=ctx["namespace"],
                              document_ids=eligible, history=history)
        # Flush standard endpoint tokens
        usage.flush_tokens(ctx["session_id"], document_id=req.document_id)
        return AskResponse(**_save_answer(result.to_dict(), req, ctx, cid, eligible, reservation))
    except HTTPException:
        raise
    except usage.BudgetExceeded:
        raise
    except ValidationError as exc:
        raise HTTPException(400, "Invalid question") from exc
    except Exception as exc:
        log.exception("ask_failed")
        raise HTTPException(502, _SAFE_ERROR) from exc
    finally:
        usage.finish_ask(reservation, False)


class _StreamWork:
    def __init__(self, req, ctx, prepared):
        self.req, self.ctx = req, ctx
        self.cid, self.eligible, self.history, self.reservation = prepared
        self.events: queue.Queue = queue.Queue(maxsize=_STREAM_QUEUE_SIZE)
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.started = False
        self.released = False
        self.thread: threading.Thread | None = None

    def _release(self):
        with self.lock:
            if not self.released:
                self.released = True
                _STREAM_SLOTS.release()

    def _discard_events(self):
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                break

    def cancel(self):
        self.cancelled.set()
        self._discard_events()
        with self.lock:
            not_started = not self.started
            if not_started:
                self.started = True
        if not_started:
            try:
                usage.finish_ask(self.reservation, False)
            finally:
                self._release()

    def start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
        self.thread = threading.Thread(target=self._run, name="vrag-answer", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            try:
                usage.finish_ask(self.reservation, False)
            finally:
                self._release()
            raise

    def _put(self, event) -> bool:
        while not self.cancelled.is_set():
            try:
                self.events.put(event, timeout=.05)
                if self.cancelled.is_set():
                    self._discard_events()
                    return False
                return True
            except queue.Full:
                pass
        return False

    def _run(self):
        iterator = None
        try:
            if self.cancelled.is_set():
                return
            iterator = iter(ask_stream(self.req.question, namespace=self.ctx["namespace"],
                                       document_ids=self.eligible, history=self.history))
            while not self.cancelled.is_set():
                try:
                    event = next(iterator)
                except StopIteration:
                    self._put({"type": "error", "error": _SAFE_ERROR})
                    break
                if self.cancelled.is_set():
                    break
                kind = event.get("type")
                if kind == "error":
                    self._put({"type": "error", "error": _SAFE_ERROR})
                    break
                if kind == "done":
                    with self.lock:
                        if self.cancelled.is_set():
                            break
                        result = _save_answer(event, self.req, self.ctx, self.cid,
                                              self.eligible, self.reservation)
                        
                        # --- Flush tokens and capture them ---
                        p_toks, c_toks = usage.flush_tokens(self.ctx["session_id"], document_id=self.req.document_id)
                        
                        # --- Inject into result so the UI displays them ---
                        result["prompt_tokens"] = p_toks
                        result["completion_tokens"] = c_toks
                        
                    self._put({"type": "done", **result})
                    break
                if kind == "meta":
                    event = {"type": "meta", **_public_payload(event)}
                if kind in {"meta", "stage", "token"} and not self._put(event):
                    break
        except Exception:
            log.exception("ask_stream_failed")
            self._put({"type": "error", "error": _SAFE_ERROR})
        finally:
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
            finally:
                try:
                    usage.finish_ask(self.reservation, False)
                finally:
                    close_connection()
                    try:
                        self._put(None)
                    finally:
                        if self.cancelled.is_set():
                            self._discard_events()
                        self._release()

class _AnswerStreamResponse(EventSourceResponse):
    def __init__(self, content, work: _StreamWork):
        super().__init__(content, send_timeout=30, headers={"Cache-Control": "private, no-store"})
        self.work = work

    async def __call__(self, scope, receive, send):
        async def receive_with_cancel():
            message = await receive()
            if message["type"] == "http.disconnect":
                self.work.cancel()
            return message

        try:
            await super().__call__(scope, receive_with_cancel, send)
        finally:
            self.work.cancel()

@router.post("/ask/stream")
@limiter.limit(settings.rate_ask)
async def ask_stream_endpoint(request: Request, req: AskRequest, ctx=Depends(session_ctx)):
    if not _STREAM_SLOTS.acquire(blocking=False):
        raise HTTPException(503, "Answer stream capacity reached. Retry later.", headers={"Retry-After": "5"})
    try:
        work = _StreamWork(req, ctx, _prepare(req, ctx))
    except BaseException:
        _STREAM_SLOTS.release()
        raise

    async def gen():
        try:
            if await request.is_disconnected():
                return
            yield {"event": "start", "data": json.dumps({"conversation_id": work.cid})}
            work.start()
            while not work.cancelled.is_set():
                if await request.is_disconnected():
                    return
                try:
                    event = work.events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(.02)
                    continue
                if event is None:
                    return
                kind = event["type"]
                yield {"event": kind, "data": json.dumps({k: v for k, v in event.items() if k != "type"})}
                if kind in {"done", "error"}:
                    return
        finally:
            work.cancel()

    try:
        return _AnswerStreamResponse(gen(), work)
    except BaseException:
        work.cancel()
        raise