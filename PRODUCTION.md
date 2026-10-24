# PRODUCTION.md — Problem → Solution catalog

Version 2. Each entry states a concrete problem, the solution as
implemented, the source of truth, and the residual limits. Only claims
that are verifiable against the code in this repository are included.

---

## Group A — Answer quality

### A1. The model can produce fluent text that is not backed by the evidence

**Problem.** Retrieval-augmented models happily hallucinate when the
retrieved context is weak or when the model ignores it.

**Solution.**

1. Evidence is compacted into a labelled pack `E1, E2, …` before
   generation, and the system prompt requires the model to cite the
   labels it uses.
2. `_cited_context()` in [src/rag/pipeline.py](src/rag/pipeline.py) runs
   sentence-level citation validation: every non-boilerplate sentence
   must carry at least one label that exists in the pack.
3. A grounding gate then asks a small model whether the answer is
   supported by the *cited* evidence text. Unsupported answers do not
   reach the client — they trigger a retry on a stronger tier and, if
   that also fails, an abstention.
4. Two attempts, escalating tier on the second.

**Residual limits.** The grounding gate is itself an LLM and can be
wrong; on a Groq 5xx it currently fails closed and abstains without a
retry. Vision grounding is weaker than text grounding.

### A5. Combined citations `[E1, E2]` were rejected as malformed

**Problem.** The citation gate accepted only singleton `[Ei]` tokens,
so a natural LLM output like `"...as shown in [E1, E2]."` was flagged
`Invalid citation [E1, E2]` and forced abstention on otherwise
well-grounded answers.

**Solution.** `_citation_check()` in
[src/rag/pipeline.py](src/rag/pipeline.py) now extracts every bracket
group, splits on `,`, `;`, `/`, `|`, or whitespace, and validates each
token individually with `re.fullmatch(r"[Ee]\d+")`. Non-citation
brackets (e.g. `[note]`) are ignored rather than warned on. Invalid
labels still produce warnings; the accepted-set is the union of every
valid token.

**Technique.** Bracket-scan → per-token normalisation
(`.upper()`) → allow-list intersection.

### A2. Callers cannot tell whether an answer was verified or invented

**Solution.** The response schema
([src/api/schemas.py](src/api/schemas.py)) always includes
`answerable`, `abstained`, `abstention_reason`, `status`,
`citation_warnings`, and per-source `cited: bool`. Clients render an
explicit “Answered · verified” chip only when `status='answered'`.

### A3. Abstention was implicit and inconsistent

**Solution.** A single constant `ABSTENTION` in
[src/rag/pipeline.py](src/rag/pipeline.py) is used everywhere, and one
of seven reason codes is returned:
`no_evidence`, `insufficient_evidence`, `invalid_or_missing_citations`,
`grounding_unavailable`, `unsupported_answer`,
`confidence_unavailable`, `low_confidence`.

### A4. Hand-tuned confidence was displayed as if it were truth

**Problem.** The prior confidence combined retrieval / rerank / citation
signals with hand-picked weights, and callers rendered it as a truth
score.

**Solution.** [src/rag/confidence.py](src/rag/confidence.py) now
optionally consumes a logistic calibrator trained on
👍/👎 feedback. The artifact is validated (`schema_version=2`,
`target="user_preference"`, matching feature keys, finite weights);
mismatches are rejected at load time. The response carries
`confidence_available` and, if false, a
`confidence_unavailable_reason`. Vision paths return `available=false`
with `"Text samples cannot measure image-only support"`. The UI hides
the numeric score whenever `available=false`.

**Residual limits.** This calibrates *preference*, not truth.

---

## Group B — Frontend and integration defects

### B1. SSE parser lost events split across TCP reads

**Solution.** [web/app.js](web/app.js) `createSSEParser()` is an
incremental parser: it buffers a trailing `\r`, only emits an event on a
blank line, and preserves multi-line `data:` fields.
`consumeAskStream()` treats a stream without an explicit `done` event as
an error rather than silently returning a partial answer.

### B2. SSE lost the session cookie set on the first request

**Solution.** The session middleware in
[src/api/app.py](src/api/app.py) calls `resolve_session` on the request
and `set_session_cookie` on every response — including streaming
responses.

### B3. Modal dialog was always visible / trapped focus incorrectly

**Solution.** [web/app.js](web/app.js) `showModal()`/`closeModal()`
toggle the `hidden` attribute, mark siblings `inert`, capture the
originating focus target, and restore it on close. A tab-cycle handler
constrains focus to focusable descendants; Escape closes.

### B4. Retries could target a stale conversation

**Solution.** Messages carry a `retry` record with `view` and
`conversation_id`. The retry button refuses to fire when either has
changed.

### B5. UI errors surfaced as `[object Object]`

**Solution.** `errorText()` in [web/app.js](web/app.js) recursively
flattens FastAPI `detail` payloads (dict / list / str) into readable
text.

### B6. Runaway ingestion had no user-visible stop

**Solution.** The document list exposes a Stop button that calls
`POST /api/documents/{id}/cancel`. The runner’s
`checkpoint()` picks this up and raises `JobCancelled`.

---

## Group C — Document lifecycle dashboard

### C1. Failed ingestions were opaque

**Solution.** [src/api/lifecycle.py](src/api/lifecycle.py) exposes
`GET /api/documents/{id}/detail`, returning the document row plus a
paginated `jobs` history: stage, progress, message, per-page statuses,
`failed_page`, `warnings`, and `pages_without_content`. The frontend
renders each attempt as its own card in the lifecycle modal.

### C2. There was no explicit retry or cancel path

**Solution.** `POST /api/documents/{id}/retry` and
`POST /api/documents/{id}/cancel` are first-class endpoints. Retry is
only allowed from `error` or `cancelled`. Cancel sets
`jobs.cancel_requested=1` and the worker observes it at the next
`checkpoint`.

### C3. Jobs disappeared on process restart

**Solution.** `runner.reconcile_interrupted_jobs()` on startup walks
`queued` and `running` jobs, marks them `error` with reason
`interrupted`, and releases pending upload reservations. The user must
explicitly retry — the system never resumes silently on stale state.

---

## Group D — Document comparison and version-aware search

### D1. Users could not tell what changed between versions

**Solution.** `GET /api/documents/compare` in
[src/api/lifecycle.py](src/api/lifecycle.py):

- Requires both documents to be owned by the caller and `ready`.
- Walks the union of pages, paginated (`offset`, `limit`).
- Computes `changed` from a SHA-256 of the full extracted text per page
  (not the truncated preview shown to the user).
- Character-caps the preview per side and sets `text_truncated: true`
  when it does so — the UI states this explicitly.

**Residual limits.** This is an extracted-text diff, not a visual pixel
diff; images, layout, and OCR omissions are not shown.

### D2. Asking a question could silently mix versions

**Solution.** `AskRequest` accepts `document_ids: list[str] | None` and
`version_mode: "latest" | "pinned" | "all"`. The default `latest`
resolves to the maximum version per `(session_id, filename)`. The
selected mode is echoed back to the client.

### D3. Scope could leak across tenants

**Solution.** Every retrieval path resolves
`allowed_document_ids(session_id, requested)` in
[src/core/db.py](src/core/db.py); if the intersection is empty,
retrieval short-circuits and the pipeline abstains with
`no_evidence`. `bm25_store._check_owner()` re-verifies at query time.

---

## Group E — Tenant usage budgets

### E1. A single tenant could exhaust storage, LLM spend, or job slots

**Solution.** [src/core/usage.py](src/core/usage.py) implements atomic
reservations against `usage_reservations`:

- `reserve_upload` / `finish_upload` (documents, storage bytes),
- `reserve_pages` / `finish_pages` (daily pages),
- `reserve_ask` / `finish_ask` (daily asks),
- `snapshot()` for `/api/usage`.

Limits are configured in [config/settings.py](config/settings.py)
(`tenant_max_documents`, `tenant_max_storage_mb`, `tenant_daily_asks`,
`tenant_daily_pages`, `job_max_pending`, `max_pdf_pages`). Overrun
raises `BudgetExceeded(resource)`, mapped in
[src/api/app.py](src/api/app.py) to HTTP 429 with body
`{"detail": {"message": "Browser session quota exceeded.", "resource": ..., "remaining": 0}}`. Possible `resource` values: `documents`,
`storage_bytes`, `daily_asks`, `daily_pages`, `pdf_pages`.

> Note: job-runner **queue** overflow is a different case and returns
> HTTP **503** with `Retry-After: 5` (see
> [src/api/documents.py](src/api/documents.py) and
> [src/api/lifecycle.py](src/api/lifecycle.py)), not 429.

### E2. Users could not see their own quota

**Solution.** `GET /api/usage` returns `usage`, `limits`, and
`remaining`. The sidebar renders it in a `<details>` panel.

**Residual limits.** Atomic within one SQLite database only. Multi-node
deployment requires a shared coordinator.

---

## Group F — Answerability and abstention

### F1. Weak evidence used to produce confident-sounding answers

**Solution.** The pipeline in
[src/rag/pipeline.py](src/rag/pipeline.py) runs an answerability check
before generation and gates on citation + grounding after generation.
Failures escalate once, then abstain with a reason code. The client
distinguishes “Abstained: …” from an answered message.

### F2. Callers could not build their own gates

**Solution.** The response exposes `retrieval_query`,
`citation_warnings`, and `calibration` (when present) so downstream
systems can enforce stricter policies (e.g. refuse `low_confidence` for
regulated flows).

---

## Group G — Evidence-linked answers

### G1. Citations pointed at documents, not passages

**Solution.** Every extracted chunk carries a deterministic
`evidence_id = sha256(document_id ⊕ source ⊕ page ⊕ modality ⊕ locator)`
(see [src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py)).
Citation labels `E1..En` resolve to these IDs.

### G2. Users could not see the page that supported a claim

**Solution.**

- `GET /api/documents/{id}/evidence/{eid}` returns filename, page,
  quote, bbox (may be null), and a page image URL.
- `GET /api/documents/{id}/pages/{page}.png?evidence_id=` renders the
  page, capped at 8 M pixels, and — when the evidence resolves to a
  bbox — highlights that region with `fitz.Page.draw_rect` before
  rasterisation.
- The evidence modal in [web/app.js](web/app.js) states clearly whether
  it is showing region-level or page-level evidence, and validates that
  the image URL is same-origin and under the expected document prefix.

**Residual limits.** Region-level evidence marks the recorded bbox, not
necessarily every quoted word. Table and visual bboxes may be
page-level only.

---

## Group H — Infrastructure hygiene

### H1. Load balancers used liveness as readiness

**Solution.** `/readyz` returns HTTP **503** with `ready=false` and a
`details` map while DB, embeddings, vectorstore, or the runner are not
yet initialised. `/healthz` remains a pure liveness probe.

### H2. Uploads with the same filename could alias each other

**Solution.** Storage paths are derived from a UUID `doc_id`, not from
digest or filename ([src/api/documents.py](src/api/documents.py)).

### H3. Deletes raced against reads

**Solution.** `_DELETE_LOCK` serialises deletion; the row is first
marked `status='deleting'` so retrieval and BM25 lookups filter it out
before Pinecone / storage removal begins.

### H4. Pinecone metadata silently dropped nested fields

**Solution.**
[src/vectorstore/pinecone_store.py](src/vectorstore/pinecone_store.py)
stashes dict/list metadata as `_vrag_metadata_json` and restores it at
read time. Wire payloads are validated for scalar-only Pinecone types.

### H5. Job runner had no admission control

**Solution.** A bounded semaphore (`_slots`, size
`settings.job_max_pending`) gates submissions; overflow raises
`QueueFull`, surfaced to the caller as **HTTP 503** with
`Retry-After: 5` (`🔍` see
[src/api/documents.py](src/api/documents.py) and
[src/api/lifecycle.py](src/api/lifecycle.py)).

### H6. Streaming answers survived client disconnects

**Solution.** `_AnswerStreamResponse` in
[src/api/ask_router.py](src/api/ask_router.py) wraps the ASGI `receive`
to observe `http.disconnect` and cancel the pipeline, releasing the
slot and the quota reservation.

---

## What is deliberately out of scope

- Truth calibration (only preference calibration exists).
- Image OCR PII redaction (visual evidence is flagged
  `image_redacted: False`).
- Multi-node quota enforcement (single SQLite assumption).
- Provider-side safety guarantees (citation + grounding gates are the
  policy boundary).
- Live provider-token streaming (SSE forwards verified text in
  ~160-char chunks).

These are stated so operators plan around them rather than discover
them in production.

## Known limits and known bugs (post-audit)

Discovered by a source-level audit of this repository. Listed so
operators do not rely on behaviour the code does not deliver.

1. **Single-process only.** Tenant quotas
   ([src/core/usage.py](src/core/usage.py)), the delete lock
   ([src/api/documents.py](src/api/documents.py)), and the job-runner
   semaphore ([src/jobs/runner.py](src/jobs/runner.py)) all live in
   in-process state. Do not deploy with `uvicorn --workers >1` or
   multiple gunicorn workers without replacing this state.
2. **Pinecone + SQLite are not a distributed transaction.** Vectors are
   upserted before the SQLite commit that publishes `ready`. If the
   commit fails and vector cleanup also fails, orphan vectors remain in
   Pinecone (documented in
   [src/jobs/ingest_job.py](src/jobs/ingest_job.py) as
   `cleanup_required=True`). Retrieval does not surface them because
   ownership is gated on the SQLite row.
3. **Grounding gate fails closed on Groq 5xx** without a retry.
4. **Cancellation is bounded by the current LLM call.** `checkpoint()`
   fires between stages, so a long single-page vision summary cannot be
   interrupted mid-call.
5. **CORS default is `"*"` with credentials.** Browsers reject the
   combination but the setting is misleading. Set `allowed_origins`
   explicitly for real deployments.
6. **Auth is off by default.** `api_key=""` disables the check and the
   dependency is not wired into any router. Use a reverse proxy or
   wire `src/core/auth.py` before exposing publicly.
7. **No CSRF protection.** Cookie + `SameSite=lax` is the only defence
   on state-changing endpoints.
8. **`GET /api/session` allocates a session row.** Non-idempotent GET
   — fine behind a same-origin SPA, brittle otherwise.
9. **Fernet key has no rotation.** Rotating `encryption_key`
   invalidates every stored PDF.
10. **Encrypted PDFs are not garbage-collected** after failed ingests
    unless the document row is deleted via the API.
11. **Page reservations survive a mid-ingest crash** until the daily
    rollover; only upload reservations are released on restart.
12. **`/metrics` is unauthenticated** (Prometheus). Bind to a private
    port or add auth.
13. **`version_mode="pinned"` is not a valid value.** Only `latest`
    and `all` are accepted; to pin a version, pass its document id in
    `document_ids`.
14. **`allowed_document_ids()`** short-circuits to empty if a caller
    supplies a scope that is not a subset of their owned ready
    documents — by design, but callers must expect an
    `no_evidence` abstention rather than a broader fallback.
15. **Non-streaming `POST /api/ask`** runs the pipeline synchronously
    on the event loop; prefer `/api/ask/stream` for concurrency.
16. **Rate limiting** (SlowAPI) applies only where the decorator is
    present on a route; there is no global `SlowAPIMiddleware`.
    Feedback and lifecycle endpoints are not rate-limited.
17. **Requirements are floor-pinned (`>=`).** Reproducible builds
    require a lockfile; PyMuPDF in particular has drifted APIs across
    minor versions.
18. **Rotated PDFs.** Evidence bbox highlighting assumes canonical
    page orientation; pages with `/Rotate 90` may show highlights that
    drift from the quoted region.

---

## Problems solved so far — techniques and tech used

A rolling ledger of every problem this repository has solved end to
end, with the concrete technique and the open-source components that
implement it. Cross-reference to the Group entries above for the
detailed narrative; this section is the at-a-glance matrix.

### Answering quality

| Problem                               | Technique                                                                                                                        | Tech                                                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Hallucinated fluent prose             | Labelled evidence pack`E1..En` → required cite tokens → grounding-gate LLM → tier escalation, then abstain                  | Groq LLMs (`llama-3.1-8b-instant`, `openai/gpt-oss-20b/120b`), regex citation gate, small-model grounding call        |
| Citations`[E1, E2]` rejected        | Bracket-scan, per-token split on`,;/\|` + whitespace, allow-list intersection                                                   | Python`re`, `_citation_check()` in [src/rag/pipeline.py](src/rag/pipeline.py)                                          |
| No abstention contract                | Single`ABSTENTION` constant + seven reason codes on the response schema                                                        | Pydantic v2 models in[src/api/schemas.py](src/api/schemas.py)                                                              |
| Confidence read as truth              | User-preference logistic calibrator (schema v2) with strict artifact validation,`available` flag, vision explicitly unmeasured | scikit-learn-compatible logistic weights loaded by`load_calibration()` in [src/rag/confidence.py](src/rag/confidence.py) |
| Weak evidence produced confident text | Answerability check → citation gate → grounding gate → tier escalation → typed abstention                                    | Two-pass pipeline in[src/rag/pipeline.py](src/rag/pipeline.py)                                                             |

### Retrieval and evidence

| Problem                                 | Technique                                                                                                                             | Tech                                                                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Cross-tenant leakage                    | Deterministic per-session Pinecone namespace +`allowed_document_ids()` intersection on every retrieval path + BM25 `_check_owner` | SQLite ownership helpers in[src/core/db.py](src/core/db.py), [src/rag/hybrid.py](src/rag/hybrid.py)                                       |
| Dense-only recall gaps                  | Dense + BM25 with Reciprocal Rank Fusion and cross-encoder rerank                                                                     | Pinecone serverless (dense), SQLite-backed BM25 with Robertson-shifted IDF, HuggingFace`all-MiniLM-L6-v2`, `BAAI/bge-reranker-base` |
| Silent version drift across chats       | `version_mode` `latest`\|`all`, explicit pinning via `document_ids`                                                           | `AskRequest` in [src/api/schemas.py](src/api/schemas.py), version-group query in [src/rag/retriever.py](src/rag/retriever.py)           |
| No passage-level provenance             | Deterministic`evidence_id = sha256(document_id + source + page + modality + locator)`                                               | Hash-based IDs in[src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py)                                                      |
| Users could not see the supporting page | Evidence resolver + rasterised page with bbox highlight                                                                               | PyMuPDF`fitz.Page.draw_rect`, capped at 8 M pixels in [src/api/documents.py](src/api/documents.py)                                     |

### Document lifecycle

| Problem                   | Technique                                                                                                           | Tech                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Opaque ingestion failures | Durable job table + paginated per-attempt history                                                                   | SQLite`jobs` table, `GET /api/documents/{id}/detail` in [src/api/lifecycle.py](src/api/lifecycle.py) |
| No retry / cancel path    | `POST /retry`, `POST /cancel`, cooperative `checkpoint()` polling of `cancel_requested`                     | `JobCancelled` (BaseException) in [src/jobs/runner.py](src/jobs/runner.py)                             |
| Jobs lost on restart      | Startup reconciliation → mark`queued`/`running` as `error(interrupted)`, release pending upload reservations | `reconcile_interrupted_jobs()` in [src/jobs/runner.py](src/jobs/runner.py)                             |
| Upload filename aliasing  | Storage keyed by UUID`doc_id`, not by digest or filename                                                          | [src/api/documents.py](src/api/documents.py)                                                             |
| Deletes racing reads      | Serial`_DELETE_LOCK` + pre-mark `status='deleting'` before remote ops                                           | `threading.Lock` + SQLite status transitions                                                          |
| Version diffs invisible   | Page-union pagination with SHA-256 per-page change detection                                                        | `GET /api/documents/compare` in [src/api/lifecycle.py](src/api/lifecycle.py)                           |

### Tenant safety

| Problem                           | Technique                                                                                      | Tech                                                           |
| --------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Runaway single tenant             | Atomic pending/success/failure reservations against`usage_reservations` + daily UTC counters | [src/core/usage.py](src/core/usage.py), SQLite transactions     |
| Uploads without admission control | Bounded semaphore → HTTP 503 +`Retry-After: 5`                                              | `job_max_pending` in [src/jobs/runner.py](src/jobs/runner.py) |
| Quota errors indistinguishable    | Typed`BudgetExceeded(resource)` → HTTP 429 with `{message, resource, remaining}`          | Exception handler in[src/api/app.py](src/api/app.py)            |
| Users blind to their own budgets  | `GET /api/usage` snapshot with `usage`, `limits`, `remaining`                          | Sidebar`<details>` panel in [web/index.html](web/index.html)  |

### Data protection

| Problem                                   | Technique                                                                              | Tech                                                                                                                              |
| ----------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Uploaded PDFs at rest in plaintext        | Symmetric encryption to`storage_path`; decrypt to a `NamedTemporaryFile` on ingest | `cryptography.Fernet` in [src/core/crypto.py](src/core/crypto.py)                                                                |
| Text/table PII in the vector store        | Presidio-style redaction before chunking; markdown table redaction                     | `redact()` in [src/core/pii.py](src/core/pii.py), invoked from [src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py) |
| Historical messages leaking private paths | Server-side`_public_meta()` strips URLs / disk paths before serving                  | [src/api/conversations.py](src/api/conversations.py)                                                                               |
| Pinecone metadata type limits             | Nested metadata stashed as`_vrag_metadata_json`, restored on read                    | [src/vectorstore/pinecone_store.py](src/vectorstore/pinecone_store.py)                                                             |

### Frontend and integration

| Problem                             | Technique                                                                                                           | Tech                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| SSE frames split across TCP reads   | Incremental parser with trailing-`\r` buffer, blank-line dispatch, multi-line `data:` join                      | `createSSEParser()` in [web/app.js](web/app.js)                  |
| Silent partial completions          | Explicit`done` terminator required; missing `done` → error                                                     | `consumeAskStream()` in [web/app.js](web/app.js)                 |
| SSE dropped session cookie          | Session middleware runs`resolve_session` + `set_session_cookie` on *every* response, including streams        | [src/api/app.py](src/api/app.py)                                   |
| Modal always visible / stole focus  | `hidden` attribute + `inert` siblings + focus capture/restore + Tab-cycle handler                               | Vanilla JS in[web/app.js](web/app.js)                              |
| `[object Object]` in error toasts | Recursive`errorText()` flattens FastAPI `detail` (dict/list/str)                                                | [web/app.js](web/app.js)                                           |
| Stale-conversation retries          | `retry` record carries `view` + `conversation_id` and refuses on mismatch                                     | [web/app.js](web/app.js)                                           |
| Runaway ingestion had no stop       | UI Stop button →`POST /cancel` → `checkpoint()` observes flag                                                 | [web/app.js](web/app.js) + [src/jobs/runner.py](src/jobs/runner.py) |
| Dark/light theme flash on load      | Synchronous inline`<script>` in `<head>` reading `localStorage` + `prefers-color-scheme` before first paint | [web/index.html](web/index.html)                                   |

### Streaming and cancellation

| Problem                                        | Technique                                                                                     | Tech                                                                       |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Live provider tokens shown before verification | SSE forwards only*finalised* text sliced into ~160-char chunks after gates pass             | `_StreamWork` in [src/api/ask_router.py](src/api/ask_router.py)           |
| Streams outlived client disconnect             | ASGI`receive` wrapper observes `http.disconnect`, cancels pipeline, releases slot + quota | `_AnswerStreamResponse` in [src/api/ask_router.py](src/api/ask_router.py) |

### Ops and infrastructure

| Problem                                    | Technique                                                                                 | Tech                                                                                  |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Load balancers using liveness as readiness | `/healthz` (200 always alive) vs `/readyz` (503 while unready) with dependency detail | [src/api/app.py](src/api/app.py)                                                       |
| Metrics blind spot                         | Prometheus counters/histograms for latency, status, session activity                      | `prometheus-fastapi-instrumentator` (mount is currently public — see known limits) |
| Structured logging                         | JSON logs with request IDs and per-route access log                                       | `structlog` + custom middleware in [src/api/app.py](src/api/app.py)                  |
| Prompt injection via document text         | Untrusted content wrapped with`wrap_untrusted()` sentinels before sending to LLMs       | [src/core/prompt_safety.py](src/core/prompt_safety.py)                                 |
| Sessions surviving process restart         | SQLite WAL persistence + signed cookie with rotation-safe id                              | [src/core/session.py](src/core/session.py)                                             |

### Stack summary (open-source only)

- **Runtime.** Python 3.11+ (developed on 3.14), FastAPI, Uvicorn, Pydantic v2, pydantic-settings.
- **LLMs.** Groq-hosted `llama-3.1-8b-instant`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `meta-llama/llama-4-scout-17b-16e-instruct` (vision).
- **Embeddings + rerank.** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-reranker-base` (`sentence-transformers`, `transformers`, `torch`).
- **Vector store.** Pinecone serverless (dense) + SQLite-backed BM25 (`rank_bm25` semantics reimplemented over `document_chunks` for durability).
- **PDF + rendering.** PyMuPDF (`fitz`) for extraction, bbox capture, and page rasterisation.
- **OCR fallback.** Pluggable OCR path in [src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py) (Tesseract-compatible).
- **Storage.** SQLite in WAL mode with foreign keys; additive migrations in [src/core/db.py](src/core/db.py).
- **Crypto.** `cryptography.Fernet` at-rest encryption for uploaded PDFs; signed session cookies via `itsdangerous`.
- **PII.** Regex + rule-based redaction in [src/core/pii.py](src/core/pii.py).
- **Rate limiting.** `slowapi` per-IP limits on ask + upload.
- **Observability.** `structlog` for JSON logs, `prometheus-fastapi-instrumentator` for metrics.
- **Frontend.** Zero-build vanilla JS SPA in [web/](web) — no bundler, no framework, no runtime deps. Inter font via `rsms.me` (CSS only).
- **Jobs.** Custom durable runner in [src/jobs/runner.py](src/jobs/runner.py) backed by the SQLite `jobs` table with startup reconciliation.
- **Tests.** `pytest` contract + policy tests, including Node-executable SSE-parser tests via exported CommonJS from [web/app.js](web/app.js).

---

## Deep dive — retrieval, rerank, compaction, routing, extraction

The catalog above is the *what*. This section is the *how*, referenced
to the actual source. Every claim below was verified against the code
in this repository.

### 1. Ingestion pipeline (PDF → indexable evidence)

Owner: [src/jobs/ingest_job.py](src/jobs/ingest_job.py) driving
[src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py).

Stages, in order:

1. **Structural validation.** Every upload is opened with PyMuPDF and
   probed page-by-page before it is ever scheduled — this catches
   truncated PDFs, encrypted PDFs, and PDFs that pass magic-byte checks
   but fail to parse.
2. **At-rest encryption.** The raw bytes are Fernet-encrypted into
   `storage_path`, keyed by a UUID `doc_id` (never the digest or the
   filename — that is how filename aliasing was closed).
3. **Text extraction.** PyMuPDF `page.get_text("text")` per page. Text
   below `settings.ocr_min_chars_per_page` triggers the OCR fallback
   ([src/extraction/ocr.py](src/extraction/ocr.py)).
4. **Table extraction.** PyMuPDF's table finder emits markdown; each
   cell is passed through `redact()` from
   [src/core/pii.py](src/core/pii.py) before it becomes a chunk.
5. **Visual placements.** For every image on the page the extractor
   records `(bbox, image_path)` and — via
   [src/extraction/vision_summarizer.py](src/extraction/vision_summarizer.py)
   — a redacted textual summary produced by the vision LLM. The image
   pixels themselves are **not** redacted; that is stated explicitly on
   the chunk (`image_redacted: False`) and shown in the evidence UI.
6. **Chunking.** Text is split by
   [src/extraction/chunking.py](src/extraction/chunking.py) using a
   recursive character-based splitter with `chunk_size=900` and
   `chunk_overlap=150` (both env-tunable in
   [config/settings.py](config/settings.py)). Table rows and vision
   summaries are single-chunk.
7. **Evidence ID.** Every chunk carries
   `evidence_id = sha256(document_id + '\0' + source + '\0' + page + '\0' + modality + '\0' + locator)`. This is what the citation
   `[Ei]` labels resolve to when a user clicks them.

### 2. Embeddings

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- Loaded lazily via `get_embeddings()` in
  [src/embeddings/embedder.py](src/embeddings/embedder.py); the model
  is cached process-wide.
- Vectors are L2-normalised at the embedder so cosine similarity
  reduces to a dot product (see `_cosine()` in
  [src/rag/compaction.py](src/rag/compaction.py) line 30).
- The same embedder is reused for dedup, sentence-filter, and HyDE
  re-embedding — no drift between index-time and query-time
  representations.

### 3. Hybrid retrieval

Owner: [src/rag/retriever.py](src/rag/retriever.py) → callable
`retrieve()`.

Three signals feed one fusion step:

- **Dense (original query).** Pinecone `query()` scoped by
  `namespace` and, when the caller pins documents, `document_id $in`
  filter. Top-`retrieval_top_k` (default 10) results.
- **Dense (HyDE query).** `hyde()` from
  [src/rag/query_rewrite.py](src/rag/query_rewrite.py) prompts the
  cheap Groq model to write a 3–5 sentence hypothetical answer to the
  question; that pseudo-passage is embedded and used as a second
  Pinecone query. This lifts recall for zero-shot or lexically weak
  questions. Disabled globally by `hyde_enabled=False`.
- **BM25.** `bm25_store.search()` from
  [src/rag/hybrid.py](src/rag/hybrid.py). Corpus lives in the
  `document_chunks` SQLite table (durable across restarts,
  ownership-checked), tokenised by `[A-Za-z0-9]+`, indexed with
  `BM25Okapi` from `rank-bm25`, built on demand per scoped query
  (there is no long-lived in-process cache to go stale). Top
  `bm25_top_k` (default 20) hits.

Ownership enforcement runs on every path:

- `allowed_document_ids(namespace, ids)` intersects the caller's
  request with their owned, ready documents.
- The result is passed as `ids` into both Pinecone and BM25.
- After fusion, `scoped()` re-filters — a belt-and-braces check so a
  future refactor of either backend can't leak siblings.

### 4. Reciprocal Rank Fusion

Owner: `rrf_fuse()` in
[src/rag/hybrid.py](src/rag/hybrid.py). For each candidate `d` the
score is

$$
\text{score}(d) = \sum_{L \in \{\text{dense}_a,\ \text{dense}_b,\ \text{bm25}\}} \frac{1}{k + \text{rank}_L(d)}
$$

with `k = settings.rrf_k` (default 60). Documents missing from a
list contribute nothing from that list. The top `max(k, 10)` are
returned. RRF was chosen because it is calibration-free — dense
cosines and BM25 scores are on incomparable scales, and RRF only
needs ranks.

### 5. Compaction (dedup → rerank → sentence filter → token budget)

Owner: `compact()` in [src/rag/compaction.py](src/rag/compaction.py).
Runs on the RRF output before it becomes evidence for generation.

**5a. Semantic dedup.** Cosine similarity ≥ 0.95 on chunks from the
same `(document_id, page, modality)` triple drops the later one.
Cross-document near-duplicates are kept — they are legitimately
separate provenance. Visual chunks are exempt (the summary is already
condensed).

**5b. Cross-encoder rerank.** `BAAI/bge-reranker-base` scores each
`(question, chunk)` pair with a cross-encoder. Top `rerank_top_n`
(default 5) survive. If the reranker fails to load (offline, quota,
GPU-OOM) the pipeline falls back to the RRF order — logged as
`rerank_failed_fallback` in structured logs. Cross-encoders are
strictly better at fine-grained relevance than bi-encoder cosine
because they attend jointly to both texts, but they are more
expensive; running them only on the top ~20 RRF survivors is the
usual practical compromise.

**5c. Sentence-level filter.** Each surviving chunk is split on
`(?<=[.!?])\s+`, sentences are embedded, and any sentence with
`cosine(q, sentence) < 0.25` is dropped. If everything gets filtered
the original chunk is preserved — under-fitting is safer than
returning empty context. Tables and visuals bypass this stage; they
are structured, not prose.

**5d. Token budget.** `enforce_budget()` uses `tiktoken`'s
`cl100k_base` encoding to walk chunks in relevance order, keeping
whole chunks until `settings.context_token_budget` (default 3000) is
exhausted, then truncating the last chunk on token boundary
(preserving `evidence_preview` in metadata so the UI can still show
the full quote). This gives the model a fixed and predictable
context window regardless of upstream widths.

### 6. Model router

Owner: [src/rag/router.py](src/rag/router.py).

- Tiers: `cheap` → `llama-3.1-8b-instant`, `mid` →
  `openai/gpt-oss-20b`, `frontier` → `openai/gpt-oss-120b`, `vision`
  → `meta-llama/llama-4-scout-17b-16e-instruct` (env-configurable).
- The router classifies the question with `CLASSIFIER_PROMPT` on the
  cheap model, returning a JSON tag (factual / analytical / visual /
  chit-chat) which maps to a tier.
- Escalation is orthogonal: if the citation gate or grounding gate
  fails at tier N, the pipeline's second attempt jumps to a higher
  tier before abstaining ([src/rag/pipeline.py](src/rag/pipeline.py)).

### 7. Answering, citation, and grounding

Owner: [src/rag/pipeline.py](src/rag/pipeline.py) as an event
generator.

- **Query rewrite.** `standalone_query()` from
  [src/rag/query_rewrite.py](src/rag/query_rewrite.py) resolves
  pronouns and ellipsis against the bounded history (`memory` in
  [src/rag/memory.py](src/rag/memory.py) keeps
  `conversation_memory_turns=4` most-recent turns). History is a
  retrieval hint only — it is **never** cited.
- **Evidence pack.** Compacted chunks are labelled `E1..En` in the
  system prompt. The `[Ei]` tokens are what the model is told to use.
- **Citation gate.** `_citation_check()` scans every `[...]`
  bracket, splits on `,;/\|` + whitespace, validates each token with
  `re.fullmatch(r"[Ee]\d+")`, uppercases, and intersects with the
  allowed labels. Combined forms (`[E1, E2]`) are supported;
  non-citation brackets (`[note]`) are ignored. Invalid labels
  produce warnings.
- **Grounding gate.** A cheap-model call is given only the *cited*
  chunks as text (no image pixels — vision cannot be verified from
  text) and asked whether the answer is supported. Failure → retry
  on higher tier → abstain.
- **Abstention.** One `ABSTENTION` constant, seven typed reasons
  (`no_evidence`, `insufficient_evidence`,
  `invalid_or_missing_citations`, `grounding_unavailable`,
  `unsupported_answer`, `confidence_unavailable`, `low_confidence`).
- **Streaming.** SSE emits verified text in ~160-char chunks *after*
  the gates pass — never live provider deltas. `_StreamWork` owns
  the slot + reservation; `_AnswerStreamResponse` wraps the ASGI
  `receive` to observe `http.disconnect` and cancel the pipeline
  promptly ([src/api/ask_router.py](src/api/ask_router.py)).

### 8. Confidence (preference calibration)

Owner: [src/rag/confidence.py](src/rag/confidence.py).

Feature vector (all in `[0,1]`):

- Top-1 retrieval score.
- Mean of top-3 retrieval scores.
- Rerank margin (top-1 minus mean of the rest).
- Fraction of the answer's sentences that carry a valid citation.
- Grounding-gate verdict (1 = supported, 0 = otherwise).

These features feed a logistic calibrator trained on 👍/👎 signals
from `POST /api/feedback`. The artifact is versioned
(`schema_version=2`, `target="user_preference"`) and rejected at
load time on any mismatch. Vision-only answers return
`available=false, reason="Text samples cannot measure image-only support"`; the UI hides the numeric score when `available=false`.

**Explicit stance.** This calibrates *preference*, not truth. A
popular answer need not be correct.

### 9. Vector-store contract (Pinecone)

Owner: [src/vectorstore/pinecone_store.py](src/vectorstore/pinecone_store.py).

- Serverless index, `pinecone_cloud`/`pinecone_region` env-configurable.
- Namespaces are derived deterministically from the session ID,
  giving hard tenant isolation at the store level.
- Pinecone metadata must be scalar / list-of-scalar. Nested dicts
  are stashed as a single JSON blob field (`_vrag_metadata_json`)
  at write time and restored at read time; this preserves per-chunk
  bboxes, evidence IDs, and modality info without silent drops.
- Every retrieval query is filtered by namespace and, when scoped,
  by `document_id $in [...]`.

### 10. Prompt safety

Owner: [src/core/prompt_safety.py](src/core/prompt_safety.py).

- All untrusted content (document text, table markdown, vision
  summaries, history) is wrapped in `wrap_untrusted("CONTEXT", ...)`
  sentinels before being sent to any LLM. The system prompt tells the
  model to treat sentinel-wrapped content as data, never as
  instructions.
- Question length is capped at `max_question_chars=1000`.
- Groq calls are bounded by `request_timeout_secs=60`.

### 11. Session and cookie contract

Owner: [src/core/session.py](src/core/session.py).

- Session id is a URL-safe random hex; the cookie value is signed
  with `itsdangerous.URLSafeTimedSerializer` keyed by
  `session_secret`.
- The middleware in [src/api/app.py](src/api/app.py) calls
  `resolve_session(request)` on the way in and
  `set_session_cookie(request, resp)` on every response — including
  streaming responses. This is the fix for "SSE loses the
  first-request cookie so subsequent asks appear on a new tenant."
- `Cache-Control: private, no-store` is set on every `/api/*`
  response.
- Security headers on every response: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`.

### 12. Concurrency model

- Single Uvicorn worker (see the "known limits" section for why).
- The job runner is a bounded thread pool with an explicit
  semaphore. Cancellation is cooperative via
  `checkpoint(job_id)` which reads `jobs.cancel_requested` and
  raises `JobCancelled` (a `BaseException`, not `Exception` — this
  is deliberate so `except Exception:` cannot swallow the signal).
- SQLite runs in WAL mode with foreign keys enforced. All
  reservation math and job transitions use `BEGIN IMMEDIATE`
  transactions via the `tx()` helper in
  [src/core/db.py](src/core/db.py).
- Deletes serialise on a module-level `threading.Lock` and pre-mark
  `status='deleting'` before touching Pinecone, so retrieval never
  observes a stale ownership row.

### 13. Rendering evidence with bounding boxes

Owner: [src/api/documents.py](src/api/documents.py).

- `GET /api/documents/{id}/evidence/{eid}` resolves an evidence
  label to filename, page, quote, bbox, and page-image URL.
- `GET /api/documents/{id}/pages/{page}.png?evidence_id=…` renders
  the page with PyMuPDF. Before rasterisation, if the resolved
  evidence carries a bbox, `fitz.Page.draw_rect()` overlays a
  highlight. The output is capped at `MAX_RENDER_PIXELS = 8_000_000`
  by clamping the render matrix; this cap is enforced before the
  pixmap is allocated.
- The UI states whether the highlight is region-level or page-level
  and validates the returned URL is same-origin and prefixed by the
  document path — a defence against a compromised backend serving
  arbitrary URLs.

---