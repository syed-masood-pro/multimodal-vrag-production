# TECHNICAL.md — Multimodal Visual-RAG

Version 3. This document reflects the code that actually ships in this
repository. Where a feature is a heuristic, it is called a heuristic. Where a
limitation exists, it is stated. Older revisions of this file overstated
several claims; this revision was rewritten against source.

---

## 1. Purpose

The system ingests PDFs, extracts text / tables / visual placements per page,
indexes them into a hybrid vector + BM25 store, and answers questions with
evidence-linked citations, abstention on weak evidence, and per-tenant quotas.

Runtime: Python 3.11+ (developed on 3.14), FastAPI + Uvicorn, Groq LLMs
(`llama-3.1-8b-instant`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`,
`meta-llama/llama-4-scout-17b-16e-instruct` for vision), Pinecone serverless
dense index, HuggingFace `all-MiniLM-L6-v2` embeddings, `BAAI/bge-reranker-base`
cross-encoder rerank. Local state in SQLite (WAL). Uploaded PDFs are encrypted
at rest with Fernet.

---

## 2. Repository layout

- [config/settings.py](config/settings.py) — Pydantic-settings; all tunables
  (LLM tiers, quotas, retrieval widths, thresholds) resolved from env.
- [src/core/](src/core) — cross-cutting infra: `db`, `session`, `crypto`,
  `pii`, `file_check`, `prompt_safety`, `audit`, `rate_limit`, `observability`,
  `usage`.
- [src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py) — PyMuPDF
  extraction, chunking, table redaction, per-placement bounding boxes,
  deterministic `evidence_id`.
- [src/embeddings/](src/embeddings) — MiniLM text embeddings, vision captions.
- [src/vectorstore/pinecone_store.py](src/vectorstore/pinecone_store.py) —
  Pinecone client with wire-safe metadata (JSON stash for dicts/lists),
  ownership-scoped queries.
- [src/rag/](src/rag) — `retriever`, `hybrid` (SQLite-backed BM25), `pipeline`
  (evidence-first answering), `confidence` (preference calibration), `memory`.
- [src/jobs/](src/jobs) — durable job runner and ingestion job.
- [src/api/](src/api) — FastAPI app: `app`, `documents`, `ask_router`,
  `conversations`, `lifecycle`, `schemas`.
- [web/](web) — zero-build vanilla JS SPA.
- [tests/](tests) — contract and policy tests.

---

## 3. Data model (SQLite, WAL, FK on)

Schema is created and migrated additively in [src/core/db.py](src/core/db.py).

- `sessions(id, ...)` — anonymous browser session; used as tenancy key and to
  derive per-tenant Pinecone namespace.
- `documents(id, session_id, filename, sha256, size_bytes, pages, chunks, version, status, storage_path, created_at, updated_at, attempt_count, last_error, job_id)` — one row per uploaded version.
- `document_chunks(namespace, document_id, chunk_id, content, metadata_json)`
  — SQLite-backed BM25 corpus. A trigger cascades chunk deletion when the
  parent document row is removed.
- `jobs(id, session_id, kind, document_id, status, stage, progress, total, result, message, created_at, updated_at, cancel_requested)` — durable job
  state; survives process restarts.
- `usage_reservations(id, session_id, kind, day, amount, state, document_id, success, created_at)` — atomic quota reservation ledger (see §8).
- `conversations`, `messages`, `feedback`, `audit_log` — chat and audit.

Ownership helpers in `db`:

- `namespace_owner(namespace)` → session that owns the namespace.
- `allowed_document_ids(session_id, requested)` → intersect a caller-supplied
  scope with the caller’s owned, ready documents; retrieval short-circuits to
  empty if the intersection is empty.

Migrations are additive: `ALTER TABLE ... ADD COLUMN` guarded by `PRAGMA table_info` reads. No destructive migrations.

---

## 4. Ingestion pipeline

Entry: `POST /api/documents` in
[src/api/documents.py](src/api/documents.py).

1. **Upload validation** — magic-byte check + full PyMuPDF structural probe
   (`_validate_pdf`). Storage path is derived from `doc_id` (a UUID hex), not
   from the filename or digest, which prevents cross-tenant aliasing of
   uploads with the same name.
2. **At-rest encryption** — the PDF is Fernet-encrypted before being written
   to `storage_path`. The key comes from `settings.storage_key`.
3. **Pending reservation** — `usage.reserve_upload()` records a `pending`
   reservation charging one document and `size_bytes` against the tenant’s
   quota. If quota is exceeded the endpoint returns HTTP 429 with a resource
   identifier (see §8).
4. **Job creation** — `runner.create_job(session_id, "ingest", document_id=…)`
   inserts a row bound to the document. The runner is bounded by
   `settings.job_max_pending`.

Job execution: [src/jobs/ingest_job.py](src/jobs/ingest_job.py).

- Decryption writes the plaintext PDF to a `tempfile.NamedTemporaryFile`
  (the whole ciphertext is currently loaded into memory before write; this
  is a known limitation for very large PDFs). The temp file is deleted in
  the job's `finally` block.
- `checkpoint(job_id)` is called before every stage; if
  `cancel_requested=1`, the worker raises `JobCancelled` (a
  `BaseException`, not `Exception`, so it is never swallowed by `except Exception`) and terminates cleanly.
- Per-page extraction from
  [src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py) yields:
  - text chunks with token-bounded windows,
  - table Markdown with PII redaction (`_table_docs` calls `redact()`),
  - visual placements with per-placement bbox and `image_redacted: False`
    (visuals are not OCR-scrubbed; that limitation is stated in evidence UI).
- `reserve_pages()` charges pages against the daily page budget; if it
  exceeds, the job aborts before any vector writes.
- **Publication ordering (not a distributed transaction).** Dense vectors
  are upserted into Pinecone first. Then a single SQLite transaction
  (1) writes chunks into `document_chunks` (BM25 corpus),
  (2) sets `documents.status='ready'` and clears `last_error`,
  (3) marks the job `done`,
  (4) commits the upload reservation as `success=1`.
  This means Pinecone and SQLite are **not** in a common transaction. If
  the SQLite commit fails after the Pinecone upsert succeeds, the failure
  handler attempts to delete the just-upserted vectors; if that cleanup
  itself fails, the document stays in `error` with `cleanup_required=True`
  and orphan vectors remain in Pinecone until a manual sweep. Retrieval
  never surfaces the orphans because ownership is gated on the SQLite
  `documents` row.

Failure paths:

- Any exception → `documents.status='error'`, `last_error` populated,
  `attempt_count` incremented, temp file removed, unpublished vectors
  deleted, upload reservation finalised as failure.
- Process crash → on startup `runner.reconcile_interrupted_jobs()` walks
  `jobs` with status `queued` or `running`, marks them `error` with reason
  `interrupted`, releases pending **upload** reservations, and requires an
  explicit retry via `POST /api/documents/{id}/retry`. Pending **page**
  reservations for the same crash window are not released automatically
  and will decay only by the daily rollover; this is a known limitation.
  Encrypted `storage_path` files are also not swept on startup — they are
  cleaned up only when the document row is deleted via the API.

---

## 5. Retrieval

Hybrid retrieval in [src/rag/retriever.py](src/rag/retriever.py) and
[src/rag/hybrid.py](src/rag/hybrid.py).

- **Ownership first.** Every retrieval path resolves
  `allowed_document_ids(session_id, requested)`. If the caller pinned a
  scope and it is not a subset of their owned ready documents, retrieval
  short-circuits to an empty result — no fallback to broad search that
  could leak siblings.
- **Dense.** Pinecone query filtered by `namespace` and optional
  `document_id` `$in`. Metadata that is a dict/list is stashed as
  `_vrag_metadata_json` at write time and restored at read time; Pinecone’s
  metadata type restrictions cannot silently drop nested fields.
- **BM25.** `BM25Store` in [src/rag/hybrid.py](src/rag/hybrid.py) is
  SQLite-backed. Rows come from `document_chunks`. Ownership is re-checked
  per query via `_check_owner()`. Small corpora use a Robertson shifted IDF
  (positive floor) so a single-document corpus is still ranked sensibly.
- **Fusion + rerank.** Reciprocal-rank fusion of the two lists, then
  `BAAI/bge-reranker-base` cross-encoder rerank of the top-k. Widths and
  thresholds come from `config/settings.py`.
- **Version mode.** `AskRequest.version_mode` is `latest` (default) or
  `all`. `latest` narrows a scope to the maximum `documents.version`
  per `(session_id, filename)` group. To pin a specific version, pass
  its `document_id` in `document_ids` — there is no separate `"pinned"`
  mode value.

---

## 6. Answering pipeline

Implemented in [src/rag/pipeline.py](src/rag/pipeline.py) as an event
generator. Every answer is verified before it is streamed to the client.

Order of operations:

1. **Retrieval** with the caller’s scope.
2. **Evidence compaction.** Retrieved chunks become an evidence list with
   deterministic labels `E1, E2, …` and stable `evidence_id`s from
   extraction. Labels are what the model is instructed to cite.
3. **Query verification pass** (`verifying` event) — a small model
   answerability check.
4. **Generation.** The chosen tier LLM is called with the evidence-labelled
   context.
5. **Citation gate.** `_citation_check()` scans every `[...]` bracket in
   the answer, splits its contents on `,`, `;`, `/`, `|` or whitespace,
   and validates each token individually against the regex `[Ee]\d+`
   before intersecting with the evidence-pack labels. This accepts
   both singleton `[E1]` and combined forms such as `[E1, E2]`,
   `[E1; E3]`, `[E1 E2]`. Non-citation brackets (e.g. `[note]`) are
   ignored. Missing / invented labels are recorded in
   `citation_warnings`. This is a document-level check, not a
   per-sentence entailment check.6. **Grounding gate.** A small model is asked whether the answer is
   supported by the cited evidence text. Unsupported answers do not reach
   the client.
6. **Escalation.** Up to two attempts. On citation / grounding failure the
   pipeline escalates to a stronger tier before abstaining.
7. **Confidence (optional).** See §7.

**Abstention.** If evidence is missing, insufficient, or the generated
answer fails a gate, the pipeline emits the constant

```
ABSTENTION = "I cannot answer this question reliably from the available document evidence."
```

with one of the reasons:

- `no_evidence`
- `insufficient_evidence`
- `invalid_or_missing_citations`
- `grounding_unavailable`
- `unsupported_answer`
- `confidence_unavailable`
- `low_confidence`

The response carries `answerable`, `abstained`, `abstention_reason`,
`status ∈ {answered, abstained}`, and `citation_warnings`.

**Streaming.** SSE tokens are sliced from the *finalised* answer text in
~160-char chunks, not live provider deltas. This means the client never
sees text that later fails a gate. See
[src/api/ask_router.py](src/api/ask_router.py) `_StreamWork` and
`_AnswerStreamResponse`, which wraps the ASGI `receive` callable to observe
`http.disconnect` and cancel the pipeline promptly.

---

## 7. Confidence (preference calibration, not truth)

[src/rag/confidence.py](src/rag/confidence.py) implements the confidence
score. It is derived from retrieval + rerank + citation features and
optionally passed through a logistic calibrator trained on **user
preference feedback** (👍/👎).

Explicit stance:

- The calibrator predicts *user preference*, not truth. A popular answer
  need not be correct.
- The response includes `confidence_available: bool` and, when false, a
  `confidence_unavailable_reason`. Callers must not display a score if
  `available=false`.
- Vision answers currently return `available=false` with reason
  `"Text samples cannot measure image-only support"`.

Calibration artifact schema:

- `schema_version: 2`, `feature_version`, `target: "user_preference"`,
  `mode`, `method: "logistic"`, `keys`, `component_weights`, `intercept`,
  `metrics.train`, `metrics.heldout`.
- `load_calibration()` refuses legacy artifacts, mixed feature sets, and
  non-finite weights.

---

## 8. Tenant usage quotas

[src/core/usage.py](src/core/usage.py) implements atomic reservations
against `usage_reservations`.

Settings (from [config/settings.py](config/settings.py)):

- `tenant_max_documents=100`
- `tenant_max_storage_mb=500`
- `tenant_daily_asks=100`
- `tenant_daily_pages=500`
- `job_max_pending=8`
- `max_pdf_pages=300`

State machine per reservation:

- `pending` on entry (`reserve_ask`, `reserve_upload`, `reserve_pages`),
- `success=1` on `finish_*` when the caller reports success,
- `success=0` on `finish_*` when the caller reports failure or aborts.

Daily counters are computed against the **UTC** calendar day. There is
no timezone override; a `tenancy_timezone` setting does not exist.
`snapshot()` returns `usage`, `limits`, and `remaining` for `/api/usage`.

`BudgetExceeded(resource)` is mapped to HTTP 429 in
[src/api/app.py](src/api/app.py) with body
`{"detail": {"message": "Browser session quota exceeded.", "resource": ..., "remaining": 0}}`. Possible resources: `documents`,
`storage_bytes`, `daily_asks`, `daily_pages`, `pdf_pages`. Job-runner
queue overflow is a separate case: it returns HTTP **503** with
`Retry-After: 5` (see [src/api/documents.py](src/api/documents.py) and
[src/api/lifecycle.py](src/api/lifecycle.py)), not 429.

Known limit: reservations are atomic within a single SQLite database.
Horizontal scaling requires replacing this backend or fronting it with a
shared coordinator.

---

## 9. HTTP surface

Routers registered in [src/api/app.py](src/api/app.py):

- `GET /healthz` — liveness (always 200 while the process is up).
- `GET /readyz` — readiness. Returns HTTP **503** with `ready=false` and
  a `details` map while dependencies (DB, embeddings, vectorstore, runner)
  are still coming up. Load balancers must gate on this, not `/healthz`.
- `POST /api/documents` — upload PDF.
- `GET /api/documents` — list caller’s documents.
- `DELETE /api/documents/{id}` — delete. Serialised by `_DELETE_LOCK`;
  sets `status='deleting'` before touching Pinecone so a concurrent
  retrieval never sees stale ownership.
- `GET /api/documents/{id}/detail` — document + paginated job history.
- `POST /api/documents/{id}/retry` — retry after `error`/`cancelled`.
- `POST /api/documents/{id}/cancel` — request cancellation of an active
  ingest.
- `GET /api/documents/{id}/evidence/{evidence_id}` — resolve an evidence
  handle: filename, page, quote, bbox (may be null), and page image URL.
- `GET /api/documents/{id}/pages/{page}.png?evidence_id=…` — rendered
  page image, capped at `MAX_RENDER_PIXELS = 8_000_000`; when
  `evidence_id` resolves to a bbox, that bbox is highlighted with
  `fitz.Page.draw_rect` before rasterisation.
- `GET /api/documents/compare?left_id=&right_id=&offset=&limit=` —
  paginated per-page extracted-text comparison. Char-capped per side;
  `changed` is computed on the *full* extracted text (SHA-256 per page),
  not the truncated preview.
- `POST /api/ask` — non-stream ask.
- `POST /api/ask/stream` — SSE ask. Requires an explicit `done` terminator
  event.
- `GET /api/conversations`, `GET /api/conversations/{id}/messages`,
  `DELETE /api/conversations/{id}`.
- `POST /api/feedback`.
- `GET /api/jobs/{id}`, `GET /api/jobs/{id}/stream`.
- `GET /api/usage` — quota snapshot.

Session middleware ([src/api/app.py](src/api/app.py)) calls
`resolve_session` on the request and `set_session_cookie` on every
response, including SSE responses. This is the fix for “SSE loses the
first-request cookie so subsequent asks appear on a new tenant.”

---

## 10. Frontend

Single-page vanilla JS in [web/](web).

- [web/index.html](web/index.html) — sidebar with document list, refresh,
  compare, usage `<details>` panel, version-mode selector, clear-scope
  and stop buttons; live-region stream status; evidence modal with
  `dialog` semantics.
- [web/app.js](web/app.js) — incremental `createSSEParser()` that handles
  CR/LF correctly (buffers trailing CR); `consumeAskStream()` treats a
  stream without `done` as an error; `errorText()` recursively flattens
  API `detail` payloads (dict / list / str); `controls()` disables
  navigation while a request is in flight; scope state
  (`applyScope/changeScope/saveScope/restoreScope`) survives conversation
  switches; evidence rendering shows the bbox highlight when precise,
  and shows an explicit “page-level evidence only” note when not.

---

## 11. Extraction, PII, and evidence identity

[src/extraction/pdf_extractor.py](src/extraction/pdf_extractor.py) emits,
for every chunk, a metadata block including a deterministic

```
evidence_id = sha256(document_id ⊕ '\0' ⊕ source ⊕ '\0' ⊕ page ⊕ '\0' ⊕ modality ⊕ '\0' ⊕ locator)
```

This is what citation labels resolve to and what
`GET /api/documents/{id}/evidence/{eid}` looks up. `original_source` is
preserved so the registered filename survives temp-file paths.

PII handling:

- Text and table chunks are passed through `redact()` from
  [src/core/pii.py](src/core/pii.py) before indexing.
- Visual placements are indexed with the flag `image_redacted: False`.
  Images are not OCR-scrubbed; the evidence UI states this explicitly.
- The Pinecone metadata that reaches the wire is a projection: quotes and
  redacted previews, not raw text. `_public_meta()` in
  [src/api/conversations.py](src/api/conversations.py) additionally strips
  image URLs / private paths from historical messages before serving.

---

## 12. Configuration reference (selected)

All names below are the **actual** field names in
[config/settings.py](config/settings.py). Everything is env-overridable
via a `.env` file or process environment.

- LLMs: `model_cheap`, `model_mid`, `model_frontier`, `model_vision`.
- Embeddings & rerank: `embedding_model`, `reranker_model`.
- Retrieval: `retrieval_top_k`, `bm25_top_k`, `rerank_top_n`,
  `rrf_k`, `context_token_budget`, `hyde_enabled`.
- Confidence: `confidence_mode` (`off`/`fast`/`full`), `confidence_high`,
  `confidence_low`.
- Storage & security: `sqlite_path`, `upload_dir`, `encryption_key`
  (Fernet, auto-generated if empty), `session_secret`, `session_cookie`.
- API: `api_host`, `api_port`, `api_key` (empty disables auth in dev —
  see §13), `allowed_origins`.
- Quotas: `tenant_max_documents`, `tenant_max_storage_mb`,
  `tenant_daily_asks`, `tenant_daily_pages`, `job_max_pending`,
  `max_pdf_pages`.
- Rate limiting: `rate_ask`, `rate_upload` (per-IP, SlowAPI syntax).
- Runtime: `request_timeout_secs`, `max_question_chars`, `log_level`,
  `log_json`.

There is no `otel_endpoint` and no `tenancy_timezone` setting. Only
Prometheus metrics are exported (§13).

---

## 13. What this system does *not* do

Stated so operators are not surprised:

- **Single-process only.** Tenant quotas (`src/core/usage.py`), the
  delete lock, and the job-runner semaphore all live in in-process
  state. Running with `uvicorn --workers >1` or multiple gunicorn
  workers silently breaks quota accounting, delete serialisation, and
  admission control. `scripts/serve.py` launches a single worker; do
  not scale it horizontally without replacing this state.
- **Pinecone + SQLite are not a distributed transaction.** See §4;
  crash between vector upsert and SQLite commit can orphan vectors.
- **CORS default is `*` with `allow_credentials=True`.** Browsers
  reject the combination, so nothing works cross-origin out of the box
  — but you must set `allowed_origins` explicitly for real cross-origin
  clients rather than relying on the default.
- **Auth is off by default.** `api_key=""` disables the API-key check
  and the dependency is not currently wired into any router. All
  endpoints are anonymous-cookie-only. Add reverse-proxy auth or wire
  `src/core/auth.py` before exposing publicly.
- **Session cookie `Secure` follows the request scheme.** Behind a TLS
  proxy without forwarded-header trust, cookies may be issued without
  `Secure`. Configure `--forwarded-allow-ips` or terminate TLS on the
  app.
- **No CSRF token.** State-changing endpoints trust the session cookie
  with `SameSite=lax`. Not sufficient for hostile same-site subdomains.
- **Fernet key has no rotation.** Rotating `encryption_key` invalidates
  every stored PDF; use `MultiFernet` externally if rotation is needed.
- **PDF decryption loads the whole ciphertext into memory** before
  writing the tempfile.
- **No image OCR redaction.** Visual evidence may contain PII pixels;
  chunks are flagged `image_redacted: False`.
- **No truth calibration.** Only user-preference calibration.
- **SSE forwards verified text, not live provider tokens;**
  time-to-first-token is the pipeline's, not Groq's.
- **Only Prometheus metrics.** No OpenTelemetry exporter is wired,
  despite settings-file appearances.
- **`/metrics` is unauthenticated.** Bind to a private port in
  production or front with auth.
- **Cancellation latency is bounded by the current LLM call.**
  `checkpoint()` fires between stages, not during a Groq request.
- **Page-quota reservations survive a mid-ingest crash** until the day
  rolls over; only upload reservations are reconciled at startup.
- **Encrypted PDFs are not garbage-collected** after failed ingests
  unless the document row is deleted via the API.

---

## 14. Tests

Contract and policy tests in [tests/](tests):

- [test_answer_quality.py](tests/test_answer_quality.py) — citation and
  grounding gate behaviour, abstention reasons.
- [test_api_contract.py](tests/test_api_contract.py) — response schemas.
- [test_lifecycle.py](tests/test_lifecycle.py),
  [test_lifecycle_contracts.py](tests/test_lifecycle_contracts.py) —
  retry / cancel / detail semantics.
- [test_frontend_contract.py](tests/test_frontend_contract.py) — SSE
  parser and stream-consumer invariants (Node-executable exports from
  [web/app.js](web/app.js)).
- [test_quality_policy.py](tests/test_quality_policy.py) — pipeline
  refuses to publish uncited or unsupported answers.
- [test_pipeline_validation.py](tests/test_pipeline_validation.py),
  [test_compaction.py](tests/test_compaction.py),
  [test_router.py](tests/test_router.py),
  [test_hashing.py](tests/test_hashing.py).

Run: `pytest -q` from the repo root.

---

## 15. Change log against v2 of this document

- Removed the “32/32 addressed” claim (unverifiable).
- Removed “PII never enters the vector store” (visuals are not scrubbed).
- Added durable job runner, reconciliation, SQLite-backed BM25, tenant
  quotas, evidence-first pipeline, calibration schema v2, evidence bbox
  rendering, version-aware search and comparison, `/readyz` 503
  semantics, SSE cookie fix, `_StreamWork` disconnect cancellation,
  additive migrations.
- Documented every abstention reason and made the abstention contract
  part of the API surface.

### Corrections applied in v3.1 (post-audit)

- **§4** — replaced “atomic publication” with an honest ordering
  description; Pinecone upsert precedes the SQLite commit and is not
  part of the same transaction.
- **§4** — noted that decryption loads the whole ciphertext into
  memory before writing the tempfile.
- **§4** — clarified that reconcile releases upload reservations only;
  page reservations and encrypted files are not swept on startup.
- **§5** — removed `version_mode="pinned"`; only `latest` and `all`
  are accepted.
- **§6** — citation gate is document-level with singleton `[Ei]`
  regex; the combined `[E1, E2]` form is a known false-positive.
- **§8** — corrected 429 body shape to actual
  `{message, resource, remaining}`; noted queue overflow returns 503,
  not 429; corrected timezone to UTC.
- **§12** — rewrote configuration reference to use the real setting
  names in [config/settings.py](config/settings.py); the previous
  section named several fields that do not exist.
- **§13** — expanded honest limits (single-process assumption, CORS
  default, auth off by default, no CSRF, no Fernet rotation, no OTEL,
  unauthenticated `/metrics`, cancellation latency).