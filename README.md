# Multimodal VRAG (Production Grade)

Production-ready multimodal Retrieval-Augmented Generation over a PDF, with:

- **PDF extraction**: text + tables + images (PyMuPDF)
- **Vision summarization** (Groq vision LLM) with **on-disk cache** (no repeat cost)
- **HuggingFace embeddings** + **Pinecone serverless** with **idempotent** deterministic-ID upsert
- **Context Compaction**: dedup → cross-encoder rerank → sentence filter → token budget (tiktoken)
- **Model Routing** across four tiers: cheap / mid / frontier / specialist (vision)
- **Calibrated Confidence (RLCD-inspired)**: grounding + self-consistency + contrastive delta, with automatic escalation on LOW confidence
- **Retries** (tenacity), **structured logging** (structlog), **cost tracking**, **typed exceptions**, **pydantic-settings** config, **pytest** suite

## Architecture

![Multimodal Visual-RAG Architecture](https://github.com/user-attachments/assets/584c2743-3a4c-4ab1-a5a1-38b5168a99db)

## UI Walkthrough & Features

The repository includes a zero-build vanilla JS SPA featuring live SSE streaming, document management, and interactive evidence rendering.

**Evidence-First Answering & Visual Citations**
The model strictly cites its sources. Clicking a citation (`[E1]`) opens the evidence modal, which renders the specific page and highlights the exact bounding box of the extracted text, table, or visual chart used to generate the answer.
![Evidence Bounding Box](https://github.com/user-attachments/assets/2eaacba5-1f95-49e4-8ac0-b8663f5682bf)


**Multimodal Capabilities & Model Routing**
Queries are automatically routed to the appropriate model tier (cheap, mid, frontier, or vision). The UI displays the confidence score, the model tier used, and the specific chunks retrieved.
![Multimodal Q&A](https://github.com/user-attachments/assets/83f01923-da7c-4f17-a56c-423d64e98cfc)


![Standard Q&A](https://github.com/user-attachments/assets/fb44f472-2302-4683-8c81-0cecd97a61bb)

**Document Lifecycle & Version Control**
Upload documents, track ingestion jobs, and manage file versions. The system supports semantic comparison across different uploaded versions of a document.
![Document Lifecycle](https://github.com/user-attachments/assets/40e3a470-4613-4771-8fc3-098ea1332d31)

![Document Compare](https://github.com/user-attachments/assets/cef0f115-b243-49e6-89e4-b5df67a49309)


**Tenant Usage & Quotas**
Atomic quota reservations track daily asks, pages processed, and storage limits to prevent runaway tenant usage.
![Usage Quotas](https://github.com/user-attachments/assets/27d8f30e-f1c5-4ef3-a63d-c60ad0a24dca)


**Clean Chat Interface**
![Chat Interface](https://github.com/user-attachments/assets/cc34ced4-9763-400e-9b72-f23125d3382e)

![Document Scope](https://github.com/user-attachments/assets/5858a171-466d-4729-ab34-198eca77a8c6)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env with real GROQ_API_KEY and PINECONE_API_KEY
```

Put your PDF at the path configured in `.env` (default `data/HPSAs by Score .pdf`).

## Use

Ingest (idempotent — safe to re-run):

```powershell
python -m scripts.ingest
python -m scripts.ingest --reset     # only if you want to wipe the namespace
```

Ask:

```powershell
python -m scripts.query "According to the revenue graph, which quarter had the highest revenue?"
python -m scripts.query "..." --json
```

Or run the demo:

```powershell
python main.py
```

## Web UI (production)

Start the FastAPI server + polished web UI:

```powershell
python -m scripts.serve
```

Open http://127.0.0.1:8000 in your browser.

Features:

- Dark-theme chat interface with suggestion chips
- Confidence gauge (HIGH / MED / LOW) with component breakdown
- Model-tier badge (cheap / mid / frontier / vision) + escalation flag
- Source chips (page + modality) and inline image previews (click to zoom)
- Live cost snapshot, latency stats, health indicator
- Optional `X-API-Key` header — set `API_KEY` in `.env` for production

Endpoints:

- `POST /api/ask`      — `{question}` → full RAGResponse JSON
- `POST /api/ingest`   — `{reset?, pdf_path?}` → `{upserted, pdf}`
- `GET  /api/health`   — liveness
- `GET  /api/stats`    — per-model cost snapshot

## Test

```powershell
pytest -q
```

## Layout

```
config/settings.py          Pydantic-validated env
src/core/                   logger, exceptions, retry, cost_tracker
src/clients/                Groq + Pinecone wrappers
src/extraction/             pdf_extractor + vision_summarizer (cached)
src/embeddings/             HF embeddings singleton
src/vectorstore/            Pinecone store, deterministic IDs
src/rag/
  prompts.py                All prompts
  retriever.py              Retrieval + context format
  compaction.py             Dedup / rerank / sentence filter / budget
  router.py                 Cheap/Mid/Frontier/Vision routing + fallback
  confidence.py             RLCD-inspired calibrated confidence
  text_chain.py             Text answering
  vision_chain.py           Vision answering
  pipeline.py               Orchestrator
scripts/ingest.py           Ingest CLI
scripts/query.py            Query CLI
tests/                      pytest suite
```

## Confidence Modes

Set in `.env`:

- `CONFIDENCE_MODE=off`  — no extra calls
- `CONFIDENCE_MODE=fast` — grounding + 2-sample consistency (default)
- `CONFIDENCE_MODE=full` — grounding + 5-sample consistency + contrastive delta

LOW-confidence answers on cheap/mid tiers automatically **escalate** one tier and re-answer.
