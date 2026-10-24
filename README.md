# Multimodal VRAG (Production Grade)

Production-ready multimodal Retrieval-Augmented Generation over a PDF, with:

- **PDF extraction**: text + tables + images (PyMuPDF)
- **Vision summarization** (Groq vision LLM) with **on-disk cache** (no repeat cost)
- **HuggingFace embeddings** + **Pinecone serverless** with **idempotent** deterministic-ID upsert
- **Context Compaction**: dedup → cross-encoder rerank → sentence filter → token budget (tiktoken)
- **Model Routing** across four tiers: cheap / mid / frontier / specialist (vision)
- **Calibrated Confidence (RLCD-inspired)**: grounding + self-consistency + contrastive delta, with automatic escalation on LOW confidence
- **Retries** (tenacity), **structured logging** (structlog), **cost tracking**, **typed exceptions**, **pydantic-settings** config, **pytest** suite

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