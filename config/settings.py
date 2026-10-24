"""Centralized, validated configuration. Fails fast on missing/invalid values."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Secrets (optional so the UI can boot; API calls will fail until set)
    groq_api_key: str = ""
    pinecone_api_key: str = ""

    # Pinecone
    pinecone_index_name: str = "multimodal-vrag"
    pinecone_namespace: str = "demo"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Paths
    pdf_path: Path = Path("data/HPSAs by Score .pdf")
    image_dir: Path = Path("extracted_image")
    cache_dir: Path = Path("cache")

    # Models
    model_cheap: str = "qwen/qwen3.8-27b"
    model_mid: str = "openai/gpt-oss-20b"
    model_frontier: str = "openai/gpt-oss-120b"
    model_vision: str = "qwen/qwen3.8-27b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Retrieval / compaction
    retrieval_top_k: int = 10
    rerank_top_n: int = 5
    rerank_min_score: float=0.10
    context_token_budget: int = 3000

    # Confidence
    confidence_mode: Literal["off", "fast", "full"] = "fast"
    confidence_high: float = 0.75
    confidence_low: float = 0.50

    # Logging
    log_level: str = "INFO"
    log_json: bool = False
    log_dir: Path = Path("")

    # Safety
    request_timeout_secs: int = 60
    max_question_chars: int = 1000
    min_image_bytes: int = 5000
    min_image_dim: int = 100

    # API server
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: str = ""  # empty = auth disabled (dev)
    allowed_origins: str = "*"  # comma-separated

    # Persistence & security
    sqlite_path: Path = Path("data/vrag.db")
    upload_dir: Path = Path("uploads")
    encryption_key: str = ""  # base64 fernet key; auto-generated if empty
    session_secret: str = "change-me-in-prod"
    session_cookie: str = "vrag_sid"

    # RAG upgrades
    chunk_size: int = 1200
    chunk_overlap: int = 300
    bm25_top_k: int = 20
    rrf_k: int = 60
    hyde_enabled: bool = True
    conversation_memory_turns: int = 4
    ocr_min_chars_per_page: int = 40  # under this → try OCR

    # Rate limiting (per IP)
    rate_ask: str = "30/minute"
    rate_upload: str = "5/minute"

    # Local single-process tenant quotas; daily counters use UTC calendar days.
    tenant_max_documents: int = Field(default=100, ge=0)
    tenant_max_storage_mb: int = Field(default=500, ge=0)
    tenant_daily_asks: int = Field(default=100, ge=0)
    tenant_daily_pages: int = Field(default=500, ge=0)
    job_max_pending: int = Field(default=8, ge=1)
    max_pdf_pages: int = Field(default=300, ge=1)

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @field_validator("image_dir", "cache_dir", "upload_dir")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("sqlite_path")
    @classmethod
    def _ensure_sqlite_parent(cls, v: Path) -> Path:
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


settings = Settings()  # type: ignore[call-arg]
