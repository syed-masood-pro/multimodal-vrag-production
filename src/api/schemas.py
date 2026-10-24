"""Pydantic schemas for API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from config import settings


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=settings.max_question_chars)
    conversation_id: str | None = None
    document_id: str | None = None
    document_ids: list[str] | None = Field(default=None, max_length=1000)
    version_mode: Literal["latest", "all"] = "latest"

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()


class SourceItem(BaseModel):
    label: str | None = None
    evidence_id: str | None = None
    chunk_id: str | None = None
    filename: str | None = None
    page: int | None = None
    modality: str | None = None
    source: str | None = None
    document_id: str | None = None
    bbox: list[float] | None = None
    quote: str | None = None
    preview: str | None = None
    cited: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    tier: str
    model: str
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_components: dict[str, float] = Field(default_factory=dict)
    escalated: bool = False
    sources: list[SourceItem] = Field(default_factory=list)
    # Preserve the RAGResponse shape without exposing private filesystem paths.
    image_paths: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    conversation_id: str | None = None
    message_id: int | None = None
    injection_flagged: bool = False
    classification: dict[str, Any] = Field(default_factory=dict)
    answerable: bool = True
    abstained: bool = False
    abstention_reason: str | None = None
    status: str = "answered"
    citation_warnings: list[str] = Field(default_factory=list)
    retrieval_query: str = ""
    confidence_available: bool = False
    confidence_unavailable_reason: str | None = None
    calibration: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class ReadinessResponse(BaseModel):
    ready: bool
    details: dict[str, str] = Field(default_factory=dict)


class DocumentItem(BaseModel):
    id: str
    filename: str
    version: int
    size_bytes: int
    pages: int | None = None
    chunks: int | None = None
    status: str
    created_at: float
    updated_at: float
    attempt_count: int = 0
    last_error: str | None = None
    job_id: str | None = None


class ConversationRequest(BaseModel):
    document_id: str | None = None
    title: str | None = Field(default=None, max_length=200)


class ConversationItem(BaseModel):
    id: str
    title: str | None = None
    document_id: str | None = None
    created_at: float
    updated_at: float


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    meta: dict[str, Any] | None = None
    created_at: float


class JobItem(BaseModel):
    id: str
    kind: str
    status: str
    progress: int
    total: int
    stage: str | None
    message: str | None
    result: dict[str, Any] | None
    created_at: float
    updated_at: float
    document_id: str | None = None
    cancel_requested: bool = False


class DocumentDetailResponse(BaseModel):
    document: DocumentItem
    jobs: list[JobItem] = Field(max_length=100)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    total_jobs: int = Field(ge=0)
    next_offset: int | None = Field(ge=0)


class DocumentVersionItem(BaseModel):
    id: str
    filename: str
    version: int = Field(ge=1)


class ComparisonPageItem(BaseModel):
    page: int = Field(ge=1)
    left_text: str = Field(max_length=200000)
    right_text: str = Field(max_length=200000)
    changed: bool


class DocumentCompareResponse(BaseModel):
    left: DocumentVersionItem
    right: DocumentVersionItem
    pages: list[ComparisonPageItem] = Field(max_length=50)
    truncated: bool
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=50)
    max_chars: int = Field(ge=1, le=200000)
    total_pages: int = Field(ge=0)
    next_offset: int | None = Field(ge=0)
    text_truncated: bool


class UsageCounters(BaseModel):
    documents: int = Field(ge=0)
    storage_bytes: int = Field(ge=0)
    daily_asks: int = Field(ge=0)
    daily_pages: int = Field(ge=0)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)


class UsageSnapshotResponse(BaseModel):
    day: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    timezone: Literal["UTC"]
    limits: UsageCounters
    usage: UsageCounters
    remaining: UsageCounters
    pending_uploads: int = Field(ge=0)
    pending_storage_bytes: int = Field(ge=0)
    stored_documents: int = Field(ge=0)
    stored_bytes: int = Field(ge=0)


class DocumentRetryResponse(BaseModel):
    document_id: str
    job_id: str
    status: Literal["pending"]


class DocumentCancelResponse(BaseModel):
    document_id: str
    job_id: str
    cancel_requested: Literal[True]


class FeedbackRequest(BaseModel):
    message_id: int
    vote: Literal[-1, 1]
    reason: str | None = Field(default=None, max_length=2000)
