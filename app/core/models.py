from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class KBFileInfo(BaseModel):
    filename: str
    size_bytes: int


class KBStatus(BaseModel):
    file_count: int
    chunk_count: int
    last_indexed: Optional[str]
    embedding_model: Optional[str]


class EvidenceItem(BaseModel):
    source_file: str
    chunk_id: str
    text: str
    score: float


class SpanResult(BaseModel):
    start: int
    end: int
    label: str
    confidence: float
    claim: str
    evidence: List[EvidenceItem]


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = 5
    mode: str = Field("local", pattern="^(local|heuristic|openai)$")
    return_debug: bool = False


class CheckResponse(BaseModel):
    input_text: str
    spans: List[SpanResult]
    summary: dict
    debug: Optional[dict] = None
