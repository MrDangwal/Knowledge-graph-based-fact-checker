from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.text_utils import SentenceSpan
from app.core.verification import VerificationResult
from app.core.retrieval import RetrievedChunk


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    confidence: float
    claim: str
    evidence: List[RetrievedChunk]


def build_spans(
    sentence_spans: List[SentenceSpan],
    results: List[VerificationResult],
    evidence: List[List[RetrievedChunk]],
) -> List[Span]:
    spans: List[Span] = []
    for sentence, result, ev in zip(sentence_spans, results, evidence):
        spans.append(
            Span(
                start=sentence.start,
                end=sentence.end,
                label=result.label,
                confidence=result.confidence,
                claim=sentence.text,
                evidence=ev,
            )
        )
    return spans
