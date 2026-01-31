from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.models import CheckRequest, CheckResponse, EvidenceItem, SpanResult
from app.core.text_utils import split_claims_with_offsets
from app.core.retrieval import retrieve, RetrievedChunk
from app.core.verification import (
    LABEL_CONTRADICTED,
    LABEL_NEI,
    LABEL_SUPPORTED,
    VerificationResult,
    verify_with_heuristics,
    verify_with_local_nli,
)
from app.core.highlight import build_spans
from app.kb.index import get_index, IndexManager
from app.llm.openai_client import OpenAIClient

router = APIRouter()


@router.post("/check", response_model=CheckResponse)
async def check(request: CheckRequest) -> CheckResponse:
    text = request.text.strip()
    if len(text) > settings.max_input_chars:
        raise HTTPException(status_code=400, detail="Input too long")

    index = get_index()
    if index is None:
        manager = IndexManager()
        index = manager.load()
    if index is None or not index.texts:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload files and rebuild index.")

    sentences = split_claims_with_offsets(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found in input")

    evidence_sets: List[List[RetrievedChunk]] = []
    results = []
    openai_client = OpenAIClient()

    for sentence in sentences:
        retrieved = retrieve(sentence.text, index, request.top_k)
        evidence_sets.append(retrieved)
        if not retrieved or retrieved[0].score < settings.min_retrieval_score:
            results.append(VerificationResult(label=LABEL_NEI, confidence=0.2))
            continue
        if request.mode == "openai" and openai_client.enabled():
            evidence_text = "\n\n".join([f"[{r.source_file}] {r.text}" for r in retrieved])
            verdict = openai_client.judge_claim(sentence.text, evidence_text)
            if verdict:
                label = str(verdict.get("label", LABEL_NEI)).upper()
                if label not in {LABEL_SUPPORTED, LABEL_CONTRADICTED, LABEL_NEI}:
                    label = LABEL_NEI
                confidence = float(verdict.get("confidence", 0.5))
                results.append(VerificationResult(label=label, confidence=confidence))
                continue
        if request.mode == "heuristic":
            result = verify_with_heuristics(sentence.text, retrieved)
        else:
            result = verify_with_local_nli(sentence.text, retrieved, settings.nli_model)
        results.append(result)

    spans = build_spans(sentences, results, evidence_sets)
    span_results = [
        SpanResult(
            start=span.start,
            end=span.end,
            label=span.label,
            confidence=span.confidence,
            claim=span.claim,
            evidence=[
                EvidenceItem(
                    source_file=ev.source_file,
                    chunk_id=ev.chunk_id,
                    text=ev.text,
                    score=ev.score,
                )
                for ev in span.evidence
            ],
        )
        for span in spans
    ]

    summary = {
        "supported": sum(1 for s in span_results if s.label == LABEL_SUPPORTED),
        "contradicted": sum(1 for s in span_results if s.label == LABEL_CONTRADICTED),
        "nei": sum(1 for s in span_results if s.label == LABEL_NEI),
    }

    debug = None
    if request.return_debug:
        debug = {
            "sentences": [s.text for s in sentences],
            "retrieved": [
                [
                    {
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                        "semantic_score": r.semantic_score,
                        "keyword_score": r.keyword_score,
                    }
                    for r in retrieved
                ]
                for retrieved in evidence_sets
            ],
        }

    return CheckResponse(input_text=text, spans=span_results, summary=summary, debug=debug)
