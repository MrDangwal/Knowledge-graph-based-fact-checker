from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.core.retrieval import RetrievedChunk
from app.core.text_utils import split_sentences_with_offsets

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None

logger = logging.getLogger(__name__)

LABEL_SUPPORTED = "SUPPORTED"
LABEL_CONTRADICTED = "CONTRADICTED"
LABEL_NEI = "NOT_ENOUGH_INFO"

_negation_re = re.compile(r"\b(not|never|no|none|nobody|nothing)\b", re.IGNORECASE)
_number_re = re.compile(r"\d+(?:\.\d+)?")
_stopwords = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "by", "for",
    "is", "was", "are", "were", "be", "been", "being", "with", "as", "from",
}
_region_re = re.compile(
    r"\b(asia|africa|europe|australia|antarctica|north america|south america)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerificationResult:
    label: str
    confidence: float


_nli_pipeline = None


def _get_nli_pipeline(model_name: str):
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline
    if pipeline is None:
        return None
    try:
        _nli_pipeline = pipeline("text-classification", model=model_name, return_all_scores=True)
        return _nli_pipeline
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load NLI model: %s", exc)
        return None


def _pick_evidence_sentences(text: str, claim: str, max_sentences: int = 2) -> List[str]:
    sentences = split_sentences_with_offsets(text)
    if not sentences:
        return [text]
    scored: List[Tuple[float, str]] = []
    for sent in sentences:
        scored.append((_token_overlap(claim, sent.text), sent.text))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [s for score, s in scored if score > 0.05][:max_sentences]
    return picked if picked else [text]


def _normalize_outputs(outputs) -> List[dict]:
    if isinstance(outputs, dict):
        return [outputs]
    if isinstance(outputs, list):
        if outputs and isinstance(outputs[0], list):
            return outputs[0]
        return outputs
    return []


def _nli_score(nli, premise: str, hypothesis: str) -> Tuple[float, float, float]:
    raw = nli({"text": premise, "text_pair": hypothesis})
    outputs = _normalize_outputs(raw)
    label_scores = {o["label"].lower(): o["score"] for o in outputs if isinstance(o, dict)}
    entail = label_scores.get("entailment", 0.0)
    contra = label_scores.get("contradiction", 0.0)
    neutral = label_scores.get("neutral", 0.0)
    return entail, contra, neutral


def _content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    return {t for t in tokens if t not in _stopwords and len(t) > 2}


def _strong_support(claim: str, evidence_sentence: str) -> bool:
    if _negation_re.search(evidence_sentence):
        return False
    claim_tokens = _content_tokens(claim)
    evidence_tokens = _content_tokens(evidence_sentence)
    if not claim_tokens:
        return False
    return claim_tokens.issubset(evidence_tokens)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_claim(evidence_sentence: str, claim: str) -> bool:
    return _normalize_text(claim) in _normalize_text(evidence_sentence)


def _extract_regions(text: str) -> set[str]:
    return {match.lower() for match in _region_re.findall(text)}


def verify_with_local_nli(claim: str, evidence: List[RetrievedChunk], model_name: str) -> VerificationResult:
    nli = _get_nli_pipeline(model_name)
    if nli is None:
        return verify_with_heuristics(claim, evidence)

    candidate_sentences: List[str] = []
    for chunk in evidence:
        candidate_sentences.extend(_pick_evidence_sentences(chunk.text, claim))
    for sentence in candidate_sentences:
        if _contains_claim(sentence, claim):
            return VerificationResult(label=LABEL_SUPPORTED, confidence=0.9)

    best_label = LABEL_NEI
    best_conf = 0.0
    for chunk in evidence:
        for sentence in _pick_evidence_sentences(chunk.text, claim):
            entail, contra, neutral = _nli_score(nli, sentence, claim)
            if contra > 0.7 and contra > entail + 0.1 and contra > best_conf:
                best_label = LABEL_CONTRADICTED
                best_conf = contra
            elif entail > 0.65 and entail > contra + 0.1 and entail > best_conf:
                best_label = LABEL_SUPPORTED
                best_conf = entail
            elif neutral > best_conf:
                best_label = LABEL_NEI
                best_conf = neutral

    nli_result = VerificationResult(label=best_label, confidence=float(best_conf))
    claim_regions = _extract_regions(claim)
    if nli_result.label == LABEL_SUPPORTED and claim_regions:
        if not any(_extract_regions(s) & claim_regions for s in candidate_sentences):
            nli_result = VerificationResult(label=LABEL_NEI, confidence=0.45)
    heuristic = verify_with_heuristics(claim, evidence)
    if nli_result.label == LABEL_CONTRADICTED and nli_result.confidence < 0.85:
        if heuristic.label == LABEL_SUPPORTED and heuristic.confidence >= 0.5:
            return heuristic
    if nli_result.label == LABEL_NEI:
        if heuristic.label == LABEL_SUPPORTED and heuristic.confidence >= 0.6:
            return heuristic
    if nli_result.label == LABEL_CONTRADICTED:
        for chunk in evidence:
            for sentence in _pick_evidence_sentences(chunk.text, claim):
                if _strong_support(claim, sentence):
                    return VerificationResult(label=LABEL_SUPPORTED, confidence=0.7)
    return nli_result


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
    b_tokens = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def verify_with_heuristics(claim: str, evidence: List[RetrievedChunk]) -> VerificationResult:
    if not evidence:
        return VerificationResult(label=LABEL_NEI, confidence=0.1)

    for chunk in evidence:
        for sentence in _pick_evidence_sentences(chunk.text, claim):
            if _contains_claim(sentence, claim):
                return VerificationResult(label=LABEL_SUPPORTED, confidence=0.85)

    best_overlap = 0.0
    best_score = 0.0
    for chunk in evidence:
        overlap = _token_overlap(claim, chunk.text)
        best_overlap = max(best_overlap, overlap)
        if _strong_support(claim, chunk.text):
            return VerificationResult(label=LABEL_SUPPORTED, confidence=0.65)
        numbers_claim = _number_re.findall(claim)
        numbers_evidence = _number_re.findall(chunk.text)
        if numbers_claim and numbers_evidence and numbers_claim != numbers_evidence and overlap > 0.2:
            if overlap > 0.2:
                return VerificationResult(label=LABEL_CONTRADICTED, confidence=0.6)
        if _negation_re.search(chunk.text) and overlap > 0.2:
            return VerificationResult(label=LABEL_CONTRADICTED, confidence=0.55)
        best_score = max(best_score, chunk.score)

    if best_overlap > 0.3 and best_score > 0.5:
        return VerificationResult(label=LABEL_SUPPORTED, confidence=0.55)

    return VerificationResult(label=LABEL_NEI, confidence=0.35)
