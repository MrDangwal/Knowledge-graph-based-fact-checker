from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_SENTENCE_END_RE = re.compile(r"([.!?]+)(\s+|$)")
_CLAUSE_SPLIT_RE = re.compile(r"(;|,(?=\s+(?:and|but|however|yet|while)\b))", re.IGNORECASE)


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences_with_offsets(text: str) -> List[SentenceSpan]:
    spans: List[SentenceSpan] = []
    if not text:
        return spans

    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        segment = text[start:end].strip()
        if segment:
            leading_ws = len(text[start:end]) - len(text[start:end].lstrip())
            seg_start = start + leading_ws
            seg_end = seg_start + len(segment)
            spans.append(SentenceSpan(text=segment, start=seg_start, end=seg_end))
        start = end

    if start < len(text):
        tail = text[start:].strip()
        if tail:
            leading_ws = len(text[start:]) - len(text[start:].lstrip())
            seg_start = start + leading_ws
            seg_end = seg_start + len(tail)
            spans.append(SentenceSpan(text=tail, start=seg_start, end=seg_end))

    return spans


def split_claims_with_offsets(text: str) -> List[SentenceSpan]:
    claims: List[SentenceSpan] = []
    for sentence in split_sentences_with_offsets(text):
        clauses = _split_clauses(sentence)
        claims.extend(clauses)
    return claims


def _split_clauses(sentence: SentenceSpan) -> List[SentenceSpan]:
    spans: List[SentenceSpan] = []
    segment_start = 0
    text = sentence.text
    for match in _CLAUSE_SPLIT_RE.finditer(text):
        segment = text[segment_start:match.start()]
        segment = segment.strip()
        if segment:
            raw_segment = text[segment_start:match.start()]
            leading_ws = len(raw_segment) - len(raw_segment.lstrip())
            seg_start = sentence.start + segment_start + leading_ws
            seg_end = seg_start + len(segment)
            spans.append(SentenceSpan(text=segment, start=seg_start, end=seg_end))
        segment_start = match.end()

    tail = text[segment_start:].strip()
    if tail:
        raw_tail = text[segment_start:]
        leading_ws = len(raw_tail) - len(raw_tail.lstrip())
        seg_start = sentence.start + segment_start + leading_ws
        seg_end = seg_start + len(tail)
        spans.append(SentenceSpan(text=tail, start=seg_start, end=seg_end))

    return spans if spans else [sentence]
