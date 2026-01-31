from __future__ import annotations

import re
from typing import List


_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,})\b")
_STOPWORDS = {
    "The", "A", "An", "And", "Or", "Of", "In", "On", "At", "To", "By", "For",
    "Is", "Was", "Are", "Were", "Be", "Been", "Being", "With", "As", "From",
}


def extract_entities(text: str) -> List[str]:
    entities = set()
    for match in _ENTITY_RE.findall(text):
        if match in _STOPWORDS:
            continue
        entities.add(match)
    for match in _ACRONYM_RE.findall(text):
        if match in _STOPWORDS:
            continue
        entities.add(match)
    return sorted(entities)
