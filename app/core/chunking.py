from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_file: str
    text: str


def chunk_text(text: str, source_file: str, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    if chunk_size <= 0:
        return chunks

    start = 0
    idx = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunk_id = f"{source_file}::{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, source_file=source_file, text=chunk_text))
            idx += 1
        if end == length:
            break
        start = max(end - overlap, 0)

    return chunks
