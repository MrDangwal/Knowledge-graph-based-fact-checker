from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import settings
from app.core.chunking import chunk_text
from app.core.graph import extract_entities
from app.core.retrieval import IndexData, build_tfidf, EmbeddingBackend
from app.kb.storage import KBStorage

logger = logging.getLogger(__name__)

CHUNKS_FILE = "kb_chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"
ENTITY_INDEX_FILE = "entity_index.json"

_cached_index: Optional[IndexData] = None
_cached_model: Optional[str] = None


class IndexManager:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or settings.data_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> IndexData:
        storage = KBStorage(str(self.base_dir))
        files = storage.list_files()
        chunks = []
        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(
                chunk_text(
                    text,
                    source_file=file_path.name,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        source_files = [chunk.source_file for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        chunk_entities = [extract_entities(text) for text in texts]

        embeddings = np.zeros((0, 0), dtype=np.float32)
        if texts:
            backend = EmbeddingBackend(settings.embedding_model)
            embeddings = backend.embed(texts)

        tfidf_vectorizer, tfidf_matrix = build_tfidf(texts)

        self._persist_chunks(chunks)
        self._persist_entity_index(chunk_ids, chunk_entities)
        np.save(self.base_dir / EMBEDDINGS_FILE, embeddings)
        meta = {
            "embedding_model": settings.embedding_model,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunk_count": len(chunks),
        }
        (self.base_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        index = IndexData(
            chunk_ids=chunk_ids,
            source_files=source_files,
            texts=texts,
            embeddings=embeddings,
            embedding_model=settings.embedding_model,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            chunk_entities=chunk_entities,
        )
        self._cache(index)
        return index

    def load(self) -> Optional[IndexData]:
        chunks_path = self.base_dir / CHUNKS_FILE
        embeddings_path = self.base_dir / EMBEDDINGS_FILE
        meta_path = self.base_dir / META_FILE
        if not (chunks_path.exists() and embeddings_path.exists() and meta_path.exists()):
            return None

        with chunks_path.open("r", encoding="utf-8") as handle:
            chunk_ids = []
            source_files = []
            texts = []
            for line in handle:
                record = json.loads(line)
                chunk_ids.append(record["chunk_id"])
                source_files.append(record["source_file"])
                texts.append(record["text"])

        embeddings = np.load(embeddings_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embedding_model = meta.get("embedding_model", settings.embedding_model)
        tfidf_vectorizer, tfidf_matrix = build_tfidf(texts)
        chunk_entities = self._load_entity_index(chunk_ids, texts)

        index = IndexData(
            chunk_ids=chunk_ids,
            source_files=source_files,
            texts=texts,
            embeddings=embeddings,
            embedding_model=embedding_model,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            chunk_entities=chunk_entities,
        )
        self._cache(index)
        return index

    def status(self) -> dict:
        meta_path = self.base_dir / META_FILE
        if not meta_path.exists():
            return {
                "file_count": len(KBStorage(str(self.base_dir)).list_files()),
                "chunk_count": 0,
                "last_indexed": None,
                "embedding_model": None,
            }
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "file_count": len(KBStorage(str(self.base_dir)).list_files()),
            "chunk_count": meta.get("chunk_count", 0),
            "last_indexed": meta.get("created_at"),
            "embedding_model": meta.get("embedding_model"),
        }

    def clear_cache(self) -> None:
        global _cached_index
        _cached_index = None

    def _persist_chunks(self, chunks: List) -> None:
        chunks_path = self.base_dir / CHUNKS_FILE
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_file": chunk.source_file,
                            "text": chunk.text,
                        }
                    )
                    + "\n"
                )

    def _persist_entity_index(self, chunk_ids: List[str], chunk_entities: List[List[str]]) -> None:
        entity_path = self.base_dir / ENTITY_INDEX_FILE
        mapping = {chunk_id: entities for chunk_id, entities in zip(chunk_ids, chunk_entities)}
        entity_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    def _load_entity_index(self, chunk_ids: List[str], texts: List[str]) -> List[List[str]]:
        entity_path = self.base_dir / ENTITY_INDEX_FILE
        if not entity_path.exists():
            return [extract_entities(text) for text in texts]
        data = json.loads(entity_path.read_text(encoding="utf-8"))
        return [data.get(chunk_id, []) for chunk_id in chunk_ids]

    def _cache(self, index: IndexData) -> None:
        global _cached_index, _cached_model
        _cached_index = index
        _cached_model = index.embedding_model


def get_index() -> Optional[IndexData]:
    global _cached_index
    if _cached_index is not None:
        return _cached_index
    manager = IndexManager()
    return manager.load()
