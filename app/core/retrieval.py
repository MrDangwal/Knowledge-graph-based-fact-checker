from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.graph import extract_entities
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional import failure
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_file: str
    text: str
    score: float
    semantic_score: float
    keyword_score: float


@dataclass
class IndexData:
    chunk_ids: List[str]
    source_files: List[str]
    texts: List[str]
    embeddings: np.ndarray
    embedding_model: str

    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: np.ndarray
    chunk_entities: List[List[str]]


_backend_cache: dict[str, "EmbeddingBackend"] = {}


class EmbeddingBackend:
    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def build_tfidf(texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    if not texts:
        vectorizer.fit(["placeholdertoken"])
        matrix = np.zeros((0, len(vectorizer.get_feature_names_out())), dtype=np.float32)
        return vectorizer, matrix
    matrix = vectorizer.fit_transform(texts).toarray()
    return vectorizer, matrix


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)


def retrieve(query: str, index: IndexData, top_k: int) -> List[RetrievedChunk]:
    if index.embeddings.size == 0:
        return []

    claim_entities = set(extract_entities(query))
    backend = _backend_cache.get(index.embedding_model)
    if backend is None:
        backend = EmbeddingBackend(index.embedding_model)
        _backend_cache[index.embedding_model] = backend
    query_vec = backend.embed([query])
    semantic_scores = cosine_sim(query_vec, index.embeddings)[0]

    query_tfidf = index.tfidf_vectorizer.transform([query]).toarray()
    keyword_scores = cosine_sim(query_tfidf, index.tfidf_matrix)[0]

    scores = 0.75 * semantic_scores + 0.25 * keyword_scores
    if claim_entities:
        entity_scores = []
        for entities in index.chunk_entities:
            if not entities:
                entity_scores.append(0.0)
                continue
            overlap = len(claim_entities.intersection(entities))
            entity_scores.append(overlap / max(len(claim_entities), 1))
        scores = scores + 0.1 * np.asarray(entity_scores, dtype=np.float32)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[RetrievedChunk] = []
    for idx in top_indices:
        results.append(
            RetrievedChunk(
                chunk_id=index.chunk_ids[idx],
                source_file=index.source_files[idx],
                text=index.texts[idx],
                score=float(scores[idx]),
                semantic_score=float(semantic_scores[idx]),
                keyword_score=float(keyword_scores[idx]),
            )
        )

    return results
