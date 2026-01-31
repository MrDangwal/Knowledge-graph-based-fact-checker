import numpy as np

import app.core.retrieval as retrieval
from app.core.retrieval import IndexData, build_tfidf, retrieve


class DummyBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts):
        vectors = []
        for text in texts:
            vectors.append([float(len(text))])
        return np.asarray(vectors, dtype=np.float32)


def test_retrieve_top_k(monkeypatch):
    monkeypatch.setattr(retrieval, "EmbeddingBackend", DummyBackend)
    texts = ["alpha beta", "gamma delta"]
    embeddings = np.asarray([[1.0], [2.0]], dtype=np.float32)
    vectorizer, tfidf = build_tfidf(texts)
    index = IndexData(
        chunk_ids=["a", "b"],
        source_files=["a.txt", "b.txt"],
        texts=texts,
        embeddings=embeddings,
        embedding_model="dummy",
        tfidf_vectorizer=vectorizer,
        tfidf_matrix=tfidf,
        chunk_entities=[[], []],
    )
    results = retrieve("alpha", index, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id in {"a", "b"}
