import numpy as np
from fastapi.testclient import TestClient

import app.core.retrieval as retrieval
from app.config import settings
from app.kb.index import IndexManager
from app.kb.storage import KBStorage
from app.main import app


class DummyBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_api_check_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "EmbeddingBackend", DummyBackend)
    settings.data_dir = str(tmp_path)
    storage = KBStorage(str(tmp_path))
    storage.save_files([("kb.txt", b"Paris is the capital of France.")])
    manager = IndexManager(str(tmp_path))
    manager.build()

    client = TestClient(app)
    resp = client.post(
        "/api/check",
        json={"text": "Paris is the capital of France.", "top_k": 3, "mode": "heuristic"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "spans" in data
    assert len(data["spans"]) == 1
