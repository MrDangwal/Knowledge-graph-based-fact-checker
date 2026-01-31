from __future__ import annotations

from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: str = "./data"
    max_input_chars: int = 20000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k_default: int = 5
    nli_model: str = "facebook/bart-large-mnli"
    min_retrieval_score: float = 0.35


settings = Settings()
