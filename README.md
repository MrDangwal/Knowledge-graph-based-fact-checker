# Fact Checker App

Local-first fact checker with an optional OpenAI-assisted mode. Upload `.txt` files as a knowledge base, build an index, and verify claims against it with evidence snippets.

## Features
- Local-first workflow with file uploads and indexing
- Evidence-backed claim checking against your knowledge base
- Hybrid retrieval: semantic embeddings + TF-IDF + entity overlap
- Multiple verification modes: heuristic, local NLI model, or OpenAI
- UI served from FastAPI with server-rendered templates

## Requirements
- macOS
- Python 3.10

## Tech Stack
- API server: FastAPI + Uvicorn
- UI: Jinja2 templates + static assets served by FastAPI
- Retrieval: sentence-transformers embeddings + scikit-learn TF-IDF
- Verification: transformers NLI pipeline (BART MNLI) + heuristics
- Optional LLM: OpenAI Chat Completions via httpx
- Storage: local filesystem under `./data/`

## Quickstart
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit: http://127.0.0.1:8000

## Configuration
Optional OpenAI mode:
```bash
export OPENAI_API_KEY="your_api_key_here"
```
Select “High accuracy” in the UI.

## How It Works
1. Upload knowledge base files (`.txt` or a `.zip` of `.txt` files).
2. Build the index:
   - Text is chunked with overlap.
   - Embeddings are generated with `sentence-transformers`.
   - A TF-IDF matrix is built for keyword matching.
   - Lightweight entity extraction is stored for overlap boosting.
3. Check claims:
   - Input is split into sentences.
   - For each sentence, top-k evidence chunks are retrieved with a hybrid score.
   - A verdict is produced via:
     - Heuristic rules, or
     - Local NLI model (`facebook/bart-large-mnli`), or
     - OpenAI mode if enabled.
4. Results return labeled spans (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO) with evidence snippets.

## Project Data
- Uploaded files and the index are stored in `./data/`.

## Default Settings
Key defaults (see `app/config.py`):
- `data_dir`: `./data`
- `max_input_chars`: `20000`
- `embedding_model`: `sentence-transformers/all-MiniLM-L6-v2`
- `chunk_size`: `500`
- `chunk_overlap`: `80`
- `top_k_default`: `5`
- `nli_model`: `facebook/bart-large-mnli`
- `min_retrieval_score`: `0.35`

## API Endpoints
- `GET /api/health`
- `POST /api/kb/upload`
- `GET /api/kb/list`
- `DELETE /api/kb/clear`
- `POST /api/kb/rebuild`
- `GET /api/kb/status`
- `POST /api/check`

## Make Targets
```bash
make setup
make run
make test
```
