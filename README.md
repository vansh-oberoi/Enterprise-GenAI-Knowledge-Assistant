# Enterprise GenAI Knowledge Assistant

A focused MVP for enterprise document Q&A using retrieval-augmented generation (RAG), source citations, a FastAPI backend, Streamlit UI, baseline evaluation, guardrails, and observability.

## Features
- Ingest PDF, DOCX, TXT, Markdown, and CSV files.
- Preserve source metadata such as filename, PDF page, and CSV row.
- Chunk text with configurable overlap.
- Generate embeddings with Sentence Transformers and store vectors in ChromaDB.
- Retrieve relevant passages and generate evidence-grounded answers with source labels.
- Provide a FastAPI API, lightweight Streamlit UI, basic metrics, and structured logs.
- Run a baseline retrieval evaluation against a small JSONL test set.

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in `.env` to an approved OpenAI-compatible chat API provider. Never commit `.env` or secrets. The embedding model is downloaded on first use.

Run the API in one terminal:

```bash
uvicorn app.api:app --reload
```

Run the UI in another terminal:

```bash
streamlit run app/ui.py
```

Open the Streamlit URL, upload documents, click **Index documents**, then ask questions. The sample policy at `data/sample_policy.txt` can be used for a quick demo.

## API
- `GET /health` — health check.
- `GET /metrics` — basic in-process counters and average query latency.
- `POST /ingest` — multipart upload using one or more `files` fields.
- `POST /retrieve` — returns ranked evidence without calling the LLM; useful for retrieval evaluation.
- `POST /query` — JSON body such as `{"question":"What is the policy?","top_k":5}`; returns a generated answer and retrieved sources.

FastAPI's interactive API docs are available at `/docs` while the server is running.

## Tests and evaluation

Run unit tests:

```bash
pytest -q
```

For retrieval evaluation, start the API and index the documents referenced in `eval/questions.jsonl` through the UI or API. Then run:

```bash
python -m eval.run_eval
```

Update `eval/questions.jsonl` to reflect the documents used in your evaluation. The runner calls `/retrieve` (no LLM credentials required) and reports source-level Hit@k, Recall@k, and MRR@k. These metrics depend on the expected-source labels in the small benchmark; they do not prove answer faithfulness or citation correctness.

## Architecture and limitations
See [`docs/architecture.md`](docs/architecture.md). This is an MVP, not production-ready enterprise software. Before real deployment, add authentication, document-level access control, tenant isolation, rate limiting, durable metrics, OCR where needed, stronger adversarial tests, and an approved data-handling policy. CSV is indexed row-wise; use SQL/tool-based execution for exact structured analytics.
