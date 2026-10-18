# Enterprise GenAI Knowledge Assistant

An enterprise-oriented retrieval-augmented generation (RAG) assistant for document-grounded Q&A, citations, evaluation, guardrails, and observability.

## Project goals
- Ingest structured and unstructured documents.
- Index document chunks using embeddings and metadata.
- Retrieve evidence and generate grounded answers with citations.
- Expose a FastAPI service and lightweight Streamlit UI.
- Evaluate retrieval/answer quality and log latency, token usage, and failures.

## Development
Implementation is developed in small, testable milestones. See `docs/architecture.md` as the architecture notes are added.
