# Architecture and design notes

## Request path
1. The ingestion endpoint validates file extension and size, extracts text, and preserves provenance (filename plus PDF page or CSV row where available).
2. Text is chunked with overlap and embedded using a Sentence Transformers model.
3. Chunks and metadata are persisted in ChromaDB.
4. Query text is embedded and top-k chunks are retrieved using cosine distance.
5. The generation layer sends retrieved evidence to a configurable OpenAI-compatible chat API.
6. The response includes source labels and evidence excerpts; application logs capture failures and latency.

## Trust boundaries
- Uploaded files are untrusted input.
- Retrieved passages are untrusted data, not instructions.
- The system prompt instructs the model to refuse unsupported answers and cite supplied evidence.
- Prompt text alone is not a complete prompt-injection defence. Production deployment requires adversarial testing, access controls, provider review, and data-handling policies.

## Current MVP limitations
- Metrics are in-process and reset when the server restarts.
- No authentication or tenant isolation yet.
- No OCR for scanned PDFs.
- CSV is indexed row-wise; vector retrieval is not a substitute for exact SQL analytics.
- Citation labels identify retrieved chunks, but citation correctness and faithfulness need automated and human evaluation.
- No production-grade authorization, rate limiting, or persistent observability backend.
