import logging
import time
from .config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from .ingestion import extract_file, chunk_units
from .retrieval import get_store
from .generation import generate_answer

logger = logging.getLogger(__name__)
METRICS = {'queries': 0, 'errors': 0, 'total_latency_ms': 0.0, 'ingested_chunks': 0}

def ingest_documents(files: list[tuple[str, bytes]]) -> dict:
    store = get_store()
    file_results = []
    total = 0
    for filename, content in files:
        units = extract_file(filename, content)
        chunks = chunk_units(filename, units, CHUNK_SIZE, CHUNK_OVERLAP)
        indexed = store.add_chunks(chunks)
        total += indexed
        file_results.append({'filename': filename, 'units': len(units), 'chunks_indexed': indexed})
    METRICS['ingested_chunks'] += total
    logger.info('ingestion_complete files=%d chunks=%d', len(files), total)
    return {'files': file_results, 'total_chunks_indexed': total}

def retrieve_evidence(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return ranked retrieved passages without invoking the language model."""
    return get_store().search(question, top_k=top_k)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    started = time.perf_counter()
    METRICS['queries'] += 1
    try:
        hits = retrieve_evidence(question, top_k=top_k)
        answer = generate_answer(question, hits)
        sources = []
        for index, hit in enumerate(hits, start=1):
            metadata = hit.get('metadata', {})
            sources.append({
                'citation': f'S{index}', 'source': metadata.get('source', 'unknown'),
                'page': metadata.get('page'), 'row': metadata.get('row'),
                'distance': hit.get('distance'), 'excerpt': hit['text'][:500]
            })
        return {'answer': answer, 'sources': sources}
    except Exception:
        METRICS['errors'] += 1
        logger.exception('query_failed')
        raise
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        METRICS['total_latency_ms'] += elapsed
        logger.info('query_finished latency_ms=%.2f', elapsed)
