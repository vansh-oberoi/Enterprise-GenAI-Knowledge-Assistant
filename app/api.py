import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from .config import LOG_LEVEL
from .service import ingest_documents, retrieve_evidence, answer_question, METRICS

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI(title='Enterprise GenAI Knowledge Assistant', version='0.1.0', description='Document-grounded Q&A with citations and basic observability.')

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/metrics')
def metrics():
    count = METRICS['queries']
    return {**METRICS, 'average_query_latency_ms': METRICS['total_latency_ms'] / count if count else 0}

@app.post('/ingest')
async def ingest(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail='Upload at least one file.')
    payload = []
    for file in files:
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f'{file.filename} exceeds 20 MB.')
        payload.append((file.filename or 'uploaded_file', content))
    try:
        return ingest_documents(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('ingestion_failed')
        raise HTTPException(status_code=500, detail='Document ingestion failed.') from exc

@app.post('/retrieve')
def retrieve(request: QueryRequest):
    """Return ranked evidence without calling the configured LLM provider."""
    try:
        hits = retrieve_evidence(request.question, request.top_k)
        sources = []
        for index, hit in enumerate(hits, start=1):
            metadata = hit.get('metadata', {})
            sources.append({
                'citation': f'S{index}', 'source': metadata.get('source', 'unknown'),
                'page': metadata.get('page'), 'row': metadata.get('row'),
                'distance': hit.get('distance'), 'excerpt': hit['text'][:500]
            })
        return {'sources': sources}
    except Exception as exc:
        logger.exception('retrieval_failed')
        raise HTTPException(status_code=500, detail='Retrieval failed. Check server logs.') from exc


@app.post('/query')
def query(request: QueryRequest):
    try:
        return answer_question(request.question, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('query_failed')
        raise HTTPException(status_code=500, detail='Query failed. Check server logs.') from exc
