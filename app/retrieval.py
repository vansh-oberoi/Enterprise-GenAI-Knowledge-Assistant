from .config import CHROMA_DIR, EMBEDDING_MODEL

COLLECTION_NAME = 'enterprise_knowledge'

class KnowledgeStore:
    def __init__(self):
        # Load heavyweight retrieval dependencies only when a store is instantiated.
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={'hnsw:space': 'cosine'}
        )
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def add_chunks(self, chunks: list[dict], batch_size: int = 64) -> int:
        total = 0
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            embeddings = self.model.encode(
                [item['text'] for item in batch], normalize_embeddings=True
            ).tolist()
            ids = [
                f"{item['metadata'].get('source', 'doc')}:{item['metadata'].get('chunk_start', 0)}:{offset+i}"
                for i, item in enumerate(batch)
            ]
            metadatas = [
                {str(k): v for k, v in item['metadata'].items()
                 if v is not None and isinstance(v, (str, int, float, bool))}
                for item in batch
            ]
            self.collection.upsert(
                ids=ids, documents=[item['text'] for item in batch],
                embeddings=embeddings, metadatas=metadatas
            )
            total += len(batch)
        return total

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        count = self.collection.count()
        if not question.strip() or count == 0:
            return []
        vector = self.model.encode([question], normalize_embeddings=True).tolist()[0]
        result = self.collection.query(
            query_embeddings=[vector], n_results=min(max(1, top_k), count),
            include=['documents', 'metadatas', 'distances']
        )
        documents = result.get('documents', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]
        distances = result.get('distances', [[]])[0]
        return [
            {'text': doc, 'metadata': metadatas[i] or {},
             'distance': distances[i] if i < len(distances) else None}
            for i, doc in enumerate(documents)
        ]

_store = None
def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store
