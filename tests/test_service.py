import pytest

from app import generation, service


class FakeStore:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.added_chunks = []

    def add_chunks(self, chunks):
        self.added_chunks.extend(chunks)
        return len(chunks)

    def search(self, question, top_k=5):
        return self.hits[:top_k]


@pytest.fixture(autouse=True)
def reset_metrics():
    original = dict(service.METRICS)
    service.METRICS.update({
        "queries": 0,
        "errors": 0,
        "total_latency_ms": 0.0,
        "ingested_chunks": 0,
    })
    yield
    service.METRICS.update(original)


def test_ingestion_indexes_chunks_and_preserves_source(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(service, "get_store", lambda: store)
    payload = ("Company policy. " * 120).encode("utf-8")

    result = service.ingest_documents([("policy.txt", payload)])

    assert result["total_chunks_indexed"] > 1
    assert result["files"][0]["filename"] == "policy.txt"
    assert all(chunk["metadata"]["source"] == "policy.txt" for chunk in store.added_chunks)
    assert service.METRICS["ingested_chunks"] == result["total_chunks_indexed"]


def test_answer_question_returns_answer_and_source_provenance(monkeypatch):
    hits = [{
        "text": "Annual leave is approved by the manager.",
        "metadata": {"source": "leave_policy.pdf", "page": 3},
        "distance": 0.12,
    }]
    monkeypatch.setattr(service, "get_store", lambda: FakeStore(hits))
    monkeypatch.setattr(service, "generate_answer", lambda question, evidence: "Manager approval is required [S1].")

    result = service.answer_question("Who approves annual leave?", top_k=1)

    assert result["answer"] == "Manager approval is required [S1]."
    assert result["sources"] == [{
        "citation": "S1",
        "source": "leave_policy.pdf",
        "page": 3,
        "row": None,
        "distance": 0.12,
        "excerpt": "Annual leave is approved by the manager.",
    }]
    assert service.METRICS["queries"] == 1
    assert service.METRICS["errors"] == 0
    assert service.METRICS["total_latency_ms"] >= 0


def test_empty_retrieval_refuses_without_llm_credentials(monkeypatch):
    monkeypatch.setattr(generation, "LLM_BASE_URL", "")
    monkeypatch.setattr(generation, "LLM_API_KEY", "")
    monkeypatch.setattr(generation, "LLM_MODEL", "")
    monkeypatch.setattr(service, "get_store", lambda: FakeStore())

    result = service.answer_question("What is the leave allowance?")

    assert "can't answer reliably" in result["answer"]
    assert result["sources"] == []
    assert service.METRICS["queries"] == 1
    assert service.METRICS["errors"] == 0

def test_retrieve_evidence_does_not_invoke_generation(monkeypatch):
    hits = [{"text": "Security policy evidence.", "metadata": {"source": "security.md"}, "distance": 0.2}]
    monkeypatch.setattr(service, "get_store", lambda: FakeStore(hits))
    monkeypatch.setattr(
        service, "generate_answer",
        lambda *args: pytest.fail("retrieval-only path must not invoke generation"),
    )

    result = service.retrieve_evidence("What is the security policy?", top_k=1)

    assert result == hits

