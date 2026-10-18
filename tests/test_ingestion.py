import pytest
from app.ingestion import extract_file, chunk_units

def test_text_ingestion_and_chunking():
    units = extract_file('policy.txt', b'Annual leave policy. ' * 30)
    assert len(units) == 1
    chunks = chunk_units('policy.txt', units, chunk_size=150, overlap=20)
    assert len(chunks) > 1
    assert all(chunk['metadata']['source'] == 'policy.txt' for chunk in chunks)

def test_unsupported_file_type():
    with pytest.raises(ValueError):
        extract_file('archive.exe', b'not allowed')

def test_invalid_chunk_overlap():
    units = extract_file('policy.txt', b'Some policy text')
    with pytest.raises(ValueError):
        chunk_units('policy.txt', units, chunk_size=100, overlap=100)
