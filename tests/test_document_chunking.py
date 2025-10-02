# filepath: tests/test_document_chunking.py
from data_processing import chunk_document, Chunk


def test_chunk_document_tokens():
    text = " ".join(["word"] * 120)  # 120 tokens
    chunks = chunk_document(
        text,
        strategy="tokens",
        chunk_size=50,
        chunk_overlap=10,
        source_id="doc1",
        source_path="/path/to/doc1",
        extra_meta={"file_type": ".html"},
    )

    assert isinstance(chunks, list)
    assert len(chunks) == 3  # 50 tokens, step 40 => indices: 0-49, 40-89, 80-119
    assert all(isinstance(c, Chunk) for c in chunks)
    # Basic metadata presence
    for i, c in enumerate(chunks):
        assert c.metadata.get("source_id") == "doc1"
        assert c.metadata.get("source_path") == "/path/to/doc1"
        assert c.metadata.get("strategy") == "tokens"
        assert c.metadata.get("chunk_index") == i
        assert c.metadata.get("start_token") is not None
        assert c.metadata.get("end_token") is not None


def test_chunk_document_empty():
    assert chunk_document("") == []

