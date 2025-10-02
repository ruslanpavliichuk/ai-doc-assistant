import pytest

from data_processing.chunker import TokenChunker

transformers = pytest.importorskip(
    "transformers",
    reason="transformers is required for hf_tokens chunking tests"
)


def test_hf_token_chunking_basic():
    text = "This is a simple test string to validate Hugging Face tokenizer based chunking with overlap. " * 5

    chunker = TokenChunker(strategy="hf_tokens", model_name="bert-base-uncased")
    chunks = chunker.chunk(text, chunk_size=32, chunk_overlap=8)

    assert isinstance(chunks, list)
    assert len(chunks) >= 2

    # Validate metadata and stepping logic using token indices
    for i, c in enumerate(chunks):
        assert "start_token" in c.metadata and "end_token" in c.metadata
        assert c.metadata["chunk_index"] == i
        assert c.metadata["strategy"] == "hf_tokens"
        assert c.metadata.get("tokenizer") is not None

    # Check stepping: next start should equal previous end - overlap
    for prev, curr in zip(chunks, chunks[1:]):
        prev_end = prev.metadata["end_token"]
        curr_start = curr.metadata["start_token"]
        assert curr_start == prev_end - 8

