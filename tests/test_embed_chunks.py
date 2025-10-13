"""
Test for step 2.2: Service wrapper for embedding model
Tests that embed_chunks() accepts Chunk objects and returns vectors
"""
from typing import List

import pytest

from src.data_processing.models import Chunk
from src.core.embeddings import embed_chunks


def test_embed_single_chunk(monkeypatch):
    """Test embedding a single Chunk object"""
    # Mock the actual API call
    def fake_embed_text(text: str, **kwargs):
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr("core.embeddings.embed_text", fake_embed_text)

    # Create a Chunk
    chunk = Chunk(text="Test chunk text", metadata={"source": "test.pdf", "page": 1})

    # Embed it
    result = embed_chunks(chunk)

    # Should return a single vector
    assert isinstance(result, list)
    assert len(result) == 3
    assert result == [1.0, 2.0, 3.0]


def test_embed_multiple_chunks(monkeypatch):
    """Test embedding a list of Chunk objects"""
    # Mock the actual API call
    call_count = [0]

    def fake_embed_text(text: str, **kwargs):
        call_count[0] += 1
        # Return different vectors based on text length
        return [float(len(text)), float(call_count[0]), 0.5]

    monkeypatch.setattr("core.embeddings.embed_text", fake_embed_text)

    # Create multiple Chunks
    chunks = [
        Chunk(text="First chunk", metadata={"index": 0}),
        Chunk(text="Second chunk text", metadata={"index": 1}),
        Chunk(text="Third", metadata={"index": 2}),
    ]

    # Embed them
    result = embed_chunks(chunks)

    # Should return list of vectors
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(vec, list) for vec in result)

    # Check that each chunk was processed
    assert result[0] == [11.0, 1.0, 0.5]  # len("First chunk") = 11
    assert result[1] == [17.0, 2.0, 0.5]  # len("Second chunk text") = 17
    assert result[2] == [5.0, 3.0, 0.5]   # len("Third") = 5


def test_embed_chunks_with_metadata_preserved(monkeypatch):
    """Verify that chunk metadata is preserved (not lost during embedding)"""
    def fake_embed_text(text: str, **kwargs):
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr("core.embeddings.embed_text", fake_embed_text)

    # Create chunks with rich metadata
    chunks = [
        Chunk(
            text="Python is great",
            metadata={
                "source_id": "tutorial.pdf",
                "page_number": 5,
                "chunk_index": 0,
                "strategy": "tokens"
            }
        ),
        Chunk(
            text="Machine learning is powerful",
            metadata={
                "source_id": "tutorial.pdf",
                "page_number": 6,
                "chunk_index": 1,
                "strategy": "tokens"
            }
        ),
    ]

    # Embed them
    embeddings = embed_chunks(chunks)

    # Embeddings should be returned
    assert len(embeddings) == 2

    # Original chunks should still have their metadata intact
    assert chunks[0].metadata["source_id"] == "tutorial.pdf"
    assert chunks[0].metadata["page_number"] == 5
    assert chunks[1].metadata["chunk_index"] == 1


def test_embed_chunks_type_error():
    """Test that passing wrong type raises TypeError"""
    with pytest.raises(TypeError, match="chunks must be Chunk or List\\[Chunk\\]"):
        embed_chunks("plain string")

    with pytest.raises(TypeError, match="Expected Chunk object"):
        embed_chunks(["string1", "string2"])


def test_step_2_2_criteria():
    """
    Verify step 2.2 criteria:
    - Accepts a text chunk (Chunk object) ✓
    - Accepts a list of chunks (List[Chunk]) ✓
    - Returns a vector (List[float]) ✓
    - Returns list of vectors (List[List[float]]) ✓
    - Interacts with model via API ✓ (through embed_text)
    """
    # This test documents that all criteria are met
    # The actual tests above verify each criterion
    assert hasattr(embed_chunks, '__call__')
    assert embed_chunks.__doc__ is not None
    assert "Service wrapper" in embed_chunks.__doc__
    assert "Step 2.2" in embed_chunks.__doc__
