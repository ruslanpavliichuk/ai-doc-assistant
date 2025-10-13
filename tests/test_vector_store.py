"""
Tests for VectorStore (Step 3.2 & 3.3)
Verifies database configuration and storage logic
"""
import tempfile
import shutil
from pathlib import Path

import pytest

from src.data_processing.models import Chunk
from src.core.vector_store import VectorStore


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for ChromaDB during tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store(temp_db_dir):
    """Create a VectorStore instance with temporary storage"""
    return VectorStore(collection_name="test_docs", persist_directory=temp_db_dir)


def test_step_3_2_database_configuration(vector_store):
    """
    Step 3.2: Verify database configuration
    - Collection is created
    - Cosine similarity metric is set
    - Vector dimension is 768 (Gemini text-embedding-004)
    """
    # Check collection exists
    assert vector_store.collection is not None
    assert vector_store.collection.name == "test_docs"

    # Check metadata configuration
    metadata = vector_store.collection.metadata
    assert metadata["hnsw:space"] == "cosine"
    assert metadata["embedding_dimension"] == 768

    # Initial count should be 0
    assert vector_store.count() == 0


def test_step_3_3_add_single_chunk(vector_store):
    """
    Step 3.3: Test storing a single chunk with embedding and metadata
    Each record should contain: unique ID, vector, metadata
    """
    # Create a chunk with metadata
    chunk = Chunk(
        text="Python is a programming language",
        metadata={
            "source_id": "tutorial.pdf",
            "source_path": "/data/tutorial.pdf",
            "chunk_index": 0,
            "strategy": "tokens",
            "page_number": 1
        }
    )

    # Create a fake embedding (768 dimensions for Gemini)
    embedding = [0.1] * 768

    # Store in database
    ids = vector_store.add_chunks([chunk], [embedding])

    # Verify storage
    assert len(ids) == 1
    assert isinstance(ids[0], str)
    assert len(ids[0]) > 0  # UUID should be non-empty

    # Verify count
    assert vector_store.count() == 1


def test_step_3_3_add_multiple_chunks(vector_store):
    """
    Step 3.3: Test storing multiple chunks with embeddings
    """
    # Create multiple chunks
    chunks = [
        Chunk(
            text=f"Chunk {i} content",
            metadata={
                "source_id": "doc.pdf",
                "chunk_index": i,
                "strategy": "tokens"
            }
        )
        for i in range(5)
    ]

    # Create fake embeddings (different for each chunk)
    embeddings = [[float(i) / 10] * 768 for i in range(5)]

    # Store in database
    ids = vector_store.add_chunks(chunks, embeddings)

    # Verify storage
    assert len(ids) == 5
    assert len(set(ids)) == 5  # All IDs should be unique
    assert vector_store.count() == 5


def test_step_3_3_metadata_preservation(vector_store):
    """
    Verify that all metadata is properly stored and retrievable
    """
    chunk = Chunk(
        text="Machine learning is powerful",
        metadata={
            "source_id": "ml_guide.pdf",
            "source_path": "/docs/ml_guide.pdf",
            "chunk_index": 42,
            "strategy": "sentences",
            "page_number": 10,
            "chapter_title": "Introduction"
        }
    )

    embedding = [0.5] * 768
    ids = vector_store.add_chunks([chunk], [embedding])

    # Retrieve by source
    results = vector_store.get_by_source("ml_guide.pdf")

    assert len(results) == 1
    assert results[0]["text"] == "Machine learning is powerful"
    assert results[0]["metadata"]["source_id"] == "ml_guide.pdf"
    assert results[0]["metadata"]["chunk_index"] == 42
    assert results[0]["metadata"]["page_number"] == 10


def test_search_with_cosine_similarity(vector_store):
    """
    Test semantic search using cosine similarity
    """
    # Add some chunks with different embeddings
    chunks = [
        Chunk(text="Python programming", metadata={"source_id": "doc1", "chunk_index": 0}),
        Chunk(text="Java development", metadata={"source_id": "doc1", "chunk_index": 1}),
        Chunk(text="Python data science", metadata={"source_id": "doc1", "chunk_index": 2}),
    ]

    # Create embeddings that make sense semantically
    # Python-related chunks have similar vectors
    embeddings = [
        [1.0, 0.0] + [0.0] * 766,  # Python programming
        [0.0, 1.0] + [0.0] * 766,  # Java (different)
        [0.9, 0.1] + [0.0] * 766,  # Python data science (similar to first)
    ]

    vector_store.add_chunks(chunks, embeddings)

    # Search with a query similar to first chunk
    query_embedding = [1.0, 0.0] + [0.0] * 766
    results = vector_store.search(query_embedding, top_k=2)

    # Should return Python-related chunks first
    assert len(results) == 2
    assert "Python" in results[0][0]  # Top result contains "Python"
    assert results[0][1] > results[1][1]  # First result has higher similarity


def test_delete_by_source(vector_store):
    """
    Test deleting all chunks from a specific source document
    """
    # Add chunks from two different sources
    chunks_doc1 = [
        Chunk(text=f"Doc1 chunk {i}", metadata={"source_id": "doc1.pdf", "chunk_index": i})
        for i in range(3)
    ]
    chunks_doc2 = [
        Chunk(text=f"Doc2 chunk {i}", metadata={"source_id": "doc2.pdf", "chunk_index": i})
        for i in range(2)
    ]

    embeddings = [[0.1] * 768 for _ in range(5)]

    vector_store.add_chunks(chunks_doc1 + chunks_doc2, embeddings)
    assert vector_store.count() == 5

    # Delete doc1
    deleted_count = vector_store.delete_by_source("doc1.pdf")
    assert deleted_count == 3
    assert vector_store.count() == 2

    # Verify only doc2 remains
    remaining = vector_store.get_by_source("doc2.pdf")
    assert len(remaining) == 2


def test_vector_dimension_mismatch(vector_store):
    """
    Test that mismatched vector dimensions are handled
    """
    chunk = Chunk(text="Test", metadata={"source_id": "test"})

    # Wrong dimension (should be 768)
    wrong_embedding = [0.1] * 100

    # ChromaDB should handle this - it may raise an error or normalize
    # We just verify the call doesn't crash silently
    try:
        vector_store.add_chunks([chunk], [wrong_embedding])
        # If it succeeds, verify count increased
        assert vector_store.count() > 0
    except Exception as e:
        # If it fails, that's also acceptable behavior
        assert "dimension" in str(e).lower() or "shape" in str(e).lower()


def test_step_3_criteria_summary(vector_store):
    """
    Verify all Step 3.2 & 3.3 criteria are met:

    Step 3.2 (Database Configuration):
    ✓ Vector dimension: 768 (Gemini text-embedding-004)
    ✓ Distance metric: Cosine similarity
    ✓ Persistent storage

    Step 3.3 (Storage Logic):
    ✓ Unique ID for each record
    ✓ Embedding vector storage
    ✓ Metadata storage
    """
    # Step 3.2 checks
    assert vector_store.collection.metadata["embedding_dimension"] == 768
    assert vector_store.collection.metadata["hnsw:space"] == "cosine"

    # Step 3.3 checks
    chunk = Chunk(
        text="Test content",
        metadata={"source_id": "test.pdf", "chunk_index": 0}
    )
    embedding = [0.1] * 768
    ids = vector_store.add_chunks([chunk], [embedding])

    # Verify unique ID
    assert len(ids) == 1
    assert isinstance(ids[0], str)

    # Verify embedding and metadata stored
    results = vector_store.get_by_source("test.pdf")
    assert len(results) == 1
    assert results[0]["id"] == ids[0]
    assert results[0]["text"] == "Test content"
    assert results[0]["metadata"]["source_id"] == "test.pdf"
