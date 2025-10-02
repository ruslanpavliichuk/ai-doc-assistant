import pytest
from data_processing.chunker import TokenChunker
from data_processing.models import Chunk

@pytest.fixture(scope="module")
def chunker_instance():
    """
    Provides a single instance of TokenChunker for all tests in this module.
    """
    return TokenChunker(strategy="tokens")

def test_token_chunker_basic(chunker_instance):
    """Tests basic chunking functionality."""
    text = "This is a long sentence that needs to be split into several smaller chunks for processing by a language model."
    chunks = chunker_instance.chunk(text, chunk_size=10, chunk_overlap=2)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    # Check that the first chunk starts correctly
    assert chunks[0].text.lower().startswith("this is a long sentence")

def test_chunk_overlap(chunker_instance):
    """Tests that the overlap functionality works as expected."""
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    chunks = chunker_instance.chunk(text, chunk_size=8, chunk_overlap=4)

    assert len(chunks) > 1

    # The end of the first chunk should overlap with the start of the second.
    tokens1 = chunks[0].text.split()
    tokens2 = chunks[1].text.split()

    overlap_from_first = tokens1[-4:]
    overlap_in_second = tokens2[:4]

    assert overlap_from_first == overlap_in_second

def test_empty_text(chunker_instance):
    """Tests that chunking an empty text returns an empty list."""
    chunks = chunker_instance.chunk("", chunk_size=10, chunk_overlap=2)
    assert chunks == []

def test_short_text(chunker_instance):
    """Tests that text shorter than chunk_size results in a single chunk."""
    text = "A very short text."
    chunks = chunker_instance.chunk(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].text == text
