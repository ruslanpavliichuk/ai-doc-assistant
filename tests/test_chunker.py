import pytest
from data_processing.chunker import TokenChunker

@pytest.fixture(scope="module")
def chunker_instance():
    """
    Provides a single instance of TokenChunker for all tests in this module.
    This is efficient as the model tokenizer is loaded only once.
    """
    # We use a simple model for fast testing
    return TokenChunker(model_name="bert-base-uncased")

def test_token_chunker_basic(chunker_instance):
    """Tests basic chunking functionality."""
    text = "This is a long sentence that needs to be split into several smaller chunks for processing by a language model."
    chunks = chunker_instance.chunk(text, chunk_size=10, chunk_overlap=2)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    # Check that the first chunk starts correctly
    assert chunks[0].startswith("this is a long sentence")

def test_chunk_overlap(chunker_instance):
    """Tests that the overlap functionality works as expected."""
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    chunks = chunker_instance.chunk(text, chunk_size=8, chunk_overlap=4)

    assert len(chunks) > 1

    # The end of the first chunk should overlap with the start of the second.
    # We tokenize the chunks to see the overlap more clearly.
    tokens1 = chunker_instance.tokenizer.encode(chunks[0], add_special_tokens=False)
    tokens2 = chunker_instance.tokenizer.encode(chunks[1], add_special_tokens=False)

    # The last `chunk_overlap` tokens of the first chunk should be the same as
    # the first `chunk_overlap` tokens of the second chunk.
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
    assert chunks[0] == text.lower() # bert-base-uncased is a lowercasing model

