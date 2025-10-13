from pathlib import Path
from typing import List

import pytest

from data_processing.parser import parse_pdf
from data_processing.chunker import chunk_document


def test_parse_chunk_embed_pipeline(monkeypatch):
    # Arrange: locate the sample PDF
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data" / "raw" / "tutorial.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Missing test fixture: {pdf_path}")

    # Act 1: parse
    text = parse_pdf(str(pdf_path))
    assert isinstance(text, str) and len(text) > 0

    # Act 2: chunk
    chunks = chunk_document(
        text,
        strategy="tokens",
        chunk_size=80,
        chunk_overlap=20,
        source_id=pdf_path.name,
        source_path=str(pdf_path),
        extra_meta={"test_case": "e2e"},
    )
    assert len(chunks) > 0

    # Prepare a deterministic fake embedder
    calls: List[str] = []

    def fake_embed_text(t: str, *, model: str = "models/text-embedding-004", task_type: str = "retrieval_document"):
        calls.append(t)
        # Return a tiny feature vector derived from the text length/word count
        return [float(len(t)), float(len(t.split())), 0.0]

    # Patch embedding to avoid external API
    monkeypatch.setattr("core.embeddings.embed_text", fake_embed_text)

    # Act 3: embed
    from core.embeddings import embed_text

    vectors = [embed_text(c.text) for c in chunks]

    # Assert
    assert len(vectors) == len(chunks)
    assert len(calls) == len(chunks)
    for v in vectors:
        assert isinstance(v, list) and len(v) == 3
        assert all(isinstance(x, float) for x in v)
