import re
from typing import List, Dict, Any, Optional
from .models import Chunk

class TokenChunker:

    def __init__(self, strategy: str = "tokens"):
        if strategy not in {"tokens", "sentences", "paragraphs"}:
            raise ValueError("strategy must be one of {'tokens', 'sentences', or 'paragraphs'}")
        self.strategy = strategy



    def chunk(
        self,
        text: str,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        *,
        source_id: Optional[str] = None,
        source_path: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        base_meta: Dict[str, Any] = {
            "source_id": source_id,
            "source_path": source_path,
            "strategy": self.strategy,
        }
        if extra_meta:
            base_meta.update(extra_meta)

        if self.strategy == "tokens":
            return self._chunk_by_tokens(text, chunk_size, chunk_overlap, base_meta)
        if self.strategy == "sentences":
            return self._chunk_by_sentences(text, chunk_size, chunk_overlap, base_meta)

        return self._chunk_by_paragraphs(text, chunk_size, chunk_overlap, base_meta)

    def _chunk_by_tokens(
        self, text: str, chunk_size: int, chunk_overlap: int, base_meta: Dict[str, Any]
    ) -> List[Chunk]:
        tokens = text.split()
        if not tokens:
            return []

        step = chunk_size - chunk_overlap
        chunks: List[Chunk] = []
        start = 0
        idx = 0


        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            piece = ".".join(tokens[start:end])
            meta = {
                **base_meta,
                "chunk_index": idx,
                "start_token": start,
                "end_token": end,
            }

            chunks.append(Chunk(text=piece, metadata=meta))
            idx += 1
            if end == len(tokens):
                break
            start += step

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]

    def _chunk_by_sentences(
        self, text: str, chunk_size: int, chunk_overlap: int, base_meta: Dict[str, Any]
    ) -> List[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        step = max (1, chunk_size - chunk_overlap)
        chunks: List[Chunk] = []
        idx = 0

        for start in range(0, len(sentences), step):
            end = min(start + chunk_size, len(sentences))
            piece = ".".join(sentences[start:end])
            meta = {
                **base_meta,
                "chunk_index": idx,
                "start_sentence": start,
                "end_sentence": end,
            }
            chunks.append(Chunk(text=piece, metadata=meta))
            idx += 1
            if end == len(sentences):
                break

        return chunks


    def _chunk_by_paragraphs(
        self, text: str, chunk_size: int, chunk_overlap: int, base_meta: Dict[str, Any]
    ) -> List[Chunk]:
        paragraphs = re.split(r"\n{2,}", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.string()]
        if not paragraphs:
            return []

        step = max(1, chunk_size - chunk_overlap)
        chunks: List[Chunk] = []
        idx = 0

        for start in range(0, len(paragraphs), step):
            end = min(start + chunk_size, len(paragraphs))
            piece = "\n\n".join(paragraphs[start:end])
            meta = {
                **base_meta,
                "chunk_index": idx,
                "start_paragraph": start,
                "end_paragraph": end,
            }
            chunks.append(Chunk(text=piece, metadata=meta))
            idx += 1
            if end == len(paragraphs):
                break

        return chunks