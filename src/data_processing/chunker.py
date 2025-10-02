import re
from typing import List, Dict, Any, Optional
from .models import Chunk

class TokenChunker:

    def __init__(self, strategy: str = "tokens", model_name: Optional[str] = None, tokenizer: Optional[Any] = None):
        if strategy not in {"tokens", "sentences", "paragraphs", "hf_tokens"}:
            raise ValueError("strategy must be one of {'tokens', 'sentences', 'paragraphs', 'hf_tokens'}")
        self.strategy = strategy
        # Optional placeholders for future tokenizer-based implementation
        self.model_name = model_name
        self.tokenizer = tokenizer



    def chunk(
        self,
        text: str,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        *,
        source_id: Optional[str] = None,
        source_path: Optional[str] = None,
        page_number: Optional[int] = None,
        chapter_title: Optional[str] = None,
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
        if page_number is not None:
            base_meta["page_number"] = page_number
        if chapter_title is not None:
            base_meta["chapter_title"] = chapter_title
        if extra_meta:
            base_meta.update(extra_meta)

        if self.strategy == "tokens":
            return self._chunk_by_tokens(text, chunk_size, chunk_overlap, base_meta)
        if self.strategy == "sentences":
            return self._chunk_by_sentences(text, chunk_size, chunk_overlap, base_meta)
        if self.strategy == "hf_tokens":
            return self._chunk_by_hf_tokens(text, chunk_size, chunk_overlap, base_meta)

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
            piece = " ".join(tokens[start:end])
            meta = {
                **base_meta,
                "chunk_index": idx,
                "start_token": start,
                "end_token": end,
                "original_text": piece,
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
            piece = " ".join(sentences[start:end])
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
        paragraphs = [p.strip() for p in paragraphs if p]
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

    def _ensure_hf_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        # Lazy import to avoid hard dependency when not used
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise ImportError("Transformers is required for 'hf_tokens' strategy. Install with 'pip install transformers'.") from e
        model_name = self.model_name or "bert-base-uncased"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            # Retry without forcing fast tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def _chunk_by_hf_tokens(
        self, text: str, chunk_size: int, chunk_overlap: int, base_meta: Dict[str, Any]
    ) -> List[Chunk]:
        tokenizer = self._ensure_hf_tokenizer()
        if not text:
            return []
        # Try to use offsets to slice original text accurately
        offsets = None
        input_ids = None
        try:
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            offsets = enc.get("offset_mapping")
            input_ids = enc.get("input_ids")
        except Exception:
            # Fallback without offsets
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = enc.get("input_ids")

        if not input_ids:
            return []

        num_tokens = len(input_ids)
        step = chunk_size - chunk_overlap
        chunks: List[Chunk] = []
        idx = 0
        start = 0

        while start < num_tokens:
            end = min(start + chunk_size, num_tokens)
            if offsets:
                # Use character spans to slice the original text
                start_char = offsets[start][0]
                end_char = offsets[end - 1][1]
                piece = text[start_char:end_char]
                meta = {
                    **base_meta,
                    "chunk_index": idx,
                    "start_token": start,
                    "end_token": end,
                    "start_char": start_char,
                    "end_char": end_char,
                    "tokenizer": getattr(tokenizer, "name_or_path", None),
                }
            else:
                # Decode token window (may not exactly match original spacing)
                piece = tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
                meta = {
                    **base_meta,
                    "chunk_index": idx,
                    "start_token": start,
                    "end_token": end,
                    "tokenizer": getattr(tokenizer, "name_or_path", None),
                }

            chunks.append(Chunk(text=piece, metadata=meta))
            idx += 1
            if end == num_tokens:
                break
            start += step

        return chunks


def chunk_document(
    text: str,
    *,
    strategy: str = "tokens",
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    source_id: Optional[str] = None,
    source_path: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    tokenizer: Optional[Any] = None,
) -> List[Chunk]:
    """
    High-level helper to chunk a full document string into Chunk objects.

    Args:
        text: Full document text.
        strategy: One of 'tokens', 'sentences', 'paragraphs', 'hf_tokens'.
        chunk_size: Size of a chunk in the chosen unit.
        chunk_overlap: Overlap size between consecutive chunks.
        source_id: Optional identifier of the source (e.g., filename or doc ID).
        source_path: Optional absolute path to the source file.
        extra_meta: Extra metadata to attach to each chunk.
        model_name: HF model name to load tokenizer from (only for 'hf_tokens').
        tokenizer: Pre-loaded HF tokenizer to reuse (only for 'hf_tokens').

    Returns:
        List[Chunk]: The chunked document.
    """
    chunker = TokenChunker(strategy=strategy, model_name=model_name, tokenizer=tokenizer)
    return chunker.chunk(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_id=source_id,
        source_path=source_path,
        extra_meta=extra_meta,
    )
