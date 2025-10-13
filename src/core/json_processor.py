import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from .embeddings import configure_gemini, embed_text

def _normalize_items(payload: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    for key in ("items", "chunks", "data"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    raise ValueError("Unsupported JSON structure. Expect a list of items or a dict with 'items'/'chunks'/'data'.")

def process_json_and_embed(
    json_path: str,
    output_path: Optional[str] = None,
    *,
    model: str = "models/text-embedding-004",
    task_type: str = "retrieval_document",
    api_key: Optional[str] = None,
    overwrite: bool = False,
    skip_empty: bool = True,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    - Reads chunks from json_path (expects each item to have 'text' and optional 'metadata').
    - Generates embeddings with Gemini.
    - Returns enriched items and optionally writes to output_path (.json or .jsonl).
    """
    configure_gemini(api_key)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = _normalize_items(payload)

    enriched: List[Dict[str, Any]] = []

    for item in items:
        text = (item.get("text") or "").strip()
        if not text and skip_empty:
            continue

        metadata = item.get("metadata") or {}
        chunk_id = item.get("id") or metadata.get("id") or str(uuid.uuid4())

        # Reuse existing embedding unless overwrite=True
        if not overwrite and isinstance(item.get("embedding"), list) and item["embedding"]:
            embedding = item["embedding"]
        else:
            attempt = 0
            while True:
                try:
                    embedding = embed_text(text, model=model, task_type=task_type)
                    break
                except Exception:
                    attempt += 1
                    if attempt >= max_retries:
                        raise
                    time.sleep(2 ** attempt)

        enriched.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
            }
        )

    if output_path:
        if output_path.endswith(".jsonl"):
            with open(output_path, "w", encoding="utf-8") as out:
                for row in enriched:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as out:
                json.dump(enriched, out, ensure_ascii=False, indent=2)

    return enriched