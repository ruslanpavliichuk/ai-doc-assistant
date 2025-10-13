import os
from typing import List, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai

from src.data_processing.models import Chunk


def configure_gemini(api_key: Optional[str] = None) -> None:
    load_dotenv("../.env.local")
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)


def embed_text(text: str,
               model: str = "models/text-embedding-004",
               task_type: str = "retrieval_document") -> List[float]:
    """Low-level function to embed a single text string."""
    result = genai.embed_content(model=model, content=text, task_type=task_type)
    return result["embedding"]


def embed_chunks(
    chunks: Union[Chunk, List[Chunk]],
    model: str = "models/text-embedding-004",
    task_type: str = "retrieval_document"
) -> Union[List[float], List[List[float]]]:
    """
    Service wrapper for embedding model (Step 2.2).
    Accepts a Chunk object or list of Chunk objects and returns embedding vector(s).

    Args:
        chunks: Single Chunk or list of Chunk objects
        model: Gemini embedding model name
        task_type: Task type for embeddings (retrieval_document or retrieval_query)

    Returns:
        Single embedding vector (List[float]) if input is single Chunk,
        or list of embedding vectors (List[List[float]]) if input is list of Chunks

    Example:
        # Single chunk
        chunk = Chunk(text="Hello world", metadata={"source": "doc1"})
        vec = embed_chunks(chunk)

        # Multiple chunks
        chunks = [Chunk(text="chunk 1", metadata={}), Chunk(text="chunk 2", metadata={})]
        vecs = embed_chunks(chunks)
    """
    # Handle single Chunk
    if isinstance(chunks, Chunk):
        return embed_text(chunks.text, model=model, task_type=task_type)

    # Handle list of Chunks
    elif isinstance(chunks, list):
        embeddings = []
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise TypeError(f"Expected Chunk object, got {type(chunk)}")
            embedding = embed_text(chunk.text, model=model, task_type=task_type)
            embeddings.append(embedding)
        return embeddings

    else:
        raise TypeError(f"chunks must be Chunk or List[Chunk], got {type(chunks)}")
