"""
Vector store implementation using ChromaDB (Step 3.2 & 3.3)
Handles storage and retrieval of document embeddings with metadata.
"""
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

from src.data_processing.models import Chunk


class VectorStore:
    """
    Service for storing and retrieving document embeddings in ChromaDB.

    Configuration (Step 3.2):
    - Vector dimension: 768 (Gemini text-embedding-004)
    - Distance metric: Cosine similarity
    - Storage: Chroma Cloud (remote) or local persistent database

    Storage logic (Step 3.3):
    - Each record contains: unique ID, embedding vector, metadata
    - Metadata includes: source document, chunk index, original text
    """

    def __init__(
        self,
        collection_name: str = "embeddings",
        persist_directory: str = "./chroma_db",
        use_cloud: bool = True
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection to store embeddings
            persist_directory: Directory to persist the database (local mode)
            use_cloud: Whether to use Chroma Cloud (True) or local storage (False)
        """
        # Step 3.2: Configure ChromaDB with cloud or local persistence
        if use_cloud:
            # Use Chroma Cloud
            chroma_api_key = os.environ.get("CHROMA_API_KEY")
            tenant = os.environ.get("TENANT")
            database = os.environ.get("DATABASE")

            if not all([chroma_api_key, tenant, database]):
                raise ValueError(
                    "Chroma Cloud credentials missing. Please set CHROMA_API_KEY, "
                    "TENANT, and DATABASE in .env.local"
                )

            self.client = chromadb.HttpClient(
                host="api.trychroma.com",
                ssl=True,
                headers={
                    "x-chroma-token": chroma_api_key
                },
                tenant=tenant,
                database=database
            )
            print(f"✅ Connected to Chroma Cloud - Database: {database}, Tenant: {tenant}")
        else:
            # Use local persistent storage
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print(f"✅ Connected to local ChromaDB at: {persist_directory}")

        # Create or get collection with cosine similarity metric
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity
                "description": "Document chunks with embeddings from Gemini text-embedding-004",
                "embedding_dimension": 768  # Gemini text-embedding-004 dimension
            }
        )
        print(f"✅ Using collection: {collection_name}")

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Step 3.3: Store chunks with their embeddings and metadata.

        Args:
            chunks: List of Chunk objects with text and metadata
            embeddings: List of embedding vectors (must match chunks length)
            ids: Optional list of unique IDs (auto-generated if not provided)

        Returns:
            List of IDs for the stored chunks

        Example:
            chunks = [Chunk(text="Hello", metadata={"page": 1})]
            embeddings = [[0.1, 0.2, ...]]  # 768-dim vector
            ids = store.add_chunks(chunks, embeddings)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match embeddings ({len(embeddings)})")

        # Generate unique IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Prepare data for ChromaDB
        documents = []  # Original text for each chunk
        metadatas = []  # Metadata for each chunk

        for chunk in chunks:
            documents.append(chunk.text)

            # Flatten metadata for ChromaDB (must be JSON-serializable)
            meta = {
                "source_id": str(chunk.metadata.get("source_id", "unknown")),
                "source_path": str(chunk.metadata.get("source_path", "")),
                "chunk_index": int(chunk.metadata.get("chunk_index", 0)),
                "strategy": str(chunk.metadata.get("strategy", "unknown")),
            }

            # Add optional fields if present
            if "page_number" in chunk.metadata:
                meta["page_number"] = int(chunk.metadata["page_number"])
            if "chapter_title" in chunk.metadata:
                meta["chapter_title"] = str(chunk.metadata["chapter_title"])

            metadatas.append(meta)

        # Step 3.3: Store in vector database
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query vector (768-dim for Gemini)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"source_id": "doc1.pdf"})

        Returns:
            List of tuples: (chunk_text, similarity_score, metadata)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        output = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                text = results["documents"][0][i]
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert distance to similarity (ChromaDB returns distances)
                similarity = 1.0 - distance  # For cosine, distance = 1 - similarity
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                output.append((text, similarity, metadata))

        return output

    def get_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific source document.

        Args:
            source_id: The source document ID

        Returns:
            List of chunk data with metadata
        """
        results = self.collection.get(
            where={"source_id": source_id}
        )

        output = []
        if results["documents"]:
            for i in range(len(results["documents"])):
                output.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })

        return output

    def delete_by_source(self, source_id: str) -> int:
        """
        Delete all chunks from a specific source document.

        Args:
            source_id: The source document ID

        Returns:
            Number of chunks deleted
        """
        # Get IDs to delete
        results = self.collection.get(
            where={"source_id": source_id}
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()

    def clear(self):
        """Delete all data from the collection."""
        self.client.delete_collection(self.collection.name)
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={
                "hnsw:space": "cosine",
                "description": "Document chunks with embeddings from Gemini text-embedding-004",
                "embedding_dimension": 768
            }
        )
