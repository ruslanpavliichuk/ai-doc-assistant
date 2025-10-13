"""
Example: Complete pipeline - Parse → Chunk → Embed → Store in Vector DB
Demonstrates integration of all steps (1-3) for Step 3.2 & 3.3
"""
from pathlib import Path

from data_processing.parser import parse_pdf
from data_processing.chunker import chunk_document
from core.embeddings import configure_gemini, embed_text
from core.vector_store import VectorStore


def process_and_store_document(pdf_path: str, store: VectorStore) -> int:
    """
    Complete pipeline: parse PDF → chunk → embed → store in vector DB

    Args:
        pdf_path: Path to PDF file
        store: VectorStore instance

    Returns:
        Number of chunks stored

    Example:
        store = VectorStore(persist_directory="./my_db")
        configure_gemini()
        num_chunks = process_and_store_document("tutorial.pdf", store)
        print(f"Stored {num_chunks} chunks")
    """
    # Step 1: Parse PDF
    print(f"📄 Parsing {pdf_path}...")
    text = parse_pdf(pdf_path)

    if not text or len(text.strip()) == 0:
        raise ValueError(f"No text extracted from {pdf_path}")

    # Step 2: Chunk document
    print(f"✂️  Chunking document...")
    chunks = chunk_document(
        text,
        strategy="tokens",
        chunk_size=200,
        chunk_overlap=50,
        source_id=Path(pdf_path).name,
        source_path=pdf_path
    )

    if not chunks:
        raise ValueError("No chunks created")

    print(f"   Created {len(chunks)} chunks")

    # Step 2.2: Generate embeddings
    print(f"🔢 Generating embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"   Processing chunk {i+1}/{len(chunks)}...")

        embedding = embed_text(
            chunk.text,
            model="models/text-embedding-004",
            task_type="retrieval_document"
        )
        embeddings.append(embedding)

    # Step 3.3: Store in vector database
    print(f"💾 Storing in vector database...")
    ids = store.add_chunks(chunks, embeddings)

    print(f"✅ Successfully stored {len(ids)} chunks")
    return len(ids)


def search_documents(query: str, store: VectorStore, top_k: int = 3):
    """
    Search for relevant chunks using semantic similarity

    Args:
        query: Search query
        store: VectorStore instance
        top_k: Number of results to return

    Returns:
        List of (text, similarity, metadata) tuples

    Example:
        results = search_documents("What is Python?", store, top_k=3)
        for text, score, meta in results:
            print(f"[{score:.3f}] {meta['source_id']}: {text[:100]}...")
    """
    # Generate query embedding
    query_embedding = embed_text(
        query,
        model="models/text-embedding-004",
        task_type="retrieval_query"
    )

    # Search in vector database
    results = store.search(query_embedding, top_k=top_k)

    return results


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting document processing pipeline...\n")

    # Configure Gemini API
    configure_gemini()

    # Initialize vector store
    store = VectorStore(
        collection_name="tutorial_docs",
        persist_directory="./chroma_db"
    )

    # Process a document
    pdf_path = "./data/raw/tutorial.pdf"

    try:
        num_chunks = process_and_store_document(pdf_path, store)

        print(f"\n📊 Database stats:")
        print(f"   Total chunks: {store.count()}")

        # Example search
        print(f"\n🔍 Testing search...")
        query = "What is Python?"
        results = search_documents(query, store, top_k=3)

        print(f"\nTop {len(results)} results for: '{query}'")
        for i, (text, similarity, metadata) in enumerate(results, 1):
            print(f"\n{i}. [Score: {similarity:.3f}] {metadata.get('source_id', 'Unknown')}")
            print(f"   {text[:150]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")

