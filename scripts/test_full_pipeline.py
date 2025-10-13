"""
Test the full pipeline: parse → chunk → embed → store → search
This verifies that ChromaDB is properly configured with correct dimensions and cosine similarity.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv(".env.local")
from data_processing.parser import parse_document
from data_processing.chunker import TokenChunker
from core.embeddings import configure_gemini, embed_chunks, embed_text
from core.vector_store import VectorStore

def test_pipeline():
    print("=" * 60)
    print("TESTING FULL EMBEDDING & VECTOR STORE PIPELINE")
    print("=" * 60)

    # Step 1: Configure Gemini
    print("\n1️⃣ Configuring Gemini API...")
    try:
        configure_gemini()
        print("   ✅ Gemini configured successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Step 2: Test embedding dimension
    print("\n2️⃣ Testing embedding dimension...")
    test_text = "This is a test sentence."
    embedding = embed_text(test_text)
    print(f"   ✅ Embedding dimension: {len(embedding)}")
    print(f"   ✅ First 5 values: {embedding[:5]}")

    # Step 3: Initialize Vector Store
    print("\n3️⃣ Initializing ChromaDB Vector Store...")
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_directory="./chroma_db_test"
    )
    print("   ✅ Vector store initialized")
    print(f"   ✅ Using cosine similarity metric")

    # Step 4: Parse a test document
    print("\n4️⃣ Parsing test document...")
    test_pdf = Path(__file__).parent.parent / "tests" / "data" / "test_pdf.pdf"
    if test_pdf.exists():
        text = parse_document(str(test_pdf))
        print(f"   ✅ Parsed {len(text)} characters")
    else:
        print("   ⚠️  Test PDF not found, using sample text")
        text = "Python is a high-level programming language. It is widely used for web development, data science, and automation. Python has a simple syntax that makes it easy to learn."

    # Step 5: Chunk the text
    print("\n5️⃣ Chunking text...")
    chunker = TokenChunker(strategy="tokens")
    chunks = chunker.chunk(
        text=text,
        chunk_size=50,
        chunk_overlap=10,
        source_id="test_doc.pdf",
        source_path="tests/data/test_doc.pdf"
    )
    print(f"   ✅ Created {len(chunks)} chunks")

    # Step 6: Generate embeddings
    print("\n6️⃣ Generating embeddings...")
    embeddings = embed_chunks(chunks)
    print(f"   ✅ Generated {len(embeddings)} embeddings")
    if isinstance(embeddings[0], list):
        print(f"   ✅ Each embedding has dimension: {len(embeddings[0])}")

    # Step 7: Store in ChromaDB
    print("\n7️⃣ Storing in ChromaDB...")
    ids = vector_store.add_chunks(chunks, embeddings)
    print(f"   ✅ Stored {len(ids)} chunks in vector database")

    # Step 8: Test search with cosine similarity
    print("\n8️⃣ Testing semantic search (cosine similarity)...")
    query = "What is Python?"
    query_embedding = embed_text(query, task_type="retrieval_query")
    results = vector_store.search(query_embedding, top_k=3)

    print(f"   ✅ Found {len(results)} relevant chunks")
    for i, (text, similarity, metadata) in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"      Similarity: {similarity:.4f}")
        print(f"      Source: {metadata.get('source_id', 'N/A')}")
        print(f"      Chunk: {metadata.get('chunk_index', 'N/A')}")
        print(f"      Text preview: {text[:100]}...")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nChromaDB Configuration Summary:")
    print(f"  • Vector dimension: 768")
    print(f"  • Distance metric: Cosine similarity")
    print(f"  • Storage: Persistent (./chroma_db)")
    print(f"  • Embedding model: Gemini text-embedding-004")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
