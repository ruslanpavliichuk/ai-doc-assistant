"""
Diagnostic script to check ChromaDB contents
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.vector_store import VectorStore

def check_database():
    """Check what's stored in ChromaDB"""

    print("=" * 60)
    print("ChromaDB Diagnostic Tool - Chroma Cloud")
    print("=" * 60)

    # Initialize vector store (same as in app.py) - using Chroma Cloud
    vector_store = VectorStore(
        collection_name="embeddings",
        use_cloud=True
    )

    print(f"\n📊 Collection: {vector_store.collection.name}")
    print(f"🌐 Storage: Chroma Cloud")

    # Get total count
    total_count = vector_store.count()
    print(f"\n📈 Total chunks stored: {total_count}")

    if total_count == 0:
        print("\n⚠️  Database is EMPTY!")
        print("\nPossible reasons:")
        print("1. No documents have been processed yet")
        print("2. Processing failed before storage")
        print("3. Different collection name or path being used")
        return

    # Get all data
    print("\n" + "=" * 60)
    print("Stored Documents:")
    print("=" * 60)

    try:
        # Get all items from collection
        all_items = vector_store.collection.get()

        if all_items and all_items["ids"]:
            # Group by source_id
            sources = {}
            for i, item_id in enumerate(all_items["ids"]):
                metadata = all_items["metadatas"][i] if all_items["metadatas"] else {}
                source_id = metadata.get("source_id", "unknown")

                if source_id not in sources:
                    sources[source_id] = []

                sources[source_id].append({
                    "id": item_id,
                    "text": all_items["documents"][i][:100] + "..." if all_items["documents"] else "N/A",
                    "metadata": metadata
                })

            # Display by source
            for source_id, chunks in sources.items():
                print(f"\n📄 Source: {source_id}")
                print(f"   Chunks: {len(chunks)}")
                print(f"   Sample chunks:")
                for idx, chunk in enumerate(chunks[:3], 1):  # Show first 3
                    print(f"\n   [{idx}] ID: {chunk['id'][:8]}...")
                    print(f"       Index: {chunk['metadata'].get('chunk_index', 'N/A')}")
                    print(f"       Text: {chunk['text']}")

                if len(chunks) > 3:
                    print(f"\n   ... and {len(chunks) - 3} more chunks")

        # Test search functionality
        print("\n" + "=" * 60)
        print("Testing Search:")
        print("=" * 60)

        from src.core.embeddings import embed_text

        test_query = "Python programming"
        print(f"\n🔍 Query: '{test_query}'")

        query_embedding = embed_text(test_query, task_type="retrieval_query")
        results = vector_store.search(query_embedding, top_k=3)

        if results:
            print(f"\n✅ Found {len(results)} results:")
            for i, (text, similarity, metadata) in enumerate(results, 1):
                print(f"\n[{i}] Similarity: {similarity:.2%}")
                print(f"    Source: {metadata.get('source_id', 'unknown')}")
                print(f"    Text: {text[:150]}...")
        else:
            print("\n❌ No results found (but database has data)")

    except Exception as e:
        print(f"\n❌ Error reading database: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_database()
