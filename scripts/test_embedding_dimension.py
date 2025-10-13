"""
Test script to verify Gemini embedding dimensions and ChromaDB configuration.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv(".env.local")

# Configure Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise Exception("GOOGLE_API_KEY not found in environment")

genai.configure(api_key=api_key)

# Test embedding
test_text = "This is a test sentence to check embedding dimensions."
result = genai.embed_content(
    model="models/text-embedding-004",
    content=test_text,
    task_type="retrieval_document"
)

embedding = result["embedding"]
dimension = len(embedding)

print(f"✅ Embedding model: text-embedding-004")
print(f"✅ Vector dimension: {dimension}")
print(f"✅ First 5 values: {embedding[:5]}")
print(f"\nThis dimension should be configured in ChromaDB vector store.")

