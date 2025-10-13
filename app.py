import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
import tempfile

# Import project modules
from src.data_processing.parser import parse_document
from src.data_processing.chunker import TokenChunker
from src.core.embeddings import configure_gemini as configure_gemini_embeddings, embed_chunks, embed_text
from src.core.vector_store import VectorStore

# Load environment variables
load_dotenv(".env.local")

# ---- Streamlit Setup ---- #
st.set_page_config(layout="wide", page_title="AI Documentation Assistant")
st.title("🤖 AI Documentation Assistant")

# ---- Configure Gemini ---- #
@st.cache_resource
def configure_gemini():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY environment variable not set. Please add it to your .env.local file.")
        st.stop()
    genai.configure(api_key=api_key)
    configure_gemini_embeddings(api_key)  # Also configure for embeddings
    return genai.GenerativeModel('gemini-2.5-pro')

model = configure_gemini()

# ---- Initialize Vector Store (Step 3.2) ---- #
@st.cache_resource
def get_vector_store():
    """
    Initialize ChromaDB with:
    - Vector dimension: 768 (Gemini text-embedding-004)
    - Distance metric: Cosine similarity
    - Storage: Chroma Cloud
    """
    return VectorStore(
        collection_name="embeddings",  # Your Chroma Cloud collection
        use_cloud=True  # Use Chroma Cloud instead of local storage
    )

vector_store = get_vector_store()

# ---- Sidebar Settings ---- #
st.sidebar.header("⚙️ Settings")
MAX_HISTORY = st.sidebar.number_input(
    "Max History Messages",
    min_value=2,
    max_value=20,
    value=10,
    step=2,
    help="Number of messages to keep in chat history"
)

st.sidebar.divider()

# ---- Document Upload Section ---- #
st.sidebar.header("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload documentation (PDF or HTML)",
    type=["pdf", "html"],
    help="Upload a document to add to the knowledge base"
)

if uploaded_file is not None:
    if st.sidebar.button("🚀 Process Document"):
        with st.sidebar.status("Processing document...", expanded=True) as status:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                st.write("📖 Parsing document...")
                # Step 1: Parse document
                text = parse_document(tmp_path)

                st.write("✂️ Chunking text...")
                # Step 2: Chunk the text with metadata
                chunker = TokenChunker(strategy="tokens")
                chunks = chunker.chunk(
                    text=text,
                    chunk_size=512,
                    chunk_overlap=50,
                    source_id=uploaded_file.name,
                    source_path=uploaded_file.name
                )

                st.write(f"Created {len(chunks)} chunks")

                st.write("🧠 Generating embeddings...")
                # Step 3: Generate embeddings (768-dimensional vectors)
                embeddings = embed_chunks(chunks)

                st.write("💾 Storing in vector database...")
                # Step 3.3: Store in ChromaDB with cosine similarity
                ids = vector_store.add_chunks(chunks, embeddings)

                # Clean up temporary file
                os.unlink(tmp_path)

                status.update(label=f"✅ Successfully processed {uploaded_file.name}!", state="complete")
                st.sidebar.success(f"Added {len(chunks)} chunks to knowledge base!")

            except Exception as e:
                status.update(label="❌ Error processing document", state="error")
                st.sidebar.error(f"Error: {str(e)}")

st.sidebar.divider()

# ---- Database Statistics ---- #
st.sidebar.header("📊 Database Stats")
try:
    total_chunks = vector_store.count()
    st.sidebar.metric("Total Chunks", total_chunks)

    if total_chunks > 0:
        # Get all items to show unique sources
        all_items = vector_store.collection.get()
        if all_items and all_items["metadatas"]:
            unique_sources = set(
                meta.get("source_id", "unknown")
                for meta in all_items["metadatas"]
            )
            st.sidebar.metric("Documents", len(unique_sources))

            with st.sidebar.expander("📄 View Sources"):
                for source in unique_sources:
                    chunks_in_source = sum(
                        1 for meta in all_items["metadatas"]
                        if meta.get("source_id") == source
                    )
                    st.write(f"• {source} ({chunks_in_source} chunks)")
    else:
        st.sidebar.info("👈 Upload a document to get started")

except Exception as e:
    st.sidebar.error(f"Error reading stats: {str(e)}")

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# ---- Initialize Chat History ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources if available
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}:** (Similarity: {source['similarity']:.2%})")
                    st.text(f"📄 Document: {source['metadata'].get('source_id', 'Unknown')}")
                    st.text(f"📍 Chunk: {source['metadata'].get('chunk_index', 'N/A')}")
                    st.markdown(f"```\n{source['text'][:300]}...\n```")

# ---- Trim Function (Removes Oldest Messages) ---- #
def trim_history():
    """Keep only the most recent MAX_HISTORY messages"""
    if len(st.session_state.chat_history) > MAX_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]

# ---- RAG Function ---- #
def get_relevant_context(query: str, top_k: int = 3) -> tuple:
    """
    Retrieve relevant context from vector store using RAG.
    Uses cosine similarity to find most relevant chunks.

    Returns:
        tuple: (context_text, sources_list)
    """
    try:
        # Generate embedding for the query (768-dimensional)
        query_embedding = embed_text(query, task_type="retrieval_query")

        # Search vector store using cosine similarity
        results = vector_store.search(query_embedding, top_k=top_k)

        if not results:
            return "", []

        # Format context
        context_parts = []
        sources = []

        for i, (text, similarity, metadata) in enumerate(results, 1):
            context_parts.append(f"[Context {i}]:\n{text}\n")
            sources.append({
                "text": text,
                "similarity": similarity,
                "metadata": metadata
            })

        context = "\n".join(context_parts)
        return context, sources

    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "", []

# ---- Handle User Input ---- #
if prompt := st.chat_input("Ask me anything about your documentation..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate AI response with RAG
    with st.chat_message("assistant"):
        try:
            # Get relevant context from vector store (RAG)
            context, sources = get_relevant_context(prompt, top_k=3)

            # Create enhanced prompt with context
            if context:
                enhanced_prompt = f"""Based on the following context from the documentation, please answer the user's question.

Context:
{context}

User Question: {prompt}

Please provide a clear and helpful answer based on the context above. If the context doesn't contain relevant information, say so and provide a general answer."""
            else:
                enhanced_prompt = prompt

            # Create chat context from history
            chat = model.start_chat(history=[])

            # Add previous messages for context (excluding the current one)
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] == "user":
                    chat.send_message(msg["content"])

            # Send current message and get streaming response
            response = chat.send_message(enhanced_prompt, stream=True)

            # Display streaming response
            message_placeholder = st.empty()
            full_response = ""

            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")

            # Remove cursor and show final response
            message_placeholder.markdown(full_response)

            # Show sources if available
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** (Similarity: {source['similarity']:.2%})")
                        st.text(f"📄 Document: {source['metadata'].get('source_id', 'Unknown')}")
                        st.text(f"📍 Chunk: {source['metadata'].get('chunk_index', 'N/A')}")
                        st.markdown(f"```\n{source['text'][:300]}...\n```")

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            st.error(full_response)
            sources = []

    # Add assistant response to history
    history_entry = {"role": "assistant", "content": full_response}
    if sources:
        history_entry["sources"] = sources

    st.session_state.chat_history.append(history_entry)

    # Trim history to keep it manageable
    trim_history()
