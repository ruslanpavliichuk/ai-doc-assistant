import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.parser import parse_pdf, parse_html
from src.data_processing.chunker import TokenChunker


def parse_document(file_path: str) -> str:
    """
    Parses a document, automatically detecting the file type (PDF or HTML).
    Args:
        file_path (str): The path to the document.
    Returns:
        str: The extracted text content.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"The file was not found: {file_path}")

    file_ext = file_path_obj.suffix.lower()

    if file_ext == ".pdf":
        return parse_pdf(file_path)
    elif file_ext in [".html", ".htm"]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return parse_html(content)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Only .pdf and .html are supported.")


def process_document(file_path: str):
    """
    Processes a single document by parsing it and splitting it into chunks.
    """
    print(f"--> Processing document: {file_path}")
    text = parse_document(file_path)
    if not text:
        print(f"Warning: No text extracted from {file_path}")
        return []

    # Instantiate the chunker and then call the chunk method with parameters
    chunker = TokenChunker()
    chunks = chunker.chunk(text, chunk_size=200, chunk_overlap=50)

    return chunks


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this path to the document you want to process.
    # Using the test HTML file as an example.
    document_path = "tests/data/test.html"

    try:
        document_chunks = process_document(document_path)

        if document_chunks:
            print(f"\n--> Successfully created {len(document_chunks)} chunks.")
            for i, chunk in enumerate(document_chunks[:3], 1):
                print(f"\n--- Chunk {i} ---\n{chunk}")
        else:
            print("--> No chunks were created.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
