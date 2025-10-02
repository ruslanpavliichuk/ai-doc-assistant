import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.parser import parse_pdf, parse_html
from src.data_processing.chunker import TokenChunker, chunk_document

def parse_document(file_path: str) -> str:
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
    print(f"--> Processing document: {file_path}")
    text = parse_document(file_path)
    if not text:
        print(f"Warning: No text extracted from {file_path}")
        return []

    source_id = Path(file_path).name
    # You can switch strategy to 'sentences' or 'paragraphs' if desired
    chunks = chunk_document(
        text,
        strategy="tokens",
        chunk_size=200,
        chunk_overlap=50,
        source_id=source_id,
        source_path=str(Path(file_path).resolve()),
        extra_meta={"file_type": Path(file_path).suffix.lower()},
    )

    return chunks


if __name__ == "__main__":
    document_path = "../data/raw/tutorial.pdf"
    try:
        document_chunks = process_document(document_path)

        if document_chunks:
            print(f"\n--> Successfully created {len(document_chunks)} chunks.")
            for i, chunk in enumerate(document_chunks[:3], 1):
                print(f"\n--- Chunk {i} ---")
                print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
                print("Metadata:", {k: v for k, v in chunk.metadata.items() if k != "source_path"})
        else:
            print("--> No chunks were created.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
