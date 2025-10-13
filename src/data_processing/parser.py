# Module for parsing HTML and PDF
from bs4 import BeautifulSoup
import pymupdf
from pathlib import Path

def parse_html(html_content):
    """
    Parse HTML content and extract text.
    Args:
        html_content (str): The HTML content to parse.
    Returns:
        str: Extracted text from the HTML.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

def parse_pdf(file_path):
    """
    Parse PDF file and extract text.
    Args:
        file_path (str): The path to the PDF file.
    Returns:
        str: Extracted text from the PDF.
    """

    all_text = []
    try:
        with pymupdf.open(file_path) as doc:
            for page in doc:
                all_text.append(page.get_text(sort=True, flags=pymupdf.TEXT_PRESERVE_WHITESPACE))
        return "\f".join(all_text)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""

def parse_document(file_path):
    """
    Parse a document (PDF or HTML) and extract text.
    Automatically detects file type based on extension.

    Args:
        file_path (str): The path to the document file.

    Returns:
        str: Extracted text from the document.

    Raises:
        ValueError: If file type is not supported.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension == '.pdf':
        return parse_pdf(str(file_path))
    elif extension in ['.html', '.htm']:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return parse_html(html_content)
    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported types: .pdf, .html, .htm")
