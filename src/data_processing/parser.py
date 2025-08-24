# Module for parsing HTML and PDF

from bs4 import BeautifulSoup
import pymupdf


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
                all_text.append(page.get_text(sort=True))
        return "\f".join(all_text)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""
