import pytest
from pathlib import Path

from data_processing.parser import parse_html, parse_pdf

@pytest.fixture
def html_content():
    """Reads the content of the test HTML file."""
    file_path = Path(__file__).resolve().parent / 'data' / 'test.html'
    with open(file_path, 'r') as f:
        return f.read()

def test_parse_html(html_content):
    """Tests the parse_html function."""
    expected_text = "Test Page\nWelcome\nThis is a test paragraph.\nSome nested text."
    actual_text = parse_html(html_content)
    assert actual_text.strip() == expected_text.strip()

def test_parse_pdf():
    """Tests the parse_pdf function."""
    file_path = Path(__file__).resolve().parent / 'data' / 'test_pdf.pdf'

    # Make sure the test PDF file exists before running the test
    if not file_path.exists():
        pytest.fail(f"Test file not found: {file_path}")

    expected_text = "This is text for testing PDF parser, you are totally fine to use your own text."

    actual_text = parse_pdf(str(file_path))
    assert expected_text in actual_text

# To run this test:
# 1. Make sure you have pytest installed: pip install pytest
# 2. Navigate to the root of your project in the terminal.
# 3. Run the command: pytest
