import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from app.parser import PDFParser

def test_pdf_parser_basic():
    # Use the first PDF found in the Books directory for testing
    books_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Books')
    pdf_files = [f for f in os.listdir(books_dir) if f.lower().endswith('.pdf')]
    assert pdf_files, 'No PDF files found in Books directory.'
    sample_pdf = os.path.join(books_dir, pdf_files[0])
    parser = PDFParser(sample_pdf)
    results = parser.parse()
    assert isinstance(results, list)
    assert all('text' in chunk for chunk in results)
    assert all('page_number' in chunk for chunk in results)
    assert all('book_name' in chunk for chunk in results)
    assert all('total_pages' in chunk for chunk in results)
    assert all('file_path' in chunk for chunk in results)
    # At least one page should have non-empty text
    assert any(chunk['text'] for chunk in results) 