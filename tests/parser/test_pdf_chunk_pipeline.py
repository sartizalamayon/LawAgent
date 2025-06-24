import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from app.parser import PDFParser
from app.parser.chunking import TextChunker

def test_pdf_to_chunks_pipeline():
    # Use the first PDF found in the Books directory for testing
    books_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Books')
    pdf_files = [f for f in os.listdir(books_dir) if f.lower().endswith('.pdf')]
    assert pdf_files, 'No PDF files found in Books directory.'
    sample_pdf = os.path.join(books_dir, pdf_files[0])
    parser = PDFParser(sample_pdf)
    page_dicts = parser.parse()
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(page_dicts)
    print(f"Parsed {len(page_dicts)} pages from {sample_pdf}")
    print(f"Generated {len(chunks)} chunks")
    print(f"Chunk sizes (first 10): {[len(c['chunk_text']) for c in chunks[:10]]}")
    assert len(chunks) > 0
    assert all('chunk_text' in c for c in chunks)
    assert all('chunk_index' in c for c in chunks)
    assert all('book_name' in c for c in chunks)
    assert all('page_number' in c for c in chunks)
    assert all('file_path' in c for c in chunks) 