import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from app.parser.chunking import TextChunker

def test_text_chunker_basic():
    # Simulate parsed page dicts
    page_dicts = [
        {
            "text": "A" * 2500,
            "book_name": "TestBook",
            "page_number": 1,
            "total_pages": 1,
            "file_path": "/fake/path.pdf"
        }
    ]
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(page_dicts)
    # LangChain may split differently, but should still cover all text
    total_len = sum(len(c["chunk_text"]) for c in chunks)
    assert total_len >= 2500
    assert all("chunk_text" in c for c in chunks)
    assert all("chunk_index" in c for c in chunks)
    assert all("book_name" in c for c in chunks)
    assert all("page_number" in c for c in chunks)
    assert all("file_path" in c for c in chunks) 