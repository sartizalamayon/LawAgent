import os
from typing import List, Dict
import pymupdf4llm
from tqdm import tqdm

class PDFParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.book_name = os.path.splitext(os.path.basename(pdf_path))[0]

    def parse(self) -> List[Dict]:
        # Extract markdown for all pages
        # pymupdf4llm.to_markdown returns the full document as markdown
        # To get per-page, we use PyMuPDF
        import fitz  # PyMuPDF
        doc = fitz.open(self.pdf_path)
        total_pages = doc.page_count
        results = []
        for i in tqdm(range(total_pages), desc=f"Parsing {self.book_name}"):
            md_text = pymupdf4llm.to_markdown(self.pdf_path, pages=[i])
            results.append({
                "book_name": self.book_name,
                "page_number": i + 1,
                "text": md_text.strip(),
                "total_pages": total_pages,
                "file_path": self.pdf_path
            })
        return results 