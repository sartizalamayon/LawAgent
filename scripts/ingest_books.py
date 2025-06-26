import os
import json
from glob import glob
from app.parser.pdf_parser import PDFParser
from app.parser.chunking import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.qdrant_client import QdrantClientWrapper
import tqdm

BOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Books')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'rag_books.json')
COLLECTION_NAME = 'law_books'

# Load or initialize the book index
def load_index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_index(index):
    with open(INDEX_PATH, 'w') as f:
        json.dump(index, f, indent=2)

def process_pdf(pdf_path, chunker):
    """Process a single PDF file"""
    try:
        parser = PDFParser(pdf_path)
        pages = parser.parse()
        chunks = chunker.chunk(pages)
        return pdf_path, pages, chunks
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def main():
    index = load_index()
    pdf_files = glob(os.path.join(BOOKS_DIR, '*.pdf'))
    new_files = [f for f in pdf_files if os.path.basename(f) not in index]
    if not new_files:
        print('No new books to ingest.')
        return
    print(f'Found {len(new_files)} new books.')

    chunker = TextChunker()
    embedder = EmbeddingModel()
    qdrant = QdrantClientWrapper(COLLECTION_NAME)

    # Process one book at a time
    for pdf_path in tqdm.tqdm(new_files, desc="Processing books"):
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        
        # Parse and chunk the PDF
        result = process_pdf(pdf_path, chunker)
        if result is None:
            print(f"Skipping {os.path.basename(pdf_path)} due to processing error")
            continue
            
        pdf_path, pages, chunks = result
        
        # Prepare texts and metadata
        texts = [c["chunk_text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {k: v for k, v in c.items() if k != "chunk_text"}
            text = c["chunk_text"]
            meta["chunk_text"] = text
            meta["page_content"] = text
            metadatas.append(meta)
        
        # Embed and store
        try:
            embeddings = embedder.embed_texts(texts)
            qdrant.upsert_chunks(embeddings, metadatas)
            
            # Update index
            index[os.path.basename(pdf_path)] = {
                "pages": len(pages),
                "chunks": len(chunks)
            }
            save_index(index)
            print(f'Successfully ingested {os.path.basename(pdf_path)} ({len(chunks)} chunks)')
            
        except Exception as e:
            print(f"Error embedding/storing {os.path.basename(pdf_path)}: {e}")
            continue

if __name__ == '__main__':
    main() 