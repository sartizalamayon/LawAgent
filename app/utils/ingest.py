import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure the app directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.parser.pdf_parser import PDFParser
from app.parser.chunking import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.qdrant_client import QdrantClientWrapper

def ingest_books(books_dir: str, collection_name: str, batch_size: int = 32):
    """
    Parses, chunks, embeds, and upserts all PDF books from a directory into Qdrant.
    """
    # Load environment variables (for QDRANT_API_KEY, etc.)
    load_dotenv()

    # Initialize components
    chunker = TextChunker()
    embedder = EmbeddingModel()
    qdrant_client = QdrantClientWrapper(collection_name)

    # Get all PDF files from the books directory
    pdf_files = [f for f in os.listdir(books_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF books to ingest.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(books_dir, pdf_file)
        
        # 1. Parse PDF
        parser = PDFParser(pdf_path)
        page_dicts = parser.parse()
        
        # 2. Chunk text
        chunks = chunker.chunk(page_dicts)
        
        # Extract text and metadata for embedding and upsert
        chunk_texts = [chunk['chunk_text'] for chunk in chunks]
        metadatas = [{k: v for k, v in chunk.items() if k != 'chunk_text'} for chunk in chunks]

        # 3. Embed chunks
        embeddings = embedder.embed_texts(chunk_texts, batch_size=batch_size)
        
        # 4. Upsert to Qdrant
        qdrant_client.upsert_chunks(embeddings, metadatas, batch_size=batch_size)
        
        print(f"Successfully ingested {pdf_file} ({len(chunks)} chunks).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest law books into Qdrant.")
    parser.add_argument("--collection", type=str, default="law_books", help="Qdrant collection name.")
    parser.add_argument("--books_dir", type=str, default="Books", help="Directory containing PDF books.")
    args = parser.parse_args()
    
    ingest_books(args.books_dir, args.collection) 