import os
import json
import hashlib
from typing import List, Dict, Set, Optional
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import logging

from ..parser.pdf_parser import PDFParser
from ..parser.chunking import TextChunker
from ..embeddings.embedding_model import EmbeddingModel
from .qdrant_client import QdrantClientWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookTrigger:
    def __init__(self, books_folder: str = "Books", collection_name: str = "law_books"):
        self.books_folder = Path(books_folder)
        self.collection_name = collection_name
        self.state_file = Path("book_ingestion_state.json")
        self.ingested_books = self._load_state()
        
        # Initialize components
        self.parser = None  # Will be initialized per book
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = EmbeddingModel()
        self.vectorstore = QdrantClientWrapper(collection_name)
        
        # Progress tracking
        self.current_book = None
        self.total_books = 0
        self.processed_books = 0
        
    def _load_state(self) -> Dict[str, Dict]:
        """Load the state of ingested books from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {}
    
    def _save_state(self):
        """Save the current state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.ingested_books, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state file: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for a file to detect changes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the books folder"""
        if not self.books_folder.exists():
            logger.error(f"Books folder {self.books_folder} does not exist")
            return []
        
        pdf_files = list(self.books_folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.books_folder}")
        return pdf_files
    
    def _is_book_ingested(self, file_path: Path) -> bool:
        """Check if a book has been ingested and hasn't changed"""
        file_hash = self._get_file_hash(file_path)
        book_name = file_path.stem
        
        if book_name in self.ingested_books:
            return self.ingested_books[book_name].get('file_hash') == file_hash
        
        return False
    
    def _mark_book_ingested(self, file_path: Path, stats: Dict):
        """Mark a book as ingested with metadata"""
        file_hash = self._get_file_hash(file_path)
        book_name = file_path.stem
        
        self.ingested_books[book_name] = {
            'file_path': str(file_path),
            'file_hash': file_hash,
            'ingested_at': datetime.now().isoformat(),
            'file_size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
            'stats': stats
        }
        self._save_state()
    
    def get_book_list(self) -> List[Dict]:
        """Get list of all books with their ingestion status"""
        pdf_files = self._get_pdf_files()
        books_info = []
        
        for file_path in pdf_files:
            book_name = file_path.stem
            file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            is_ingested = self._is_book_ingested(file_path)
            
            book_info = {
                'name': book_name,
                'file_path': str(file_path),
                'file_size_mb': file_size_mb,
                'is_ingested': is_ingested,
                'ingested_at': None,
                'stats': None
            }
            
            if is_ingested and book_name in self.ingested_books:
                book_info['ingested_at'] = self.ingested_books[book_name].get('ingested_at')
                book_info['stats'] = self.ingested_books[book_name].get('stats')
            
            books_info.append(book_info)
        
        return books_info
    
    def remove_book(self, book_name: str) -> bool:
        """Remove a book from the ingested list (doesn't delete the file)"""
        if book_name in self.ingested_books:
            del self.ingested_books[book_name]
            self._save_state()
            logger.info(f"Removed book '{book_name}' from ingested list")
            return True
        else:
            logger.warning(f"Book '{book_name}' not found in ingested list")
            return False
    
    def get_ingestion_stats(self) -> Dict:
        """Get overall ingestion statistics"""
        total_books = len(self._get_pdf_files())
        ingested_count = sum(1 for book in self.get_book_list() if book['is_ingested'])
        
        total_chunks = 0
        total_pages = 0
        
        for book_info in self.ingested_books.values():
            if 'stats' in book_info:
                total_chunks += book_info['stats'].get('total_chunks', 0)
                total_pages += book_info['stats'].get('total_pages', 0)
        
        return {
            'total_books': total_books,
            'ingested_books': ingested_count,
            'pending_books': total_books - ingested_count,
            'total_chunks': total_chunks,
            'total_pages': total_pages
        }
    
    def _process_single_book(self, file_path: Path) -> Dict:
        """Process a single book through the entire pipeline"""
        book_name = file_path.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing book: {book_name}")
        logger.info(f"File size: {round(file_path.stat().st_size / (1024 * 1024), 2)} MB")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        stats = {}
        
        try:
            # Step 1: Parse PDF
            logger.info(f"Step 1: Parsing PDF pages...")
            self.parser = PDFParser(str(file_path))
            pages = self.parser.parse()
            stats['total_pages'] = len(pages)
            logger.info(f"✓ Parsed {len(pages)} pages")
            
            # Step 2: Chunk text
            logger.info(f"Step 2: Chunking text...")
            chunks = self.chunker.chunk(pages)
            stats['total_chunks'] = len(chunks)
            logger.info(f"✓ Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info(f"Step 3: Generating embeddings...")
            chunk_texts = [chunk['chunk_text'] for chunk in chunks]
            embeddings = self.embedding_model.embed_texts(chunk_texts)
            logger.info(f"✓ Generated {len(embeddings)} embeddings")
            
            # Step 4: Prepare metadata for vectorstore
            logger.info(f"Step 4: Preparing metadata...")
            metadatas = []
            for chunk in chunks:
                metadata = {
                    'book_name': chunk['book_name'],
                    'page_number': chunk['page_number'],
                    'chunk_index': chunk['chunk_index'],
                    'chunk_text': chunk['chunk_text'],
                    'file_path': chunk['file_path'],
                    'ingested_at': datetime.now().isoformat()
                }
                metadatas.append(metadata)
            
            # Step 5: Upsert to vectorstore
            logger.info(f"Step 5: Upserting to vectorstore...")
            self.vectorstore.upsert_chunks(embeddings, metadatas)
            logger.info(f"✓ Upserted {len(embeddings)} chunks to vectorstore")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            stats['processing_time_seconds'] = round(processing_time, 2)
            stats['chunks_per_second'] = round(len(chunks) / processing_time, 2)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ Successfully processed '{book_name}'")
            logger.info(f"  - Pages: {stats['total_pages']}")
            logger.info(f"  - Chunks: {stats['total_chunks']}")
            logger.info(f"  - Processing time: {stats['processing_time_seconds']}s")
            logger.info(f"  - Rate: {stats['chunks_per_second']} chunks/s")
            logger.info(f"{'='*60}\n")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error processing book '{book_name}': {e}")
            raise
    
    def trigger_ingestion(self, force_reprocess: bool = False) -> Dict:
        """Main trigger function that processes all uningested books"""
        logger.info("🚀 Starting book ingestion trigger...")
        
        pdf_files = self._get_pdf_files()
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return {'status': 'no_files', 'processed': 0}
        
        # Filter books that need processing
        books_to_process = []
        for file_path in pdf_files:
            if force_reprocess or not self._is_book_ingested(file_path):
                books_to_process.append(file_path)
        
        if not books_to_process:
            logger.info("✓ All books are already ingested and up to date")
            return {'status': 'up_to_date', 'processed': 0}
        
        logger.info(f"📚 Found {len(books_to_process)} books to process")
        
        # Process each book
        processed_count = 0
        failed_books = []
        
        for i, file_path in enumerate(books_to_process, 1):
            try:
                logger.info(f"\n📖 Processing book {i}/{len(books_to_process)}")
                stats = self._process_single_book(file_path)
                self._mark_book_ingested(file_path, stats)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.stem}: {e}")
                failed_books.append(file_path.stem)
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Ingestion trigger completed!")
        logger.info(f"  - Successfully processed: {processed_count}")
        logger.info(f"  - Failed: {len(failed_books)}")
        if failed_books:
            logger.info(f"  - Failed books: {', '.join(failed_books)}")
        logger.info(f"{'='*60}")
        
        return {
            'status': 'completed',
            'processed': processed_count,
            'failed': len(failed_books),
            'failed_books': failed_books
        }
    
    def force_reprocess_all(self) -> Dict:
        """Force reprocess all books (useful for testing or after schema changes)"""
        logger.warning("⚠️  Force reprocessing all books...")
        return self.trigger_ingestion(force_reprocess=True)
    
    def get_pending_books(self) -> List[str]:
        """Get list of books that need to be processed"""
        pdf_files = self._get_pdf_files()
        pending = []
        
        for file_path in pdf_files:
            if not self._is_book_ingested(file_path):
                pending.append(file_path.stem)
        
        return pending 