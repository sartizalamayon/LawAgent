#!/usr/bin/env python3
"""
Book Ingestion Trigger Script

This script runs the book ingestion trigger to process all PDF books
in the Books folder and upload them to the vector database.

Usage:
    python scripts/run_book_trigger.py [--force] [--list] [--stats] [--remove BOOK_NAME]
"""

import sys
import os
import argparse
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vectorstore.book_trigger import BookTrigger
import logging

# Configure logging for better output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(description='Book Ingestion Trigger')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocess all books')
    parser.add_argument('--list', action='store_true',
                       help='List all books and their status')
    parser.add_argument('--stats', action='store_true',
                       help='Show ingestion statistics')
    parser.add_argument('--remove', type=str, metavar='BOOK_NAME',
                       help='Remove a book from the ingested list')
    parser.add_argument('--pending', action='store_true',
                       help='Show only pending books')
    
    args = parser.parse_args()
    
    # Initialize the trigger
    trigger = BookTrigger()
    
    try:
        if args.list:
            print("\n📚 Book List:")
            print("=" * 80)
            books = trigger.get_book_list()
            for book in books:
                status = "✓" if book['is_ingested'] else "⏳"
                print(f"{status} {book['name']} ({book['file_size_mb']} MB)")
                if book['is_ingested'] and book['stats']:
                    stats = book['stats']
                    print(f"   └─ Pages: {stats.get('total_pages', 0)}, "
                          f"Chunks: {stats.get('total_chunks', 0)}, "
                          f"Time: {stats.get('processing_time_seconds', 0)}s")
            print()
            
        elif args.stats:
            print("\n📊 Ingestion Statistics:")
            print("=" * 40)
            stats = trigger.get_ingestion_stats()
            print(f"Total Books: {stats['total_books']}")
            print(f"Ingested: {stats['ingested_books']}")
            print(f"Pending: {stats['pending_books']}")
            print(f"Total Pages: {stats['total_pages']}")
            print(f"Total Chunks: {stats['total_chunks']}")
            print()
            
        elif args.remove:
            success = trigger.remove_book(args.remove)
            if success:
                print(f"✓ Removed '{args.remove}' from ingested list")
            else:
                print(f"❌ Failed to remove '{args.remove}'")
                
        elif args.pending:
            pending = trigger.get_pending_books()
            if pending:
                print("\n⏳ Pending Books:")
                print("=" * 40)
                for book in pending:
                    print(f"• {book}")
            else:
                print("\n✓ No pending books - all books are ingested!")
            print()
            
        else:
            # Default: run the trigger
            print("🚀 Starting Book Ingestion Trigger...")
            print("=" * 60)
            
            if args.force:
                print("⚠️  Force mode enabled - will reprocess all books")
                result = trigger.force_reprocess_all()
            else:
                result = trigger.trigger_ingestion()
            
            print(f"\n🎉 Trigger completed with status: {result['status']}")
            if result['processed'] > 0:
                print(f"✓ Successfully processed: {result['processed']} books")
            if result['failed'] > 0:
                print(f"❌ Failed: {result['failed']} books")
                if result.get('failed_books'):
                    print(f"   Failed books: {', '.join(result['failed_books'])}")
                    
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 