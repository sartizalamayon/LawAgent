#!/usr/bin/env python3
"""
Test script for the Book Retriever

This script tests various search functionalities of the retriever
to ensure it's working correctly with the ingested books.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vectorstore.retriever import BookRetriever
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_search():
    """Test basic search functionality"""
    print("\n🔍 Testing Basic Search...")
    print("=" * 50)
    
    retriever = BookRetriever()
    
    # Test queries
    test_queries = [
        "constitutional rights",
        "human rights",
        "legal procedure",
        "evidence law",
        "criminal justice",
        "বাংলাদেশের সংবিধান",  # Bangladesh Constitution in Bangla
        "মানবাধিকার",  # Human rights in Bangla
        "আইনী প্রক্রিয়া"  # Legal procedure in Bangla
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = retriever.search(query, top_k=3)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Book: {result['book_name']}")
                print(f"     Page: {result['page_number']}")
                print(f"     Text: {result['chunk_text'][:100]}...")
        except Exception as e:
            print(f"Error: {e}")

def test_book_specific_search():
    """Test search within specific books"""
    print("\n📚 Testing Book-Specific Search...")
    print("=" * 50)
    
    retriever = BookRetriever()
    
    # Get available books
    try:
        stats = retriever.get_collection_stats()
        books = stats.get("book_names", [])
        print(f"Available books: {books}")
        
        if books:
            # Test search in first book
            book_name = books[0]
            print(f"\nSearching in book: {book_name}")
            
            results = retriever.search_by_book(book_name, "constitution", top_k=3)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Page: {result['page_number']}")
                print(f"     Text: {result['chunk_text'][:100]}...")
                
    except Exception as e:
        print(f"Error: {e}")

def test_hybrid_search():
    """Test hybrid search functionality"""
    print("\n🔀 Testing Hybrid Search...")
    print("=" * 50)
    
    retriever = BookRetriever()
    
    test_queries = [
        "constitutional law",
        "evidence matters",
        "fair trial rights"
    ]
    
    for query in test_queries:
        print(f"\nHybrid search for: '{query}'")
        try:
            results = retriever.hybrid_search(query, top_k=3)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Combined Score: {result.get('combined_score', result['score']):.3f}")
                print(f"     Book: {result['book_name']}")
                print(f"     Page: {result['page_number']}")
                print(f"     Text: {result['chunk_text'][:100]}...")
        except Exception as e:
            print(f"Error: {e}")

def test_collection_stats():
    """Test collection statistics"""
    print("\n📊 Testing Collection Statistics...")
    print("=" * 50)
    
    retriever = BookRetriever()
    
    try:
        stats = retriever.get_collection_stats()
        print("Collection Statistics:")
        print(f"  Collection Name: {stats['collection_name']}")
        print(f"  Total Chunks: {stats['total_chunks']}")
        print(f"  Unique Books: {stats['unique_books']}")
        print(f"  Vector Size: {stats['vector_size']}")
        print(f"  Distance Metric: {stats['distance']}")
        print(f"  Books: {stats['book_names']}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_book_chunks():
    """Test getting chunks from a specific book"""
    print("\n📖 Testing Book Chunks...")
    print("=" * 50)
    
    retriever = BookRetriever()
    
    try:
        stats = retriever.get_collection_stats()
        books = stats.get("book_names", [])
        
        if books:
            book_name = books[0]
            print(f"Getting chunks from: {book_name}")
            
            chunks = retriever.get_book_chunks(book_name, limit=5)
            print(f"Found {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. Page {chunk['page_number']}, Chunk {chunk['chunk_index']}")
                print(f"     Text: {chunk['chunk_text'][:100]}...")
                
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all tests"""
    print("🧪 Starting Retriever Tests...")
    print("=" * 60)
    
    try:
        # Test collection stats first
        test_collection_stats()
        
        # Test basic search
        test_basic_search()
        
        # Test book-specific search
        test_book_specific_search()
        
        # Test hybrid search
        test_hybrid_search()
        
        # Test getting book chunks
        test_book_chunks()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 