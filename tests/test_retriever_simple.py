#!/usr/bin/env python3
"""
Simple test script for the BookRetriever class
Does not require FastAPI server to be running
"""

import os
import sys
import logging
from typing import Dict, List

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from vectorstore.retriever import BookRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic BookRetriever functionality"""
    logger.info("🧪 Testing basic BookRetriever functionality...")
    
    try:
        # Initialize retriever
        retriever = BookRetriever(collection_name="law_books")
        logger.info("✅ BookRetriever initialized successfully")
        
        # Test collection stats
        logger.info("Testing collection stats...")
        stats = retriever.get_collection_stats()
        logger.info(f"✅ Collection stats retrieved:")
        logger.info(f"   - Total chunks: {stats.get('total_chunks', 0)}")
        logger.info(f"   - Unique books: {stats.get('unique_books', 0)}")
        logger.info(f"   - Vector size: {stats.get('vector_size', 'N/A')}")
        logger.info(f"   - Distance metric: {stats.get('distance', 'N/A')}")
        
        if stats.get('book_names'):
            logger.info(f"   - Sample books: {stats['book_names'][:3]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

def test_search_functionality():
    """Test search functionality"""
    logger.info("🧪 Testing search functionality...")
    
    try:
        retriever = BookRetriever(collection_name="law_books")
        
        # Test queries
        test_queries = [
            "contract law",
            "property rights", 
            "criminal procedure",
            "বাংলাদেশের আইন",  # Bangladeshi law in Bangla
            "চুক্তি আইন",      # Contract law in Bangla
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            results = retriever.search(query, top_k=3)
            logger.info(f"✅ Query '{query}' returned {len(results)} results")
            
            if results:
                first_result = results[0]
                logger.info(f"   - Best match: {first_result.get('book_name', 'N/A')}")
                logger.info(f"   - Score: {first_result.get('score', 'N/A'):.4f}")
                logger.info(f"   - Page: {first_result.get('page_number', 'N/A')}")
                logger.info(f"   - Chunk preview: {first_result.get('chunk_text', '')[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Search functionality test failed: {e}")
        return False

def test_book_specific_search():
    """Test book-specific search functionality"""
    logger.info("🧪 Testing book-specific search...")
    
    try:
        retriever = BookRetriever(collection_name="law_books")
        
        # Get available books
        stats = retriever.get_collection_stats()
        if not stats.get('book_names'):
            logger.warning("⚠️  No books available for book-specific testing")
            return True
        
        test_book = stats['book_names'][0]
        logger.info(f"Testing search within book: {test_book}")
        
        # Test search by book
        results = retriever.search_by_book(test_book, "law", top_k=3)
        logger.info(f"✅ Book search returned {len(results)} results")
        
        if results:
            logger.info(f"   - All results from book: {test_book}")
            for i, result in enumerate(results):
                logger.info(f"   - Result {i+1}: Page {result.get('page_number', 'N/A')}, Score: {result.get('score', 'N/A'):.4f}")
        
        # Test get book chunks
        chunks = retriever.get_book_chunks(test_book, limit=5)
        logger.info(f"✅ Retrieved {len(chunks)} chunks from book")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Book-specific search test failed: {e}")
        return False

def test_hybrid_search():
    """Test hybrid search functionality"""
    logger.info("🧪 Testing hybrid search...")
    
    try:
        retriever = BookRetriever(collection_name="law_books")
        
        # Test hybrid search with different weights
        test_configs = [
            {"semantic_weight": 0.7, "keyword_weight": 0.3},
            {"semantic_weight": 0.5, "keyword_weight": 0.5},
            {"semantic_weight": 0.3, "keyword_weight": 0.7},
        ]
        
        for config in test_configs:
            logger.info(f"Testing hybrid search with weights: {config}")
            results = retriever.hybrid_search("contract", top_k=3, **config)
            logger.info(f"✅ Hybrid search returned {len(results)} results")
            
            if results:
                first_result = results[0]
                logger.info(f"   - Best match: {first_result.get('book_name', 'N/A')}")
                logger.info(f"   - Score: {first_result.get('score', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid search test failed: {e}")
        return False

def test_multilingual_support():
    """Test multilingual query support"""
    logger.info("🧪 Testing multilingual support...")
    
    try:
        retriever = BookRetriever(collection_name="law_books")
        
        # Test Bangla queries
        bangla_queries = [
            "বাংলাদেশের আইন",      # Bangladeshi law
            "চুক্তি আইন",          # Contract law
            "সম্পত্তি অধিকার",      # Property rights
            "ফৌজদারি আইন",         # Criminal law
            "পারিবারিক আইন",        # Family law
        ]
        
        for query in bangla_queries:
            logger.info(f"Testing Bangla query: '{query}'")
            results = retriever.search(query, top_k=2)
            logger.info(f"✅ Bangla query returned {len(results)} results")
            
            if results:
                logger.info(f"   - Best match: {results[0].get('book_name', 'N/A')}")
                logger.info(f"   - Score: {results[0].get('score', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multilingual support test failed: {e}")
        return False

def main():
    """Main test runner"""
    logger.info("🚀 Starting BookRetriever tests...")
    
    # Check environment variables
    required_env_vars = ["QDRANT_URL", "QDRANT_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.info("Please set QDRANT_URL and QDRANT_API_KEY environment variables")
        return False
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Search Functionality", test_search_functionality),
        ("Book-Specific Search", test_book_specific_search),
        ("Hybrid Search", test_hybrid_search),
        ("Multilingual Support", test_multilingual_support),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "💥 CRASHED"
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    passed = sum(1 for result in results.values() if "PASSED" in result)
    total = len(results)
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 