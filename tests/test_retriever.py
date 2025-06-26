#!/usr/bin/env python3
"""
Test script for the retriever functionality
Tests both the custom BookRetriever class and the FastAPI endpoints
"""

import os
import sys
import asyncio
import requests
import json
from typing import Dict, List
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from vectorstore.retriever import BookRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/retriever"

class RetrieverTester:
    def __init__(self):
        self.book_retriever = BookRetriever(collection_name="law_books")
        self.test_queries = [
            "contract law",
            "property rights",
            "criminal procedure",
            "constitutional law",
            "family law",
            "বাংলাদেশের আইন",  # Bangladeshi law in Bangla
            "চুক্তি আইন",      # Contract law in Bangla
        ]
        
    def test_book_retriever_direct(self):
        """Test the BookRetriever class directly"""
        logger.info("🧪 Testing BookRetriever class directly...")
        
        try:
            # Test 1: Basic search
            logger.info("Testing basic search...")
            results = self.book_retriever.search("contract law", top_k=3)
            logger.info(f"✅ Basic search returned {len(results)} results")
            if results:
                logger.info(f"First result: {results[0].get('book_name', 'N/A')} - Score: {results[0].get('score', 'N/A')}")
            
            # Test 2: Collection stats
            logger.info("Testing collection stats...")
            stats = self.book_retriever.get_collection_stats()
            logger.info(f"✅ Collection stats: {stats.get('total_chunks', 0)} chunks, {stats.get('unique_books', 0)} books")
            
            # Test 3: Available books
            if stats.get('book_names'):
                test_book = stats['book_names'][0]
                logger.info(f"Testing book-specific search with: {test_book}")
                
                # Test search by book
                book_results = self.book_retriever.search_by_book(test_book, "law", top_k=2)
                logger.info(f"✅ Book search returned {len(book_results)} results")
                
                # Test get book chunks
                chunks = self.book_retriever.get_book_chunks(test_book, limit=5)
                logger.info(f"✅ Book chunks returned {len(chunks)} chunks")
            
            # Test 4: Hybrid search
            logger.info("Testing hybrid search...")
            hybrid_results = self.book_retriever.hybrid_search("contract", top_k=3)
            logger.info(f"✅ Hybrid search returned {len(hybrid_results)} results")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ BookRetriever test failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test the FastAPI endpoints"""
        logger.info("🧪 Testing FastAPI endpoints...")
        
        try:
            # Test 1: Basic search endpoint
            logger.info("Testing /search endpoint...")
            response = requests.get(f"{API_BASE}/search", params={"query": "contract law", "top_k": 3})
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ /search returned {data.get('total_results', 0)} results")
            else:
                logger.error(f"❌ /search failed: {response.status_code} - {response.text}")
                return False
            
            # Test 2: LangChain search endpoint
            logger.info("Testing /search/langchain endpoint...")
            response = requests.get(f"{API_BASE}/search/langchain", params={"query": "property rights", "top_k": 3})
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ /search/langchain returned {data.get('total_results', 0)} results")
            else:
                logger.error(f"❌ /search/langchain failed: {response.status_code} - {response.text}")
                return False
            
            # Test 3: Collection stats endpoint
            logger.info("Testing /stats endpoint...")
            response = requests.get(f"{API_BASE}/stats")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ /stats returned: {data.get('total_chunks', 0)} chunks, {data.get('unique_books', 0)} books")
            else:
                logger.error(f"❌ /stats failed: {response.status_code} - {response.text}")
                return False
            
            # Test 4: Available books endpoint
            logger.info("Testing /books endpoint...")
            response = requests.get(f"{API_BASE}/books")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ /books returned {data.get('total_books', 0)} books")
                if data.get('books'):
                    logger.info(f"Sample books: {data['books'][:3]}")
            else:
                logger.error(f"❌ /books failed: {response.status_code} - {response.text}")
                return False
            
            # Test 5: Book-specific search (if books exist)
            books_response = requests.get(f"{API_BASE}/books")
            if books_response.status_code == 200:
                books_data = books_response.json()
                if books_data.get('books'):
                    test_book = books_data['books'][0]
                    logger.info(f"Testing /search/book/{test_book} endpoint...")
                    response = requests.get(f"{API_BASE}/search/book/{test_book}", params={"query": "law", "top_k": 2})
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"✅ Book search returned {data.get('total_results', 0)} results")
                    else:
                        logger.error(f"❌ Book search failed: {response.status_code} - {response.text}")
            
            # Test 6: Hybrid search endpoint
            logger.info("Testing /hybrid endpoint...")
            response = requests.get(f"{API_BASE}/hybrid", params={
                "query": "contract",
                "top_k": 3,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3
            })
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ /hybrid returned {data.get('total_results', 0)} results")
            else:
                logger.error(f"❌ /hybrid failed: {response.status_code} - {response.text}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ API test failed: {e}")
            return False
    
    def test_multilingual_queries(self):
        """Test multilingual query support"""
        logger.info("🧪 Testing multilingual queries...")
        
        try:
            # Test Bangla queries
            bangla_queries = ["বাংলাদেশের আইন", "চুক্তি আইন", "সম্পত্তি অধিকার"]
            
            for query in bangla_queries:
                logger.info(f"Testing Bangla query: {query}")
                
                # Test direct retriever
                results = self.book_retriever.search(query, top_k=2)
                logger.info(f"✅ Bangla query '{query}' returned {len(results)} results")
                
                # Test API endpoint
                response = requests.get(f"{API_BASE}/search", params={"query": query, "top_k": 2})
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ API Bangla query returned {data.get('total_results', 0)} results")
                else:
                    logger.error(f"❌ API Bangla query failed: {response.status_code}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Multilingual test failed: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        logger.info("🧪 Testing error handling...")
        
        try:
            # Test 1: Invalid query
            response = requests.get(f"{API_BASE}/search", params={"query": "", "top_k": 5})
            logger.info(f"Empty query response: {response.status_code}")
            
            # Test 2: Invalid top_k
            response = requests.get(f"{API_BASE}/search", params={"query": "test", "top_k": -1})
            logger.info(f"Invalid top_k response: {response.status_code}")
            
            # Test 3: Non-existent book
            response = requests.get(f"{API_BASE}/search/book/nonexistent_book", params={"query": "test"})
            logger.info(f"Non-existent book response: {response.status_code}")
            
            logger.info("✅ Error handling tests completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting comprehensive retriever tests...")
        
        tests = [
            ("BookRetriever Direct", self.test_book_retriever_direct),
            ("API Endpoints", self.test_api_endpoints),
            ("Multilingual Queries", self.test_multilingual_queries),
            ("Error Handling", self.test_error_handling),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {test_name} test...")
            logger.info(f"{'='*50}")
            
            try:
                success = test_func()
                results[test_name] = "✅ PASSED" if success else "❌ FAILED"
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                results[test_name] = "💥 CRASHED"
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*50}")
        
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

def main():
    """Main test runner"""
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code != 200:
            logger.error("❌ FastAPI server is not running. Please start the server first.")
            logger.info("Run: uvicorn app.main:app --reload")
            return False
    except requests.exceptions.RequestException:
        logger.error("❌ Cannot connect to FastAPI server. Please start the server first.")
        logger.info("Run: uvicorn app.main:app --reload")
        return False
    
    # Check environment variables
    required_env_vars = ["QDRANT_URL", "QDRANT_API_KEY", "HUGGINGFACE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    # Run tests
    tester = RetrieverTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 