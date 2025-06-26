from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import logging
import os
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from app.vectorstore.retriever import BookRetriever

router = APIRouter(prefix="/retriever", tags=["Book Retriever"])
logger = logging.getLogger(__name__)

# Initialize custom BookRetriever
book_retriever = BookRetriever(collection_name="law_books")

# Also keep LangChain retriever for basic search
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "law_books"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACE_API_KEY,
    model_name=EMBEDDING_MODEL
)
lc_qdrant = Qdrant(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)
lc_retriever = lc_qdrant.as_retriever(search_kwargs={"k": 5})

@router.get("/search")
async def search_books(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return")
):
    """Search for relevant book chunks using custom BookRetriever"""
    try:
        results = book_retriever.search(query, top_k=top_k)
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/langchain")
async def search_books_langchain(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return")
):
    """Search for relevant book chunks using LangChain Qdrant retriever"""
    try:
        docs = lc_retriever.get_relevant_documents(query)
        results = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs[:top_k]
        ]
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error during LangChain search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/book/{book_name}")
async def search_by_book(
    book_name: str,
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return")
):
    """Search within a specific book"""
    try:
        results = book_retriever.search_by_book(book_name, query, top_k)
        return {
            "book_name": book_name,
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error during book search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/page/{book_name}/{page_number}")
async def search_by_page(
    book_name: str,
    page_number: int,
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return")
):
    """Search within a specific page of a book"""
    try:
        results = book_retriever.search_by_page(book_name, page_number, query, top_k)
        return {
            "book_name": book_name,
            "page_number": page_number,
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error during page search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hybrid")
async def hybrid_search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return"),
    semantic_weight: float = Query(0.7, description="Weight for semantic search"),
    keyword_weight: float = Query(0.3, description="Weight for keyword search")
):
    """Hybrid search combining semantic and keyword matching"""
    try:
        results = book_retriever.hybrid_search(
            query=query,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "semantic_weight": semantic_weight,
            "keyword_weight": keyword_weight
        }
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/books/{book_name}/chunks")
async def get_book_chunks(
    book_name: str,
    limit: int = Query(100, description="Maximum number of chunks to return")
):
    """Get all chunks from a specific book"""
    try:
        chunks = book_retriever.get_book_chunks(book_name, limit)
        return {
            "book_name": book_name,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error getting book chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_collection_stats():
    """Get statistics about the collection"""
    try:
        stats = book_retriever.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/books")
async def get_available_books():
    """Get list of all available books in the collection"""
    try:
        stats = book_retriever.get_collection_stats()
        return {
            "books": stats.get("book_names", []),
            "total_books": stats.get("unique_books", 0),
            "total_chunks": stats.get("total_chunks", 0)
        }
    except Exception as e:
        logger.error(f"Error getting available books: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 