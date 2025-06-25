from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import logging

from ..vectorstore.retriever import BookRetriever

router = APIRouter(prefix="/retriever", tags=["Book Retriever"])
logger = logging.getLogger(__name__)

# Global retriever instance
retriever = BookRetriever()

@router.get("/search")
async def search_books(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return"),
    score_threshold: float = Query(0.7, description="Minimum similarity score"),
    book_filter: Optional[str] = Query(None, description="Filter by book name"),
    page_filter: Optional[int] = Query(None, description="Filter by page number")
):
    """Search for relevant book chunks"""
    try:
        results = retriever.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            book_filter=book_filter,
            page_filter=page_filter
        )
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/book/{book_name}")
async def search_by_book(
    book_name: str,
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return")
):
    """Search within a specific book"""
    try:
        results = retriever.search_by_book(book_name, query, top_k)
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
        results = retriever.search_by_page(book_name, page_number, query, top_k)
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
        results = retriever.hybrid_search(
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
        chunks = retriever.get_book_chunks(book_name, limit)
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
        stats = retriever.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/books")
async def get_available_books():
    """Get list of all available books in the collection"""
    try:
        stats = retriever.get_collection_stats()
        return {
            "books": stats.get("book_names", []),
            "total_books": stats.get("unique_books", 0),
            "total_chunks": stats.get("total_chunks", 0)
        }
    except Exception as e:
        logger.error(f"Error getting available books: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 