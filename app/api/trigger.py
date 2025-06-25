from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict
import logging

from ..vectorstore.book_trigger import BookTrigger

router = APIRouter(prefix="/trigger", tags=["Book Ingestion Trigger"])
logger = logging.getLogger(__name__)

# Global trigger instance
trigger = BookTrigger()

@router.get("/books", response_model=List[Dict])
async def get_book_list():
    """Get list of all books with their ingestion status"""
    try:
        return trigger.get_book_list()
    except Exception as e:
        logger.error(f"Error getting book list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=Dict)
async def get_ingestion_stats():
    """Get overall ingestion statistics"""
    try:
        return trigger.get_ingestion_stats()
    except Exception as e:
        logger.error(f"Error getting ingestion stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pending", response_model=List[str])
async def get_pending_books():
    """Get list of books that need to be processed"""
    try:
        return trigger.get_pending_books()
    except Exception as e:
        logger.error(f"Error getting pending books: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/books/{book_name}")
async def remove_book(book_name: str):
    """Remove a book from the ingested list"""
    try:
        success = trigger.remove_book(book_name)
        if success:
            return {"message": f"Successfully removed '{book_name}' from ingested list"}
        else:
            raise HTTPException(status_code=404, detail=f"Book '{book_name}' not found in ingested list")
    except Exception as e:
        logger.error(f"Error removing book {book_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks, force: bool = False):
    """Trigger book ingestion (runs in background)"""
    try:
        if force:
            logger.info("Force reprocessing all books")
            result = trigger.force_reprocess_all()
        else:
            logger.info("Processing pending books")
            result = trigger.trigger_ingestion()
        
        return {
            "message": "Ingestion triggered successfully",
            "status": result['status'],
            "processed": result['processed'],
            "failed": result['failed'],
            "failed_books": result.get('failed_books', [])
        }
    except Exception as e:
        logger.error(f"Error triggering ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/force")
async def force_reprocess_all():
    """Force reprocess all books"""
    try:
        result = trigger.force_reprocess_all()
        return {
            "message": "Force reprocessing completed",
            "status": result['status'],
            "processed": result['processed'],
            "failed": result['failed'],
            "failed_books": result.get('failed_books', [])
        }
    except Exception as e:
        logger.error(f"Error force reprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 