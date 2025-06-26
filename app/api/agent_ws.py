import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.agent import get_agent_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent"])

@router.websocket("/chat")
async def agent_chat_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        while True:
            # Receive the message
            data = await websocket.receive_json()
            user_query = data.get("query")
            thread_id = data.get("thread_id", "default")
            
            if not user_query:
                await websocket.send_json({"error": "No query provided"})
                continue
                
            logger.info(f"Received query: {user_query}")
            
            # Stream agent responses with detailed steps
            try:
                async for update in get_agent_stream(user_query, thread_id):
                    # Send the detailed update
                    await websocket.send_json(update)
                    logger.debug(f"Sent update: {update}")
                    
                    # Add a small delay for better streaming experience
                    import asyncio
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in agent stream: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"Agent error: {str(e)}",
                    "step": "error"
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"WebSocket error: {str(e)}",
                "step": "error"
            })
        except:
            pass 