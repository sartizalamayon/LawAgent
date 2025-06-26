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
            
            if not user_query:
                await websocket.send_json({"error": "No query provided"})
                continue
                
            logger.info(f"Received query: {user_query}")
            
            # Stream agent responses
            try:
                async for update in get_agent_stream(user_query):
                    await websocket.send_json(update)
                    logger.debug(f"Sent update: {update}")
                    
                # Send completion signal
                await websocket.send_json({"status": "completed"})
                logger.info("Agent response completed")
                
            except Exception as e:
                logger.error(f"Error in agent stream: {e}")
                await websocket.send_json({"error": f"Agent error: {str(e)}"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": f"WebSocket error: {str(e)}"})
        except:
            pass 