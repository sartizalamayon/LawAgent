import asyncio
import websockets
import json
import pytest

@pytest.mark.asyncio
async def test_agent_ws():
    uri = "ws://localhost:8000/agent/chat"
    async with websockets.connect(uri) as websocket:
        # Send a test query
        await websocket.send(json.dumps({"query": "What is the constitution of Bangladesh?"}))
        print("Sent query. Streaming responses:")
        try:
            while True:
                response = await websocket.recv()
                print(response)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket closed.")

if __name__ == "__main__":
    asyncio.run(test_agent_ws()) 