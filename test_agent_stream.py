import asyncio
from app.core.agent import get_agent_stream

async def test_agent_stream():
    print("Testing agent streaming...")
    try:
        async for update in get_agent_stream("Hello, how are you?"):
            print(f"Update: {update}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent_stream()) 