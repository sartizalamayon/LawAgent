#!/usr/bin/env python3
"""
Simple test script for the fixed legal agent
"""

import asyncio
import os
from app.core.agent import get_agent_stream

async def test_agent():
    """Test the agent with a simple query"""
    print("🧪 Testing Legal Agent...")
    print("=" * 50)
    
    # Test query
    query = "What are the requirements for trademark registration in Bangladesh?"
    
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        async for event in get_agent_stream(query, thread_id="test_123"):
            event_type = event.get("type", "unknown")
            content = event.get("content", "")
            step = event.get("step", "unknown")
            
            if event_type == "status":
                print(f"📊 [{step}] {content}")
            elif event_type == "response":
                print(f"\n🤖 Final Response:")
                print(f"{content}")
                print("-" * 50)
            elif event_type == "complete":
                print(f"✅ {content}")
            elif event_type == "error":
                print(f"❌ Error: {content}")
                break
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure environment variables are set
    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set")
        exit(1)
    
    print("🚀 Starting agent test...")
    asyncio.run(test_agent())
    print("🏁 Test completed!") 