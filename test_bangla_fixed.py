#!/usr/bin/env python3
"""
Bangla test script for the fixed legal agent
"""

import asyncio
import os
from app.core.agent import get_agent_stream

async def test_bangla_agent():
    """Test the agent with a Bangla query"""
    print("🧪 Testing Legal Agent with Bangla Query...")
    print("=" * 50)
    
    # Test query in Bangla
    query = "বাংলাদেশে ট্রেডমার্ক নিবন্ধনের জন্য কী কী প্রয়োজনীয়তা আছে?"
    
    print(f"Query (Bangla): {query}")
    print("-" * 50)
    
    try:
        async for event in get_agent_stream(query, thread_id="bangla_test_456", language="bn"):
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
    
    print("🚀 Starting Bangla agent test...")
    asyncio.run(test_bangla_agent())
    print("🏁 Test completed!") 