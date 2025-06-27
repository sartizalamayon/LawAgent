#!/usr/bin/env python3
"""
Test Bangla query streaming to show bilingual capabilities
"""

import asyncio
from app.core.agent import get_agent_stream

async def test_bangla_stream():
    """Test Bangla query streaming"""
    print("🚀 TESTING BANGLA QUERY STREAMING")
    print("=" * 50)
    
    query = "বাংলাদেশে ট্রেডমার্ক আইন কী?"
    thread_id = "bangla_test_001"
    
    print(f"Query: {query}")
    print(f"Thread ID: {thread_id}")
    print("\n🔄 STREAMING OUTPUT:")
    print("-" * 30)
    
    try:
        async for update in get_agent_stream(query, thread_id, language="bn"):
            # Pretty print each update
            if update["type"] == "status":
                print(f"📊 {update['content']}")
                if 'tool_name' in update:
                    print(f"   Tool: {update['tool_name']}")
            elif update["type"] == "response":
                print(f"\n🤖 FINAL RESPONSE:")
                print(f"   Language: {update['language']}")
                print(f"   Tools Used: {update['tools_used']}")
                print(f"   Documents Retrieved: {update['documents_retrieved']}")
                print(f"\n   Content: {update['content'][:300]}...")
            elif update["type"] == "complete":
                print(f"\n✅ {update['content']}")
            elif update["type"] == "error":
                print(f"\n❌ ERROR: {update['content']}")
            
            # Add small delay for readability
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_bangla_stream()) 