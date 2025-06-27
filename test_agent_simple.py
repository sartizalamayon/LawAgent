#!/usr/bin/env python3
"""
Simple test to show detailed streaming output from the agent
"""

import asyncio
import json
from app.core.agent import get_agent_stream

async def test_simple_stream():
    """Test simple streaming with detailed output"""
    print("🚀 TESTING AGENT STREAMING")
    print("=" * 50)
    
    query = "What is trademark law in Bangladesh?"
    thread_id = "simple_test_001"
    
    print(f"Query: {query}")
    print(f"Thread ID: {thread_id}")
    print("\n🔄 STREAMING OUTPUT:")
    print("-" * 30)
    
    try:
        async for update in get_agent_stream(query, thread_id, language="en"):
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
                print(f"\n   Content: {update['content'][:200]}...")
            elif update["type"] == "complete":
                print(f"\n✅ {update['content']}")
            elif update["type"] == "error":
                print(f"\n❌ ERROR: {update['content']}")
            
            # Add small delay for readability
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_stream()) 