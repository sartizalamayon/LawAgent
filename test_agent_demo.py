#!/usr/bin/env python3
"""
Test script for the Advanced Bilingual Legal AI Agent
Demonstrates streaming, memory, tools, and bilingual capabilities
"""

import asyncio
import json
from app.core.agent import get_agent_stream, get_chat_history, clear_chat_history

async def test_english_query():
    """Test English legal query with streaming"""
    print("=" * 60)
    print("🧪 TESTING ENGLISH LEGAL QUERY")
    print("=" * 60)
    
    query = "What are the legal requirements for trademark registration in Bangladesh?"
    thread_id = "test_english_001"
    
    print(f"Query: {query}")
    print(f"Thread ID: {thread_id}")
    print("\n🔄 Streaming Response:")
    print("-" * 40)
    
    try:
        async for update in get_agent_stream(query, thread_id, language="en"):
            if update["type"] == "status":
                print(f"📊 {update['content']}")
            elif update["type"] == "response":
                print(f"\n🤖 FINAL RESPONSE:")
                print(f"Language: {update['language']}")
                print(f"Tools Used: {update['tools_used']}")
                print(f"Documents Retrieved: {update['documents_retrieved']}")
                print(f"\n{update['content']}")
            elif update["type"] == "complete":
                print(f"\n✅ {update['content']}")
            elif update["type"] == "error":
                print(f"❌ ERROR: {update['content']}")
            
            # Small delay for readability
            await asyncio.sleep(0.2)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

async def test_bangla_query():
    """Test Bangla legal query with streaming"""
    print("\n" + "=" * 60)
    print("🧪 TESTING BANGLA LEGAL QUERY")
    print("=" * 60)
    
    query = "বাংলাদেশে ট্রেডমার্ক নিবন্ধনের আইনি প্রয়োজনীয়তা কী?"
    thread_id = "test_bangla_001"
    
    print(f"Query: {query}")
    print(f"Thread ID: {thread_id}")
    print("\n🔄 Streaming Response:")
    print("-" * 40)
    
    try:
        async for update in get_agent_stream(query, thread_id, language="bn"):
            if update["type"] == "status":
                print(f"📊 {update['content']}")
            elif update["type"] == "response":
                print(f"\n🤖 FINAL RESPONSE:")
                print(f"Language: {update['language']}")
                print(f"Tools Used: {update['tools_used']}")
                print(f"Documents Retrieved: {update['documents_retrieved']}")
                print(f"\n{update['content']}")
            elif update["type"] == "complete":
                print(f"\n✅ {update['content']}")
            elif update["type"] == "error":
                print(f"❌ ERROR: {update['content']}")
            
            await asyncio.sleep(0.2)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

async def test_conversation_memory():
    """Test conversation memory across multiple queries"""
    print("\n" + "=" * 60)
    print("🧪 TESTING CONVERSATION MEMORY")
    print("=" * 60)
    
    thread_id = "test_memory_001"
    
    # First query
    print("📝 Query 1: What is contract law?")
    query1 = "What is contract law?"
    
    try:
        async for update in get_agent_stream(query1, thread_id, language="en"):
            if update["type"] == "response":
                print(f"Response: {update['content'][:100]}...")
                break
    except Exception as e:
        print(f"❌ Query 1 failed: {e}")
    
    # Second query (should remember context)
    print("\n📝 Query 2: Can you explain more about it?")
    query2 = "Can you explain more about it?"
    
    try:
        async for update in get_agent_stream(query2, thread_id, language="en"):
            if update["type"] == "response":
                print(f"Response: {update['content'][:100]}...")
                break
    except Exception as e:
        print(f"❌ Query 2 failed: {e}")
    
    # Show chat history
    print("\n📚 Chat History:")
    history = get_chat_history(thread_id)
    for i, msg in enumerate(history[-4:], 1):  # Show last 4 messages
        role = msg["role"]
        content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"{i}. {role.upper()}: {content}")

async def test_tool_selection():
    """Test different tool selections based on query types"""
    print("\n" + "=" * 60)
    print("🧪 TESTING TOOL SELECTION")
    print("=" * 60)
    
    test_queries = [
        ("What are my legal rights in a property dispute?", "property_rights"),
        ("Can you summarize this legal document?", "document_summary"),
        ("What type of legal case is this?", "case_classification"),
        ("I need legal advice about employment law", "employment_advice")
    ]
    
    for query, test_name in test_queries:
        print(f"\n🔧 Testing: {query}")
        thread_id = f"test_tools_{test_name}"
        
        try:
            async for update in get_agent_stream(query, thread_id, language="en"):
                if update["type"] == "response":
                    print(f"Tools Used: {update['tools_used']}")
                    print(f"Response: {update['content'][:80]}...")
                    break
        except Exception as e:
            print(f"❌ Test failed: {e}")

async def test_error_handling():
    """Test error handling with invalid queries"""
    print("\n" + "=" * 60)
    print("🧪 TESTING ERROR HANDLING")
    print("=" * 60)
    
    # Test with empty query
    print("📝 Testing empty query...")
    try:
        async for update in get_agent_stream("", "test_error_001", language="en"):
            if update["type"] == "error":
                print(f"✅ Error handled: {update['content']}")
                break
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test with very long query
    print("\n📝 Testing very long query...")
    long_query = "What is the law? " * 100  # Very long query
    try:
        async for update in get_agent_stream(long_query, "test_error_002", language="en"):
            if update["type"] == "response":
                print(f"✅ Long query handled: {len(update['content'])} characters")
                break
    except Exception as e:
        print(f"❌ Long query failed: {e}")

async def main():
    """Run all tests"""
    print("🚀 STARTING ADVANCED BILINGUAL LEGAL AI AGENT TESTS")
    print("=" * 80)
    
    # Run tests
    await test_english_query()
    await test_bangla_query()
    await test_conversation_memory()
    await test_tool_selection()
    await test_error_handling()
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 