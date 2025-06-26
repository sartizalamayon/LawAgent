import asyncio
import json
from app.core.agent import get_agent_stream

async def test_enhanced_agent():
    print("🚀 Enhanced Agent Test - Trademark Query")
    print("=" * 60)
    
    # Your Bangla query
    query = """নাম ট্রেড মার্ক সম্পর্কে কিছু বিষয় জানার ছিলো।

আমরা ১ বছর আগে ট্রেড লাইসেন্স করে BTCL থেকে সরকারি নিয়ম অনুযায়ী একটা ডোমেইন কিনে ছিলাম।  বিসনেস শুরু করছিলাম এখন এক কোম্পানি এসে বলে যে এই নামে তাদের ট্রেড মার্ক তারা করছে। 

আমরা এই বিষয় জানার পর ট্রেড মার্ক সার্চ করার জন্য শিল্প মন্ত্রনালয় গিয়েছিলাম কিন্তু তাদের তথ্য জানা হয়নি। তাদের সার্ভিস না কি বর্তমানে বন্ধ আছে।

কিন্তু যে কোম্পানি বলছে ট্রেড মার্ক করছে তারা আমাদের ভয় ভীতি দেখাচ্ছে পুলিশ - ডিবি দিয়ে আমাদের ধরাবে। পুলিশ আমাদের কয়েক বার কল দিয়েছে।

আমরা যেহেতু সরকারি নিয়ম অনুযায়ী ডোমেইন টি কিনেছি  এখন আমাদের করণীয় কি?"""
    
    print(f"📝 Query: {query[:100]}...")
    print("\n🔄 Starting Enhanced Agent Stream:")
    print("-" * 60)
    
    try:
        async for update in get_agent_stream(query, "trademark_case_001"):
            # Pretty print the update based on type
            update_type = update.get("type", "unknown")
            content = update.get("content", "")
            step = update.get("step", "unknown")
            
            if update_type == "analysis":
                print(f"🔍 {content}")
            elif update_type == "tool_call":
                tool_name = update.get("tool_name", "unknown")
                print(f"🛠️  {content}")
                print(f"   📋 Tool: {tool_name}")
            elif update_type == "tool_result":
                print(f"📊 {content}")
            elif update_type == "reasoning":
                print(f"🧠 {content}")
            elif update_type == "response":
                print(f"💬 {content}")
            elif update_type == "completion":
                print(f"✅ {content}")
            elif update_type == "error":
                print(f"❌ {content}")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_agent()) 