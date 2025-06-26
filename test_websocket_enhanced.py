import asyncio
import websockets
import json

async def test_enhanced_websocket():
    uri = "ws://localhost:8000/agent/chat"
    
    print("🚀 Testing Enhanced WebSocket Agent")
    print("=" * 50)
    
    async with websockets.connect(uri) as websocket:
        # Send the Bangla query
        query_data = {
            "query": """নাম ট্রেড মার্ক সম্পর্কে কিছু বিষয় জানার ছিলো।

আমরা ১ বছর আগে ট্রেড লাইসেন্স করে BTCL থেকে সরকারি নিয়ম অনুযায়ী একটা ডোমেইন কিনে ছিলাম।  বিসনেস শুরু করছিলাম এখন এক কোম্পানি এসে বলে যে এই নামে তাদের ট্রেড মার্ক তারা করছে। 

আমরা এই বিষয় জানার পর ট্রেড মার্ক সার্চ করার জন্য শিল্প মন্ত্রনালয় গিয়েছিলাম কিন্তু তাদের তথ্য জানা হয়নি। তাদের সার্ভিস না কি বর্তমানে বন্ধ আছে।

কিন্তু যে কোম্পানি বলছে ট্রেড মার্ক করছে তারা আমাদের ভয় ভীতি দেখাচ্ছে পুলিশ - ডিবি দিয়ে আমাদের ধরাবে। পুলিশ আমাদের কয়েক বার কল দিয়েছে।

আমরা যেহেতু সরকারি নিয়ম অনুযায়ী ডোমেইন টি কিনেছি  এখন আমাদের করণীয় কি?""",
            "thread_id": "trademark_case_ws_001"
        }
        
        await websocket.send(json.dumps(query_data))
        print("📤 Sent query to WebSocket")
        print("\n📥 Receiving streaming responses:")
        print("-" * 50)
        
        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                # Pretty print based on type
                update_type = data.get("type", "unknown")
                content = data.get("content", "")
                
                if update_type == "analysis":
                    print(f"🔍 {content}")
                elif update_type == "tool_call":
                    tool_name = data.get("tool_name", "unknown")
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
                    break
                elif update_type == "error":
                    print(f"❌ {content}")
                    break
                
                print("-" * 30)
                
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")

if __name__ == "__main__":
    asyncio.run(test_enhanced_websocket()) 