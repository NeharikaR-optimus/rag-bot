import requests
import json
import time

def test_streaming_chat():
    """Test the streaming chat functionality"""
    print("🌊 Testing Streaming Chat Functionality")
    print("=" * 50)
    
    # Test backend health
    print("🏥 Checking backend health...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Backend is healthy!")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    print("\n🆔 Creating new session...")
    # Create a new session
    try:
        session_response = requests.post("http://localhost:8000/sessions", json={})
        if session_response.status_code == 200:
            session_id = session_response.json()["session_id"]
            print(f"✅ Session created: {session_id[:8]}...")
        else:
            print(f"❌ Failed to create session: {session_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return
    
    print("\n🌊 Testing streaming response...")
    # Test streaming chat
    test_message = "Hello! Tell me about Paris attractions"
    
    try:
        payload = {
            "message": test_message,
            "session_id": session_id,
            "conversation_history": []
        }
        
        print(f"📝 Sending message: '{test_message}'")
        print("📡 Streaming response:")
        print("-" * 30)
        
        response = requests.post(
            "http://localhost:8000/chat/stream",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        if response.status_code == 200:
            current_response = ""
            search_results_count = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    try:
                        data_str = line[6:]  # Remove "data: " prefix
                        data = json.loads(data_str)
                        
                        if "status" in data:
                            print(f"📊 Status: {data['status']}")
                        
                        elif "session_id" in data:
                            print(f"🆔 Session ID confirmed: {data['session_id'][:8]}...")
                        
                        elif "search_results" in data:
                            search_results_count = len(data["search_results"])
                            print(f"🔍 Found {search_results_count} search results")
                            for i, result in enumerate(data["search_results"][:2], 1):
                                print(f"   {i}. {result.get('title', 'Unknown')} (Score: {result.get('score', 0):.3f})")
                        
                        elif "partial_response" in data:
                            current_response = data["partial_response"]
                            # Print only the new part (last few words)
                            words = current_response.split()
                            if len(words) <= 3:
                                print(f"🤖 {current_response}", end=" ", flush=True)
                            else:
                                print(f"{words[-1]}", end=" ", flush=True)
                        
                        elif "final_response" in data:
                            print(f"\n\n✅ Final Response:")
                            print(f"   {data['final_response']}")
                        
                        elif "error" in data:
                            print(f"❌ Error: {data['error']}")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error: {e}")
                        continue
            
            print(f"\n📊 Streaming test completed!")
            print(f"   Search results: {search_results_count}")
            print(f"   Response length: {len(current_response)} characters")
        
        else:
            print(f"❌ Streaming request failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Streaming test error: {e}")

def test_regular_chat():
    """Test the regular (non-streaming) chat functionality"""
    print("\n\n📝 Testing Regular Chat Functionality")
    print("=" * 50)
    
    # Create session
    session_response = requests.post("http://localhost:8000/sessions", json={})
    session_id = session_response.json()["session_id"]
    
    # Test regular chat
    test_message = "Budget travel tips for Europe"
    payload = {
        "message": test_message,
        "session_id": session_id,
        "conversation_history": []
    }
    
    print(f"📝 Sending message: '{test_message}'")
    
    start_time = time.time()
    response = requests.post("http://localhost:8000/chat", json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Regular chat response received in {end_time - start_time:.2f}s")
        print(f"   Response: {data['response'][:100]}...")
        print(f"   Search results: {len(data.get('search_results', []))}")
    else:
        print(f"❌ Regular chat failed: {response.status_code}")

if __name__ == "__main__":
    print("🧪 RAG Travel Assistant - Streaming Test Suite")
    print("🌐 Backend URL: http://localhost:8000")
    print("🖥️ Frontend URL: http://localhost:8505")
    print("=" * 60)
    
    # Test streaming
    test_streaming_chat()
    
    # Test regular chat for comparison
    test_regular_chat()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("🚀 You can now open the frontend at: http://localhost:8505")
    print("🌊 Make sure to enable streaming in the sidebar!")
