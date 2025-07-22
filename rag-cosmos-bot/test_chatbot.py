import asyncio
from rag_bot import RAGChatBot

async def test_chatbot():
    """Simple test to verify the chatbot works"""
    print("🧪 Testing RAG Travel Chatbot...")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = RAGChatBot()
    
    if not chatbot.initialize():
        print("❌ Failed to initialize chatbot")
        return
    
    # Test questions
    test_questions = [
        "What are the top attractions in Paris?",
        "Can you recommend family-friendly hotels?",
        "What's the best way to get around Paris?",
        "Tell me about dining options in Paris."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("🤖 Response: ", end="")
        
        try:
            response = await chatbot.get_response(question)
            print(response[:200] + "..." if len(response) > 200 else response)
            print("✅ Success!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    asyncio.run(test_chatbot())
