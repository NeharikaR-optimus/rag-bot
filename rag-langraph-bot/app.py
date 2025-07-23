from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import os
from contextlib import asynccontextmanager
from chatbot import RAGLangGraphBot
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Global bot instance
bot_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    global bot_instance
    try:
        print("🚀 Starting RAG Travel Assistant API with Streaming...")
        bot_instance = RAGLangGraphBot()
        print("✅ RAG Travel Assistant API started successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize bot: {e}")
        raise
    
    yield
    
    # Cleanup
    if bot_instance:
        try:
            await bot_instance.close()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

# FastAPI app
app = FastAPI(
    title="RAG Travel Assistant API with Streaming",
    description="Travel Assistant API using LangGraph and Azure AI Search with streaming responses",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    search_results: Optional[List[Dict[str, Any]]] = []
    session_id: str

class StreamChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = {}

class SessionResponse(BaseModel):
    session_id: str

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Travel Assistant API with Streaming",
        "status": "running",
        "version": "1.0.0",
        "features": ["chat", "streaming", "azure-ai-search", "cosmos-db"],
        "bot_initialized": bot_instance is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "bot_initialized": bot_instance is not None,
        "timestamp": "2025-01-22T12:00:00Z"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Regular chat endpoint (non-streaming)"""
    try:
        if not bot_instance:
            raise HTTPException(status_code=500, detail="Bot not initialized")
        
        print(f"📝 Received chat request: {request.message[:50]}...")
        
        # Convert conversation history to LangChain messages
        conversation = []
        for msg in request.conversation_history:
            if msg.get("role") == "user":
                conversation.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                conversation.append(AIMessage(content=msg.get("content", "")))
        
        # Generate session ID if not provided
        session_id = request.session_id
        if not session_id:
            session_id = await bot_instance.start_new_session()
            print(f"🆔 Created new session: {session_id[:8]}...")
        
        # Get response from bot
        print("🤖 Getting response from bot...")
        response = await bot_instance.chat(
            user_input=request.message,
            conversation_history=conversation,
            session_id=session_id
        )
        
        # Get search results for debugging
        search_results = await bot_instance.get_search_results_async(request.message)
        
        print(f"✅ Bot response generated: {len(response)} characters")
        print(f"🔍 Found {len(search_results)} search results")
        
        return ChatResponse(
            response=response,
            search_results=search_results,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

async def stream_chat_response(
    message: str, 
    conversation_history: List = None, 
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """Stream chat response word by word"""
    try:
        if not bot_instance:
            yield f"data: {json.dumps({'error': 'Bot not initialized'})}\n\n"
            return
        
        # Convert conversation history to LangChain messages
        conversation = conversation_history or []
        
        # Generate session ID if not provided
        if not session_id:
            session_id = await bot_instance.start_new_session()
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        
        # Send status update
        yield f"data: {json.dumps({'status': 'searching'})}\n\n"
        
        # Get the full response first
        response = await bot_instance.chat(
            user_input=message,
            conversation_history=conversation,
            session_id=session_id
        )
        
        # Get search results
        search_results = await bot_instance.get_search_results_async(message)
        
        # Send search results
        yield f"data: {json.dumps({'search_results': search_results})}\n\n"
        
        # Send status update
        yield f"data: {json.dumps({'status': 'generating'})}\n\n"
        
        # Stream the response word by word
        words = response.split()
        current_text = ""
        
        for i, word in enumerate(words):
            current_text += word + " "
            
            # Send partial response
            yield f"data: {json.dumps({'partial_response': current_text.strip()})}\n\n"
            
            # Add a small delay to simulate streaming
            await asyncio.sleep(0.05)  # 50ms delay between words
        
        # Send final response
        yield f"data: {json.dumps({'final_response': response, 'session_id': session_id})}\n\n"
        yield f"data: {json.dumps({'status': 'completed'})}\n\n"
        
    except Exception as e:
        print(f"❌ Error in stream_chat_response: {e}")
        yield f"data: {json.dumps({'error': f'Stream error: {str(e)}'})}\n\n"

@app.post("/chat/stream")
async def stream_chat(request: StreamChatRequest):
    """Streaming chat endpoint"""
    try:
        print(f"🌊 Received streaming chat request: {request.message[:50]}...")
        
        # Convert conversation history to LangChain messages
        conversation = []
        for msg in request.conversation_history:
            if msg.get("role") == "user":
                conversation.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                conversation.append(AIMessage(content=msg.get("content", "")))
        
        # Create the streaming response
        return StreamingResponse(
            stream_chat_response(
                message=request.message,
                conversation_history=conversation,
                session_id=request.session_id
            ),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
    except Exception as e:
        print(f"❌ Error in streaming chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming chat error: {str(e)}")

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new chat session"""
    try:
        if not bot_instance:
            raise HTTPException(status_code=500, detail="Bot not initialized")
        
        session_id = await bot_instance.start_new_session(request.metadata)
        print(f"🆔 Created session: {session_id[:8]}...")
        
        return SessionResponse(session_id=session_id)
        
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation error: {str(e)}")

@app.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str, limit: int = 10):
    """Get conversation history for a session"""
    try:
        if not bot_instance:
            raise HTTPException(status_code=500, detail="Bot not initialized")
        
        history = await bot_instance.get_conversation_history(session_id, limit)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        print(f"❌ Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents endpoint"""
    try:
        if not bot_instance:
            raise HTTPException(status_code=500, detail="Bot not initialized")
        
        results = await bot_instance.get_search_results_async(request.query)
        return SearchResponse(results=results)
        
    except Exception as e:
        print(f"❌ Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
    HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT = int(os.getenv("BACKEND_PORT", "8000"))
    
    print("🌟 Starting RAG Travel Assistant Backend with Streaming...")
    print(f"🌐 Host: {HOST}")
    print(f"🔌 Port: {PORT}")
    print("📚 Features: LangGraph + Azure AI Search + Cosmos DB + Streaming")
    print("-" * 70)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )
