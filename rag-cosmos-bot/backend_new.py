import os
import asyncio
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from dotenv import load_dotenv

# Import the RAG chatbot
from rag_bot import RAGChatBot

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Travel Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = RAGChatBot()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup"""
    print("🚀 Starting RAG Travel Chatbot API...")
    success = chatbot.initialize()
    if not success:
        raise RuntimeError("Failed to initialize chatbot")
    print("✅ Chatbot initialized successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG Travel Chatbot API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "chatbot_ready": chatbot.setup_complete,
        "version": "1.0.0"
    }

@app.post("/chat")
async def chat_non_streaming(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint"""
    try:
        if not chatbot.setup_complete:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        response = await chatbot.get_response(request.message)
        return ChatResponse(response=response, status="success")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat/stream")
async def chat_streaming(request: ChatRequest):
    """Streaming chat endpoint"""
    try:
        if not chatbot.setup_complete:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        async def generate_response() -> AsyncIterator[str]:
            """Generate streaming response"""
            async for chunk in chatbot.get_response_streaming(request.message):
                # Format as SSE (Server-Sent Events)
                yield f"data: {json.dumps({'chunk': chunk, 'type': 'chunk'})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating streaming response: {str(e)}")

@app.post("/memory/clear")
async def clear_memory():
    """Clear conversation memory"""
    try:
        chatbot.clear_memory()
        return {"message": "Memory cleared successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

@app.get("/documents/count")
async def get_document_count():
    """Get the number of documents in the knowledge base"""
    try:
        # Get documents from Cosmos DB
        documents = chatbot._get_documents_from_cosmos()
        return {
            "document_count": len(documents),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document count: {str(e)}")

@app.post("/documents/reload")
async def reload_documents():
    """Reload documents from source files"""
    try:
        # Re-setup the RAG pipeline to reload documents
        chatbot._setup_rag()
        documents = chatbot._get_documents_from_cosmos()
        return {
            "message": "Documents reloaded successfully",
            "document_count": len(documents),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    
    print(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
