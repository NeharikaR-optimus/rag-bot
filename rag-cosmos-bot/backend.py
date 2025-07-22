from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
from rag_bot import initialize_chatbot, get_chatbot_response

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG Travel Chatbot Backend",
    description="Backend API for RAG-powered travel chatbot using Azure OpenAI",
    version="2.0.0",
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    bot_response: str
    status: str = "success"

class StatusResponse(BaseModel):
    message: str
    status: str

# --- Global State ---
chatbot_initialized = False

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot when the server starts"""
    global chatbot_initialized
    print("🚀 Starting RAG Travel Chatbot Backend...")
    
    try:
        chatbot_initialized = initialize_chatbot()
        if chatbot_initialized:
            print("✅ Chatbot initialized successfully!")
        else:
            print("❌ Failed to initialize chatbot")
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        chatbot_initialized = False

# --- API Endpoints ---
@app.get("/", response_model=StatusResponse)
async def root():
    """Health check endpoint"""
    return StatusResponse(
        message="RAG Travel Chatbot Backend is running!",
        status="healthy" if chatbot_initialized else "unhealthy"
    )

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Detailed health check"""
    if chatbot_initialized:
        return StatusResponse(
            message="Chatbot is ready to answer your travel questions!",
            status="ready"
        )
    else:
        return StatusResponse(
            message="Chatbot is not initialized properly",
            status="error"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot_initialized:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not initialized. Please check server logs."
        )
    
    try:
        user_message = request.user_message.strip()
        
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="User message cannot be empty"
            )
        
        # Get response from the chatbot
        bot_response = await get_chatbot_response(user_message)
        
        return ChatResponse(
            bot_response=bot_response,
            status="success"
        )
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your request: {str(e)}"
        )

@app.post("/clear-memory")
async def clear_memory():
    """Clear chatbot memory"""
    if not chatbot_initialized:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not initialized"
        )
    
    try:
        from rag_bot import chatbot
        chatbot.clear_memory()
        return StatusResponse(
            message="Memory cleared successfully!",
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing memory: {str(e)}"
        )

# --- Error Handlers ---
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status": "error"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status": "error"}

# --- Main execution ---
if __name__ == "__main__":
    print("🌟 Starting RAG Travel Chatbot Backend Server...")
    print("📚 This chatbot uses RAG to answer questions about Paris travel!")
    print("🔗 Backend will be available at: http://127.0.0.1:8000")
    print("📖 API docs available at: http://127.0.0.1:8000/docs")
    print("-" * 60)
    
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
