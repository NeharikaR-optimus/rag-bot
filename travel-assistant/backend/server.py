import os
import uuid
from typing import Dict, Any, Optional, AsyncGenerator
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

from config import Config
from utils.prompts import LOG_MESSAGES, API_MESSAGES

config = Config()

app = FastAPI(
    title="Travel Assistant API",
    description="A RAG-based travel assistant using Azure AI services",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class StreamChunk(BaseModel):
    content: str
    done: bool = False
    session_id: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Travel Assistant API is running"}

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    async def generate_stream():
        try:
            session_id = request.session_id or str(uuid.uuid4())
            
            from tools.doc_search_tool import DocumentManagementTool
            from tools.chat_history_tool import ChatHistoryManager
            from tools.llm_tool import LLMTool
            
            doc_tool = DocumentManagementTool()
            history_tool = ChatHistoryManager()
            llm_tool = LLMTool()
            
            conversation_history = []
            try:
                conversation_history = await history_tool.get_conversation_history(session_id, limit=5)
            except Exception as e:
                print(f"Warning: Could not retrieve chat history: {e}")
                conversation_history = []
            
            context = ""
            try:
                search_results = await doc_tool.semantic_search(request.message)
                if search_results:
                    context = f"Relevant travel information:\n{search_results}\n\n"
            except Exception as e:
                print(f"Warning: Could not retrieve search context: {e}")
                context = ""
            
            full_response = ""
            async for chunk in llm_tool.generate_streaming_response(
                query=request.message,
                context=context,
                conversation_history=conversation_history
            ):
                if chunk:
                    full_response += chunk
                    chunk_data = StreamChunk(
                        content=chunk,
                        done=False,
                        session_id=session_id
                    )
                    yield f"data: {chunk_data.model_dump_json()}\n\n"
                    await asyncio.sleep(0.01)
            
            try:
                await history_tool.store_message(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=full_response,
                    metadata={"streaming": True}
                )
            except Exception as e:
                print(f"Warning: Could not store chat history: {e}")
            
            completion_chunk = StreamChunk(
                content="",
                done=True,
                session_id=session_id
            )
            yield f"data: {completion_chunk.model_dump_json()}\n\n"
            
        except Exception as e:
            print(f"Error in streaming chat: {str(e)}")
            error_chunk = StreamChunk(
                content=f"Error: {str(e)}",
                done=True,
                session_id=session_id
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        from tools.doc_search_tool import DocumentManagementTool
        from tools.chat_history_tool import ChatHistoryManager
        from tools.llm_tool import LLMTool
        
        doc_tool = DocumentManagementTool()
        history_tool = ChatHistoryManager()
        llm_tool = LLMTool()
        
        conversation_history = []
        try:
            conversation_history = await history_tool.get_conversation_history(session_id, limit=5)
        except Exception:
            conversation_history = []
        
        context = ""
        try:
            search_results = await doc_tool.semantic_search(request.message)
            if search_results:
                context = f"Relevant travel information:\n{search_results}\n\n"
        except Exception:
            context = ""
        
        response_text = await llm_tool.generate_response(
            query=request.message,
            context=context,
            conversation_history=conversation_history
        )
        
        try:
            await history_tool.store_message(
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                metadata={"streaming": False}
            )
        except Exception:
            pass
        
        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Travel Assistant API",
        "version": "1.0.0",
        "description": "RAG-based travel assistant",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8080,
        reload=True
    )
