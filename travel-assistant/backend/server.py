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
from workflows.travel_workflow import TravelAssistantWorkflow

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
    """Streaming chat endpoint using LangGraph workflow."""
    async def generate_stream():
        try:
            session_id = request.session_id or str(uuid.uuid4())
            
            # Get the compiled workflow
            workflow = await TravelAssistantWorkflow.invoke_graph_workflow({})
            
            # Prepare the input state
            config_dict = {"configurable": {"thread_id": session_id}}
            
            # Create the message for the workflow
            from langchain_core.messages import HumanMessage
            input_state = {
                "messages": [HumanMessage(content=request.message)]
            }
            
            # Stream the workflow execution
            full_response = ""
            
            async for event in workflow.astream(input_state, config=config_dict):
                # Extract response content from workflow events
                for node_name, node_output in event.items():
                    if "messages" in node_output and node_output["messages"]:
                        last_message = node_output["messages"][-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            # Stream the content incrementally
                            content = last_message.content
                            if content not in full_response:
                                new_content = content[len(full_response):]
                                full_response = content
                                
                                if new_content:
                                    chunk_data = StreamChunk(
                                        content=new_content,
                                        done=False,
                                        session_id=session_id
                                    )
                                    yield f"data: {chunk_data.model_dump_json()}\n\n"
                                    await asyncio.sleep(0.01)
            
            # Send completion signal
            completion_chunk = StreamChunk(
                content="",
                done=True,
                session_id=session_id
            )
            yield f"data: {completion_chunk.model_dump_json()}\n\n"
            
        except Exception as e:
            print(f"Error in LangGraph streaming chat: {str(e)}")
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
    """Main chat endpoint using LangGraph workflow."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get the compiled workflow
        workflow = await TravelAssistantWorkflow.invoke_graph_workflow({})
        
        # Prepare the input state
        config_dict = {"configurable": {"thread_id": session_id}}
        
        # Create the message for the workflow
        from langchain_core.messages import HumanMessage
        input_state = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        # Execute the workflow
        final_state = await workflow.ainvoke(input_state, config=config_dict)
        
        # Extract the response from the final state
        response_text = ""
        if "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, 'content'):
                response_text = last_message.content
        
        if not response_text:
            response_text = "I apologize, but I couldn't generate a response. Please try again."
        
        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Error processing LangGraph chat: {str(e)}")
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
