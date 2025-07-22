# simple_main.py - FastAPI Backend for the Simple Chatbot

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
import os

# Import your simple chatbot initialization function
from simple_chatbot import initialize_simple_chatbot_components

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simple Travel Chatbot Backend",
    description="Backend API for a simple travel chatbot using Azure OpenAI (no RAG).",
    version="1.0.0",
)

# Global variables to hold the chain and memory
# These will be initialized once when the app starts
chat_chain = None
memory = None

# --- Pydantic Model for Request Body ---
# This defines the structure of the JSON data your frontend will send
class ChatRequest(BaseModel):
    user_message: str

# --- FastAPI Event Handlers for Startup and Shutdown ---
@app.on_event("startup")
async def startup_event():
    """
    Initialize simple chatbot components when the FastAPI application starts.
    This ensures the LLM is loaded only once.
    """
    global chat_chain, memory
    print("FastAPI app starting up...")
    try:
        chat_chain, memory = initialize_simple_chatbot_components()
        print("Simple chatbot components initialized successfully for the backend.")
    except Exception as e:
        print(f"Failed to initialize simple chatbot components: {e}")
        # Re-raise the exception to prevent the server from starting if init fails
        raise HTTPException(status_code=500, detail=f"Backend initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Perform any cleanup when the FastAPI application shuts down.
    """
    print("FastAPI app shutting down.")
    # No specific cleanup needed for LangChain in-memory components here,
    # but you would close database connections, etc., if they were persistent.


# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """
    Root endpoint for health check.
    """
    return {"message": "Simple Travel Chatbot Backend is running!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat messages from the frontend, processes them with the simple chat chain,
    and returns the bot's response.
    """
    if chat_chain is None or memory is None:
        raise HTTPException(status_code=503, detail="Chatbot components not initialized.")

    user_message = request.user_message
    print(f"Received user message: {user_message}")

    # Retrieve current chat history from LangChain memory
    current_chat_history = memory.load_memory_variables({})["chat_history"]

    # Prepare input for the chat chain
    chain_input = {"input": user_message, "chat_history": current_chat_history}

    try:
        # Get the response from the chat chain
        # We'll collect the streaming response and return it as a complete message
        full_response = ""
        async for chunk in chat_chain.astream(chain_input):
            full_response += chunk
        
        print(f"Generated bot response: {full_response}")

        # Update LangChain memory with the latest turn
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(full_response)

        return {"response": full_response}

    except Exception as e:
        print(f"Error during chat processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

# To run this backend:
# Make sure you are in the same directory as simple_main.py and simple_chatbot.py
# Run: uvicorn simple_main:app --reload --port 8000
# The --reload flag is useful for development as it restarts the server on code changes.
# --port 8000 sets the port.

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
