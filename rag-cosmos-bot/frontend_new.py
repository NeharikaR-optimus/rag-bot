import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import time
import asyncio
import httpx

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="🌍 RAG Travel Chatbot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .bot-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    .streaming-text {
        color: #1f77b4;
        font-weight: 500;
    }
    .sidebar-content {
        padding: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_health():
    """Check if backend is running and healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Backend returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def send_message_non_streaming(message):
    """Send message to backend (non-streaming)"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": message},
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()["response"]
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

async def send_message_streaming(message):
    """Send message to backend with streaming response"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/chat/stream",
                json={"message": message},
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data.strip():
                                try:
                                    parsed = json.loads(data)
                                    if parsed.get("type") == "chunk":
                                        yield parsed["chunk"]
                                    elif parsed.get("type") == "complete":
                                        break
                                except json.JSONDecodeError:
                                    continue
                else:
                    yield f"Error: {response.status_code}"
    except Exception as e:
        yield f"Streaming error: {str(e)}"

def clear_memory():
    """Clear conversation memory"""
    try:
        response = requests.post(f"{BACKEND_URL}/memory/clear", timeout=10)
        if response.status_code == 200:
            return True, "Memory cleared successfully"
        else:
            return False, f"Error clearing memory: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

def get_document_count():
    """Get document count from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents/count", timeout=10)
        if response.status_code == 200:
            return True, response.json()["document_count"]
        else:
            return False, f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

def reload_documents():
    """Reload documents in backend"""
    try:
        response = requests.post(f"{BACKEND_URL}/documents/reload", timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = False

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌍 RAG Travel Chatbot</h1>
    <p>Your AI-powered travel companion with real-time streaming responses</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings & Status")
    
    # Backend health check
    if st.button("🔄 Check Backend Health"):
        with st.spinner("Checking backend..."):
            healthy, info = check_backend_health()
            st.session_state.backend_healthy = healthy
            
            if healthy:
                st.markdown('<p class="status-success">✅ Backend is healthy</p>', unsafe_allow_html=True)
                st.json(info)
            else:
                st.markdown('<p class="status-error">❌ Backend is not accessible</p>', unsafe_allow_html=True)
                st.error(info.get("error", "Unknown error"))
    
    # Document management
    st.markdown("### 📚 Document Management")
    
    if st.button("📊 Get Document Count"):
        with st.spinner("Getting document count..."):
            success, result = get_document_count()
            if success:
                st.success(f"📄 Documents in knowledge base: {result}")
            else:
                st.error(f"Error: {result}")
    
    if st.button("🔄 Reload Documents"):
        with st.spinner("Reloading documents..."):
            success, result = reload_documents()
            if success:
                st.success(f"✅ {result['message']}")
                st.info(f"📄 Document count: {result['document_count']}")
            else:
                st.error(f"Error: {result}")
    
    # Memory management
    st.markdown("### 🧠 Memory Management")
    
    if st.button("🗑️ Clear Conversation Memory"):
        with st.spinner("Clearing memory..."):
            success, result = clear_memory()
            if success:
                st.success(result)
                st.session_state.messages = []
            else:
                st.error(result)
    
    # Streaming settings
    st.markdown("### 🚀 Response Settings")
    use_streaming = st.checkbox("Enable Streaming Responses", value=True)
    
    # Information
    st.markdown("### ℹ️ Information")
    st.info("""
    **Features:**
    - 🔄 Real-time streaming responses
    - 🗄️ Azure Cosmos DB storage
    - 🔍 Azure AI Search integration
    - 🧠 Conversation memory
    - 📚 Multi-document knowledge base
    
    **Available Topics:**
    - European travel (Paris, London, Rome)
    - Asian destinations (Japan, Thailand, China, India)
    - Adventure travel and activities
    - Budget and family travel tips
    """)

# Main chat interface
st.markdown("## 💬 Chat with Your Travel Assistant")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋 You:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Travel Assistant:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me about travel destinations, tips, or anything travel-related!"):
    # Check backend health before sending message
    if not st.session_state.backend_healthy:
        healthy, _ = check_backend_health()
        st.session_state.backend_healthy = healthy
        
        if not healthy:
            st.error("❌ Backend is not accessible. Please check if the backend server is running.")
            st.stop()
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    st.markdown(f"""
    <div class="chat-message user-message">
        <strong>🙋 You:</strong> {prompt}
    </div>
    """, unsafe_allow_html=True)
    
    # Get bot response
    with st.spinner("🤖 Thinking..."):
        if use_streaming:
            # Streaming response
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Run async streaming function
                async def stream_response():
                    nonlocal full_response
                    async for chunk in send_message_streaming(prompt):
                        full_response += chunk
                        response_placeholder.markdown(f"""
                        <div class="chat-message bot-message">
                            <strong>🤖 Travel Assistant:</strong> 
                            <span class="streaming-text">{full_response}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        await asyncio.sleep(0.01)  # Small delay for visual effect
                
                # Run the streaming function
                asyncio.run(stream_response())
                
                if full_response:
                    # Final display without streaming styling
                    response_placeholder.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Travel Assistant:</strong> {full_response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to session state
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("No response received from the server.")
                    
            except Exception as e:
                st.error(f"Streaming error: {str(e)}")
        else:
            # Non-streaming response
            success, response = send_message_non_streaming(prompt)
            
            if success:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Travel Assistant:</strong> {response}
                </div>
                """, unsafe_allow_html=True)
                
                # Add to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error(f"Error: {response}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌍 RAG Travel Chatbot | Powered by Azure OpenAI, Cosmos DB & AI Search</p>
    <p>Backend URL: <code>{}</code></p>
</div>
""".format(BACKEND_URL), unsafe_allow_html=True)
