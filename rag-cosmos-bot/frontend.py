import streamlit as st
import requests
import time
import json

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
CLEAR_MEMORY_ENDPOINT = f"{BACKEND_URL}/clear-memory"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Paris Travel Assistant",
    page_icon="🗼",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #4ECDC4;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .status-healthy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def check_backend_health():
    """Check if the backend is running and healthy"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "ready", data.get("message", "")
        else:
            return False, f"Backend returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend. Please ensure the backend server is running."
    except requests.exceptions.Timeout:
        return False, "Backend connection timed out."
    except Exception as e:
        return False, f"Error checking backend: {str(e)}"

def send_message_to_backend(user_message):
    """Send message to backend and get response"""
    try:
        payload = {"user_message": user_message}
        response = requests.post(
            CHAT_ENDPOINT, 
            json=payload, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get("bot_response", "No response received")
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return False, f"Backend error: {error_detail}"
            
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend. Please ensure the backend server is running."
    except requests.exceptions.Timeout:
        return False, "Request timed out. The backend might be processing a complex query."
    except Exception as e:
        return False, f"Error sending message: {str(e)}"

def clear_backend_memory():
    """Clear the backend chatbot memory"""
    try:
        response = requests.post(CLEAR_MEMORY_ENDPOINT, timeout=10)
        if response.status_code == 200:
            return True, "Memory cleared successfully!"
        else:
            return False, "Failed to clear memory"
    except Exception as e:
        return False, f"Error clearing memory: {str(e)}"

# --- Main App ---
def main():
    # Header
    st.markdown('<h1 class="main-header">🗼 Paris Travel Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-powered guide to exploring Paris!</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # Health check
        if st.button("🔍 Check Backend Status"):
            with st.spinner("Checking backend..."):
                is_healthy, message = check_backend_health()
                if is_healthy:
                    st.success("✅ Backend is ready!")
                else:
                    st.error(f"❌ {message}")
        
        # Clear memory
        if st.button("🧹 Clear Chat Memory"):
            with st.spinner("Clearing memory..."):
                success, message = clear_backend_memory()
                if success:
                    st.success(message)
                    # Clear frontend session state as well
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Hello! I'm your Paris travel assistant. How can I help you plan your trip to the City of Light? 🗼"}
                    ]
                    st.rerun()
                else:
                    st.error(message)
        
        # Instructions
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask about Paris attractions
        - Get hotel recommendations
        - Learn about family activities
        - Discover dining options
        - Get transportation tips
        """)
    
    # Check backend status on load
    is_healthy, health_message = check_backend_health()
    
    if is_healthy:
        st.markdown(f'<div class="status-box status-healthy">✅ Backend Status: {health_message}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-box status-error">❌ Backend Status: {health_message}</div>', 
                   unsafe_allow_html=True)
        st.warning("⚠️ Please start the backend server first!")
        st.code("python backend.py", language="bash")
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Paris travel assistant. How can I help you plan your trip to the City of Light? 🗼"}
        ]
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user", avatar="🙋"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask me about Paris travel..."):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        with st.chat_message("user", avatar="🙋"):
            st.write(user_input)
        
        # Get bot response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                success, bot_response = send_message_to_backend(user_input)
                
                if success:
                    st.write(bot_response)
                    # Add bot response to session state
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                else:
                    error_message = f"❌ {bot_response}"
                    st.error(error_message)
                    # Add error to session state
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Setup Instructions ---
def show_setup_instructions():
    st.markdown("---")
    st.markdown("### 🚀 Setup Instructions")
    
    with st.expander("Click to see setup instructions"):
        st.markdown("""
        **To run the RAG Travel Chatbot:**
        
        1. **Install Dependencies:**
           ```bash
           cd rag-cosmos-bot
           pip install -r requirements.txt
           ```
        
        2. **Configure Environment:**
           - Make sure your `.env` file has the correct Azure OpenAI credentials
        
        3. **Start the Backend Server:**
           ```bash
           python backend.py
           ```
           - Backend will run on http://127.0.0.1:8000
           - API docs available at http://127.0.0.1:8000/docs
        
        4. **Start the Frontend (this app):**
           ```bash
           streamlit run frontend.py
           ```
        
        5. **Test the Chatbot:**
           - Try asking: "What are the top attractions in Paris?"
           - Or: "Recommend family-friendly hotels in Paris"
        """)

if __name__ == "__main__":
    main()
    show_setup_instructions()
