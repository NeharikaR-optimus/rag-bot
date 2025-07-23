import streamlit as st
import requests
import json
import time
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🌍 RAG Travel Assistant with Streaming",
    page_icon="✈️",
    layout="wide"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def create_session() -> str:
    try:
        response = requests.post(f"{BACKEND_URL}/sessions", json={})
        if response.status_code == 200:
            return response.json()["session_id"]
        else:
            st.error(f"Failed to create session: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def send_chat_message(message: str, session_id: str, conversation_history: List = None) -> Dict:
    try:
        payload = {
            "message": message,
            "session_id": session_id,
            "conversation_history": conversation_history or []
        }
        
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {e}"}

def stream_chat_message(message: str, session_id: str, conversation_history: List = None):
    try:
        payload = {
            "message": message,
            "session_id": session_id,
            "conversation_history": conversation_history or []
        }
        
        response = requests.post(
            f"{BACKEND_URL}/chat/stream",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        if response.status_code == 200:
            return response
        else:
            st.error(f"Streaming error {response.status_code}: {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def parse_stream_data(line: str) -> Dict:
    try:
        if line.startswith("data: "):
            data_str = line[6:]  # Remove "data: " prefix
            return json.loads(data_str)
        return {}
    except json.JSONDecodeError:
        return {}

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Title and description
st.title("🌍 RAG Travel Assistant with Streaming")
st.markdown("*Your AI-powered travel companion with real-time streaming responses!*")

# Sidebar
with st.sidebar:
    st.header("🔧 Settings")
    
    # Session management
    if st.button("🆕 New Session", use_container_width=True):
        session_id = create_session()
        if session_id:
            st.session_state.session_id = session_id
            st.session_state.messages = []
            st.session_state.search_results = []
            st.success(f"New session created: {session_id[:8]}...")
    
    # Show current session
    if st.session_state.session_id:
        st.info(f"Session: {st.session_state.session_id[:8]}...")
    else:
        st.warning("No active session")
    
    # Streaming toggle
    use_streaming = st.toggle("🌊 Enable Streaming", value=True)
    
    st.divider()
    
    # Example questions
    st.subheader("💡 Try asking about:")
    example_questions = [
        "Hello!",
        "Tell me about Paris attractions",
        "Budget travel tips for Europe",
        "Family travel advice",
        "Weekly plan for France"
    ]
    
    for question in example_questions:
        if st.button(question, use_container_width=True, key=f"example_{question}"):
            st.session_state.example_question = question

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Create session if needed
    if not st.session_state.session_id:
        session_id = create_session()
        if session_id:
            st.session_state.session_id = session_id
            st.success(f"Session created: {session_id[:8]}...")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about travel...") or st.session_state.get("example_question"):
        if st.session_state.get("example_question"):
            prompt = st.session_state.example_question
            del st.session_state.example_question
        
        if not st.session_state.session_id:
            st.error("No active session. Please create a new session.")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Prepare conversation history for API
        conversation_history = []
        for msg in st.session_state.messages[:-1]:  # Exclude the current message
            conversation_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get bot response
        with chat_container:
            with st.chat_message("assistant"):
                if use_streaming:
                    # Streaming response
                    response_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    with st.spinner("🔍 Searching documents..."):
                        stream_response = stream_chat_message(
                            prompt, 
                            st.session_state.session_id, 
                            conversation_history
                        )
                    
                    if stream_response:
                        current_response = ""
                        search_results = []
                        final_response = ""
                        
                        try:
                            for line in stream_response.iter_lines(decode_unicode=True):
                                if line:
                                    data = parse_stream_data(line)
                                    
                                    if "status" in data:
                                        if data["status"] == "searching":
                                            status_placeholder.info("🔍 Searching documents...")
                                        elif data["status"] == "generating":
                                            status_placeholder.info("🤖 Generating response...")
                                        elif data["status"] == "completed":
                                            status_placeholder.success("✅ Response completed!")
                                            time.sleep(1)
                                            status_placeholder.empty()
                                    
                                    elif "partial_response" in data:
                                        current_response = data["partial_response"]
                                        response_placeholder.markdown(current_response)
                                    
                                    elif "final_response" in data:
                                        final_response = data["final_response"]
                                        response_placeholder.markdown(final_response)
                                        current_response = final_response
                                    
                                    elif "search_results" in data:
                                        search_results = data["search_results"]
                                        st.session_state.search_results = search_results
                                    
                                    elif "session_id" in data:
                                        st.session_state.session_id = data["session_id"]
                                    
                                    elif "error" in data:
                                        st.error(f"Error: {data['error']}")
                                        break
                        
                        except Exception as e:
                            st.error(f"Streaming error: {e}")
                            current_response = "Sorry, there was an error with the streaming response."
                        
                        # Add assistant message to chat
                        if current_response:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": current_response
                            })
                
                else:
                    # Regular response
                    with st.spinner("🤖 Getting response..."):
                        response_data = send_chat_message(
                            prompt, 
                            st.session_state.session_id, 
                            conversation_history
                        )
                    
                    if "error" in response_data:
                        st.error(response_data["error"])
                    else:
                        bot_response = response_data["response"]
                        st.markdown(bot_response)
                        
                        # Store search results
                        st.session_state.search_results = response_data.get("search_results", [])
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": bot_response
                        })

# Right column - Search Results
with col2:
    st.subheader("📚 Source Documents")
    
    if st.session_state.search_results:
        st.success(f"Found {len(st.session_state.search_results)} relevant documents")
        
        for i, result in enumerate(st.session_state.search_results[:3], 1):
            with st.expander(f"📄 {result.get('title', f'Document {i}')} (Score: {result.get('score', 0):.3f})"):
                st.write(f"**Source:** {result.get('source', 'Unknown')}")
                content = result.get('content', '')
                if len(content) > 300:
                    st.write(f"**Content:** {content[:300]}...")
                    if st.button(f"Show full content {i}", key=f"show_full_{i}"):
                        st.write(content)
                else:
                    st.write(f"**Content:** {content}")
    else:
        st.info("No search results yet. Send a message to see relevant documents!")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.search_results = []
        st.rerun()

with col2:
    # Check backend health
    if st.button("🏥 Health Check", use_container_width=True):
        try:
            response = requests.get(f"{BACKEND_URL}/health")
            if response.status_code == 200:
                st.success("Backend is healthy! ✅")
            else:
                st.error(f"Backend error: {response.status_code}")
        except:
            st.error("Cannot connect to backend ❌")

with col3:
    st.markdown("*Powered by LangGraph + Azure AI*")

# Auto-scroll to bottom
if st.session_state.messages:
    st.markdown(
        """
        <script>
        var element = document.querySelector('.main');
        element.scrollTop = element.scrollHeight;
        </script>
        """, 
        unsafe_allow_html=True
    )
