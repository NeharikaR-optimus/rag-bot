import streamlit as st
import requests
import uuid
import json

st.set_page_config(
    page_title="Travel Assistant",
    page_icon="🌍",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

def call_chat_api_streaming(message: str, placeholder):
    """Call the streaming chat API and update the placeholder in real-time."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json={
                "message": message,
                "session_id": st.session_state.session_id
            },
            stream=True,
            timeout=60
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        chunk_data = json.loads(line[6:])
                        if chunk_data.get("content"):
                            full_response += chunk_data["content"]
                            placeholder.markdown(full_response + "▋")
                        if chunk_data.get("done"):
                            placeholder.markdown(full_response)
                            break
                    except json.JSONDecodeError:
                        continue
            return full_response if full_response else "Sorry, I couldn't process your request."
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Cannot connect to the server. Please make sure the backend is running on port 8000."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def call_chat_api(message: str) -> str:
    """Call the chat API and return the response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Sorry, I couldn't process your request.")
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Cannot connect to the server. Please make sure the backend is running on port 8000."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"

st.title("Travel Assistant")
st.markdown("Ask me anything about travel destinations, tips, or planning advice!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about travel destinations, tips, or planning advice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = call_chat_api_streaming(prompt, placeholder)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
