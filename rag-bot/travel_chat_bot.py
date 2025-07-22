# travel_chat_bot.py - Streamlit Frontend

import streamlit as st
import requests
import time

# --- Configuration ---
FASTAPI_BACKEND_URL = "http://127.0.0.1:8000/chat"
BOT_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Travel Chatbot",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("✈️ Your Personal Travel Assistant")
st.caption("Ask me about travel plans, destinations, and activities! (Powered by Simple Chatbot)")

# --- Session State Initialization ---
# This is to keep the chat history persistent across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with your travel plans today?"}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # --- Get Bot Response from FastAPI Backend ---
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Prepare the request payload
            payload = {"user_message": prompt}
            
            # Send POST request to the backend
            with requests.post(FASTAPI_BACKEND_URL, json=payload, stream=False, timeout=60) as r:
                r.raise_for_status() # Raise an exception for bad status codes
                response_data = r.json()
                bot_response = response_data.get("response", "Sorry, I encountered an error.")

                # Simulate a streaming effect for better UX
                response_so_far = ""
                for chunk in bot_response.split():
                    response_so_far += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(response_so_far + "▌")
                message_placeholder.markdown(bot_response)
                full_response = bot_response


        except requests.exceptions.RequestException as e:
            full_response = f"**Error:** Could not connect to the chatbot backend. Please ensure it's running. Details: {e}"
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"**An unexpected error occurred:** {e}"
            message_placeholder.markdown(full_response)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar for Instructions ---
with st.sidebar:
    st.header("How to Use")
    st.info(
        "This is a simple travel chatbot powered by Azure OpenAI. "
        "It can answer general travel questions and provide assistance with your travel plans."
    )
    st.markdown(
        """
        **To get started:**
        1.  Open a terminal.
        2.  Navigate to the `rag-bot` directory.
        3.  Run the backend server:
            ```bash
            uvicorn simple_main:app --reload --port 8000
            ```
        4.  Open another terminal.
        5.  Run the Streamlit frontend:
            ```bash
            streamlit run travel_chat_bot.py
            ```
        """
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you?"}
        ]
        st.rerun()

