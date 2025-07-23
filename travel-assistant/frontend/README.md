# Travel Assistant Frontend

A Streamlit-based web interface for the Travel Assistant chatbot with real-time streaming responses.

## Features

- Real-time streaming chat interface
- Clean and intuitive user experience
- Session management
- Error handling and connection status

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the backend is running on port 8000

3. Run the frontend:
```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

## Usage

Simply type your travel questions in the chat input and get real-time streaming responses from the AI assistant.

## Configuration

The frontend connects to the backend API at `http://localhost:8000` by default. This can be modified in the `API_BASE_URL` variable in `app.py`.
