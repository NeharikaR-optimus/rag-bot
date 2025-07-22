# RAG Travel Chatbot

A complete RAG (Retrieval-Augmented Generation) chatbot system for Paris travel assistance, built with Azure OpenAI, LangChain, FastAPI, and Streamlit.

## 🌟 Features

- **RAG-Powered Responses**: Uses vector search to provide accurate, context-aware answers about Paris travel
- **Multi-Turn Conversations**: Remembers conversation history for natural dialogue
- **Web Interface**: Beautiful Streamlit frontend with real-time chat
- **RESTful API**: FastAPI backend for easy integration
- **Console Mode**: Command-line interface for testing
- **Azure OpenAI Integration**: Leverages GPT-4 and advanced embeddings

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd rag-cosmos-bot
pip install -r requirements.txt
```

### 2. Configure Environment
Make sure your `.env` file contains:
```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_chat_model
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_model
AZURE_OPENAI_CHAT_API_KEY=your_api_key
AZURE_OPENAI_CHAT_API_VERSION=2025-01-01-preview
```

### 3. Run the System
**Option A: Use the launcher (Recommended)**
```bash
python run_chatbot.py
```

**Option B: Manual startup**
```bash
# Terminal 1: Start backend
python backend.py

# Terminal 2: Start frontend
streamlit run frontend.py
```

**Option C: Console only**
```bash
python rag_bot.py
```

## 📁 File Structure

```
rag-cosmos-bot/
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── travel_knowledge.txt    # Knowledge base (Paris travel info)
├── rag_bot.py             # Core RAG chatbot implementation
├── backend.py             # FastAPI server
├── frontend.py            # Streamlit web interface
├── run_chatbot.py         # Easy launcher script
├── test_chatbot.py        # Testing utilities
└── README.md              # This file
```

## 🔧 Components

### Core RAG System (`rag_bot.py`)
- Document loading and chunking
- Vector store creation with FAISS
- Azure OpenAI integration
- Conversation memory management
- History-aware retrieval

### Backend API (`backend.py`)
- FastAPI server with CORS support
- RESTful endpoints for chat and health checks
- Error handling and validation
- Memory management endpoints

### Frontend UI (`frontend.py`)
- Streamlit chat interface
- Real-time backend status monitoring
- Chat history management
- Beautiful UI with custom styling

### Easy Launcher (`run_chatbot.py`)
- Automatic dependency checking
- Environment validation
- Multiple run modes
- Process management

## 🌐 API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /chat` - Send message to chatbot
- `POST /clear-memory` - Clear conversation memory

## 🧪 Testing

```bash
# Test the core chatbot
python test_chatbot.py

# Test API endpoints
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"user_message": "What are the top attractions in Paris?"}'
```

## 📚 Knowledge Base

The chatbot uses `travel_knowledge.txt` which contains comprehensive information about:
- Top Paris attractions
- Family-friendly activities
- Hotel recommendations
- Transportation options
- Dining suggestions
- Cultural tips and safety information

## 🔨 Customization

### Adding New Knowledge
1. Edit `travel_knowledge.txt` with new information
2. Restart the system to rebuild the vector store

### Modifying the Prompt
Edit the system prompts in `rag_bot.py` in the `_create_chains()` method.

### Changing Models
Update the deployment names in your `.env` file to use different Azure OpenAI models.

## 🚨 Troubleshooting

### Backend Won't Start
- Check `.env` file configuration
- Verify Azure OpenAI credentials
- Ensure all dependencies are installed

### Frontend Can't Connect
- Make sure backend is running on port 8000
- Check firewall settings
- Verify backend health at http://127.0.0.1:8000/health

### Slow Responses
- Check Azure OpenAI quota and limits
- Consider reducing chunk size or retrieval count
- Monitor network connectivity

## 🔐 Security Notes

- Keep your Azure OpenAI API keys secure
- Don't commit `.env` file to version control
- In production, configure CORS properly
- Use HTTPS for production deployments

## 📈 Performance Tips

- Vector store is built on startup (cached in memory)
- Conversation memory is limited to last 5 exchanges
- Streaming responses for better user experience
- Async implementation for better concurrency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes.

---

**Happy travels to Paris! 🗼**
