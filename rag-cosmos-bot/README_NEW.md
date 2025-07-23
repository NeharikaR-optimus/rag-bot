# 🌍 RAG Travel Chatbot with Azure Cosmos DB & AI Search

A comprehensive **Retrieval-Augmented Generation (RAG)** travel chatbot powered by **Azure OpenAI**, **Azure Cosmos DB**, **Azure AI Search**, and **LangGraph** with real-time streaming responses.

## ✨ Features

- 🔄 **Real-time Streaming Responses** using LangGraph workflow
- 🗄️ **Azure Cosmos DB Integration** for document and embedding storage
- 🔍 **Azure AI Search** for vector-based document retrieval
- 🧠 **Conversation Memory** with context awareness
- 📚 **Multi-document Knowledge Base** covering global travel destinations
- 🎯 **FastAPI Backend** with comprehensive endpoints
- 🌐 **Streamlit Frontend** with modern UI and real-time chat
- ⚡ **Async Processing** for optimal performance

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │────│   FastAPI       │────│   RAG Engine    │
│   Frontend      │    │   Backend       │    │   (LangGraph)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Azure Cosmos DB │    │ Azure AI Search │
                       │ (Documents +    │    │ (Vector Search) │
                       │  Embeddings)    │    │                 │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                               ┌─────────────────┐
                                               │ Azure OpenAI    │
                                               │ (GPT + Embed)   │
                                               └─────────────────┘
```

## 📚 Knowledge Base

The chatbot includes comprehensive travel guides for:

### European Destinations
- **Paris Travel Guide**: Attractions, dining, accommodation, transportation
- **London Travel Guide**: Landmarks, culture, practical tips
- **Rome Travel Guide**: Historic sites, Vatican, Italian cuisine
- **European Travel Tips**: General advice for European travel

### Asian Destinations  
- **Asia Travel Guide**: Japan, Thailand, China, India, South Korea
- **Cultural insights** and practical travel information

### Specialized Travel
- **Adventure Travel Guide**: Mountain climbing, water sports, safaris
- **Budget Travel Guide**: Cost-effective travel strategies
- **Family Travel Guide**: Family-friendly destinations and tips

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure OpenAI API access
- Azure Cosmos DB account
- Azure AI Search service

### 1. Environment Setup

Create a `.env` file with your Azure credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Chat Model Configuration
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4.1
AZURE_OPENAI_CHAT_API_KEY=your-chat-api-key

# Embedding Model Configuration
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_API_KEY=your-embedding-api-key

# Azure AI Search Configuration
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=travel-index

# Azure Cosmos DB Configuration
COSMOS_DB_ENDPOINT=https://your-cosmos.documents.azure.com:443/
COSMOS_DB_KEY=your-cosmos-key
COSMOS_DB_DATABASE_NAME=travel_knowledge
COSMOS_DB_CONTAINER_NAME=documents

# Backend API Configuration
BACKEND_API_URL=http://localhost:8000
```

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-cosmos-bot

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application

#### Option A: Automatic Startup (Recommended)

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

#### Option B: Manual Startup

**Terminal 1 - Backend:**
```bash
python backend_new.py
```

**Terminal 2 - Frontend:**
```bash
streamlit run frontend_new.py --server.port 8501
```

### 4. Access the Application

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 API Endpoints

### Chat Endpoints
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - Streaming chat with real-time responses

### Management Endpoints
- `GET /health` - Backend health check
- `POST /memory/clear` - Clear conversation memory
- `GET /documents/count` - Get document count
- `POST /documents/reload` - Reload documents from files

### Example API Usage

```python
import requests

# Non-streaming chat
response = requests.post("http://localhost:8000/chat", 
                        json={"message": "Tell me about Paris attractions"})
print(response.json()["response"])

# Clear memory
requests.post("http://localhost:8000/memory/clear")
```

## 🗄️ Data Storage Architecture

### Azure Cosmos DB Structure
```json
{
  "id": "document-hash-id",
  "document_id": "unique-identifier", 
  "content": "document text content",
  "metadata": {
    "source_file": "paris_travel_guide.txt",
    "created_at": "2025-01-22"
  },
  "embedding": [0.1, 0.2, 0.3, ...],
  "created_at": "timestamp"
}
```

### Azure AI Search Integration
- **Vector Search**: Uses embeddings for semantic similarity
- **Hybrid Search**: Combines keyword and vector search
- **Index Management**: Automatic index creation and updates

## 💡 Key Components

### 1. RAG Engine (`rag_bot.py`)
- **LangGraph Workflow**: Orchestrates retrieval and generation
- **Azure Integration**: Cosmos DB and AI Search connectivity
- **Streaming Support**: Real-time response generation
- **Memory Management**: Conversation context handling

### 2. FastAPI Backend (`backend_new.py`)
- **RESTful API**: Clean endpoint structure
- **Async Processing**: Non-blocking operations
- **Error Handling**: Comprehensive error management
- **CORS Support**: Frontend integration

### 3. Streamlit Frontend (`frontend_new.py`)
- **Modern UI**: Custom CSS styling
- **Real-time Chat**: Streaming response display
- **Backend Management**: Health checks and controls
- **Responsive Design**: Mobile-friendly interface

## 🔄 Streaming Implementation

The application uses **Server-Sent Events (SSE)** for real-time streaming:

1. **Frontend** sends message to `/chat/stream`
2. **Backend** processes through LangGraph workflow
3. **LangGraph** yields response chunks in real-time
4. **Frontend** displays chunks as they arrive

```python
# Streaming response example
async def generate_response():
    async for chunk in chatbot.get_response_streaming(message):
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
```

## 🧠 LangGraph Workflow

```python
# Workflow nodes
retrieve_context -> generate_response -> END

# State management
GraphState {
    messages: List,
    user_input: str,
    context: str,
    response: str,
    chat_history: List
}
```

## 📊 Performance Optimization

- **Document Caching**: Cosmos DB stores processed documents
- **Embedding Reuse**: Avoid recomputing embeddings
- **Async Operations**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Chunking Strategy**: Optimal document splitting

## 🛠️ Development

### Adding New Documents

1. Place `.txt` files in the `documents/` folder
2. Restart the application or use the "Reload Documents" button
3. Documents are automatically processed and stored

### Customizing the Workflow

Edit the `_create_graph_nodes()` method in `rag_bot.py` to modify:
- Retrieval strategy
- Response generation
- Context processing
- Memory management

### Frontend Customization

Modify `frontend_new.py` to adjust:
- UI styling (CSS)
- Chat interface
- Sidebar functionality
- Response display

## 🔍 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Check if backend is running on port 8000
   - Verify `.env` configuration
   - Check firewall settings

2. **Azure Service Errors**
   - Validate API keys and endpoints
   - Check service quotas and limits
   - Verify resource permissions

3. **Document Loading Issues**
   - Ensure documents are in UTF-8 encoding
   - Check file permissions
   - Verify Cosmos DB connection

### Logs and Debugging

- Backend logs appear in terminal running `backend_new.py`
- Frontend errors shown in Streamlit interface
- Use `/health` endpoint for backend status

## 📈 Scaling Considerations

- **Cosmos DB**: Configure appropriate throughput (RU/s)
- **AI Search**: Monitor search units and storage
- **Azure OpenAI**: Watch rate limits and quotas
- **Caching**: Implement Redis for session management

## 🔒 Security Best Practices

- Store API keys in environment variables
- Use Azure Key Vault for production secrets
- Implement authentication for production use
- Configure CORS properly for production
- Monitor API usage and costs

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Azure service documentation
- Submit issues to the repository

---

**Built with ❤️ using Azure AI Services, LangChain, FastAPI, and Streamlit**
