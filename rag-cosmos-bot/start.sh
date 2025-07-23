#!/bin/bash

# RAG Travel Chatbot Startup Script

echo "🚀 Starting RAG Travel Chatbot with Cosmos DB and AI Search..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️ Warning: .env file not found. Please create it with your Azure credentials."
    echo "📄 See .env.example for required variables."
    exit 1
fi

# Start backend in background
echo "🖥️ Starting backend server..."
python backend_new.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Start frontend
echo "🌐 Starting frontend (Streamlit)..."
streamlit run frontend_new.py --server.port 8501

# Cleanup - kill backend when frontend stops
echo "🛑 Stopping services..."
kill $BACKEND_PID
