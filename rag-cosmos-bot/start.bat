@echo off
REM RAG Travel Chatbot Startup Script for Windows

echo 🚀 Starting RAG Travel Chatbot with Cosmos DB and AI Search...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️ Warning: .env file not found. Please create it with your Azure credentials.
    echo 📄 See .env.example for required variables.
    pause
    exit /b 1
)

REM Start backend in background
echo 🖥️ Starting backend server...
start "Backend" python backend_new.py

REM Wait a moment for backend to start
timeout /t 5 /nobreak > nul

REM Start frontend
echo 🌐 Starting frontend (Streamlit)...
streamlit run frontend_new.py --server.port 8501

echo 🛑 Frontend stopped. Please manually close the backend window.
pause
