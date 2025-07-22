#!/usr/bin/env python3
"""
RAG Travel Chatbot Runner
This script helps you run the RAG chatbot system easily.
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'requests', 
        'python-dotenv', 'langchain', 'langchain-openai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    else:
        print("✅ All required packages are installed!")
    
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        return False
    
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_CHAT_DEPLOYMENT_NAME',
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME',
        'AZURE_OPENAI_CHAT_API_KEY'
    ]
    
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    missing_vars = []
    for var in required_vars:
        if var not in env_content or f"{var}=" not in env_content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ .env file is properly configured!")
    return True

def run_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend server...")
    try:
        # Use subprocess.Popen to start backend in background
        process = subprocess.Popen([
            sys.executable, 'backend.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Backend server started successfully!")
            print("🔗 Backend running at: http://127.0.0.1:8000")
            print("📖 API docs available at: http://127.0.0.1:8000/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ Backend failed to start!")
            if stderr:
                print(f"Error: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def run_frontend():
    """Start the Streamlit frontend"""
    print("🌟 Starting frontend...")
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'frontend.py',
            '--server.port', '8501',
            '--server.address', '127.0.0.1'
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def main():
    """Main function to run the chatbot system"""
    print("🗼 RAG Travel Chatbot - Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check environment file
    if not check_env_file():
        print("🔧 Please configure your .env file with Azure OpenAI credentials")
        return
    
    print("\n🎯 Choose how to run the chatbot:")
    print("1. Full System (Backend + Frontend)")
    print("2. Backend Only")
    print("3. Frontend Only")
    print("4. Console Chat Only")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting full system...")
            backend_process = run_backend()
            if backend_process:
                print("\n⏳ Waiting 5 seconds for backend to fully initialize...")
                time.sleep(5)
                print("🌟 Now starting frontend...")
                run_frontend()
                # Clean up backend when frontend stops
                backend_process.terminate()
            
        elif choice == "2":
            print("\n🚀 Starting backend only...")
            process = run_backend()
            if process:
                try:
                    print("🔗 Backend is running. Press Ctrl+C to stop.")
                    process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    print("\n👋 Backend stopped")
        
        elif choice == "3":
            print("\n🌟 Starting frontend only...")
            print("⚠️  Make sure backend is running at http://127.0.0.1:8000")
            run_frontend()
        
        elif choice == "4":
            print("\n💬 Starting console chat...")
            from rag_bot import run_console_chat
            import asyncio
            asyncio.run(run_console_chat())
        
        else:
            print("❌ Invalid choice. Please run the script again.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
