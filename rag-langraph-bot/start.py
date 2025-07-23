#!/usr/bin/env python3
"""
RAG Travel Assistant - Startup Script
====================================

This script provides easy commands to run different components of the RAG Travel Assistant.

Usage:
    python start.py --help                  # Show this help
    python start.py indexer                 # Run document indexer
    python start.py backend                 # Start backend API server
    python start.py frontend                # Start frontend web interface
    python start.py test                    # Run tests
    python start.py full                    # Start both backend and frontend

Requirements:
    - Python 3.8+
    - All dependencies installed (pip install -r requirements.txt)
    - Azure services configured (.env file)
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, background=False):
    """Run a command with proper error handling"""
    try:
        if background:
            print(f"🚀 Starting in background: {command}")
            return subprocess.Popen(command, shell=True)
        else:
            print(f"🔄 Running: {command}")
            result = subprocess.run(command, shell=True, check=True)
            return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        return None
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return None

def check_requirements():
    """Check if required files exist"""
    required_files = [
        ".env",
        "requirements.txt",
        "chatbot.py",
        "app.py",
        "frontend.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def run_indexer():
    """Run the document indexer"""
    print("📚 Running Document Indexer...")
    print("-" * 40)
    return run_command("python indexer.py")

def run_backend():
    """Start the backend API server"""
    print("🌐 Starting Backend API Server...")
    print("-" * 40)
    print("Backend will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop")
    return run_command("python app.py")

def run_frontend():
    """Start the frontend web interface"""
    print("🖥️ Starting Frontend Web Interface...")
    print("-" * 40)
    print("Frontend will be available at: http://localhost:8505")
    print("Press Ctrl+C to stop")
    return run_command("python -m streamlit run frontend.py --server.port 8505")

def run_tests():
    """Run the test suite"""
    print("🧪 Running Test Suite...")
    print("-" * 40)
    return run_command("python test.py")

def run_full():
    """Start both backend and frontend"""
    print("🚀 Starting Full Application...")
    print("-" * 40)
    
    # Start backend in background
    backend_process = run_command("python app.py", background=True)
    if not backend_process:
        print("❌ Failed to start backend")
        return
    
    print("⏳ Waiting for backend to initialize...")
    time.sleep(5)
    
    # Start frontend
    print("🖥️ Starting frontend...")
    print("Frontend will be available at: http://localhost:8505")
    print("Backend is available at: http://localhost:8000")
    print("\nPress Ctrl+C to stop both services")
    
    try:
        frontend_process = run_command("python -m streamlit run frontend.py --server.port 8505")
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    finally:
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Travel Assistant - Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "command",
        choices=["indexer", "backend", "frontend", "test", "full"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip requirements check"
    )
    
    args = parser.parse_args()
    
    print("🌍 RAG Travel Assistant")
    print("=" * 50)
    
    # Check requirements unless skipped
    if not args.skip_check and not check_requirements():
        print("\n💡 Make sure you have:")
        print("   1. Created a .env file with Azure credentials")
        print("   2. Installed dependencies: pip install -r requirements.txt")
        print("   3. All required files are present")
        sys.exit(1)
    
    # Run the requested command
    commands = {
        "indexer": run_indexer,
        "backend": run_backend,
        "frontend": run_frontend,
        "test": run_tests,
        "full": run_full
    }
    
    command_func = commands.get(args.command)
    if command_func:
        try:
            command_func()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
