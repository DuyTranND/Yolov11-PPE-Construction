#!/bin/bash

# PPE Detection System - Startup Helper Script
# This script helps you start both the API and UI in separate terminal sessions

echo "🦺 PPE Detection System - Startup Helper"
echo "========================================"
echo ""

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ Dependencies not installed!"
    echo ""
    echo "Please run one of the following:"
    echo "  uv pip install -e ."
    echo "  OR"
    echo "  pip install -e ."
    echo ""
    exit 1
fi

echo "✅ Dependencies found"
echo ""
echo "Please choose an option:"
echo ""
echo "  1. Start FastAPI Backend (API Server)"
echo "  2. Start Gradio Frontend (UI)"
echo "  3. Run API Tests"
echo "  4. View API Documentation (opens browser)"
echo "  5. Show all commands"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting FastAPI Backend..."
        echo "📡 API will be available at: http://127.0.0.1:8000"
        echo "📚 API Docs will be at: http://127.0.0.1:8000/docs"
        echo ""
        echo "Press CTRL+C to stop"
        echo ""
        uvicorn src.main:app --reload --port 8000
        ;;
    2)
        echo ""
        echo "🎨 Starting Gradio Frontend..."
        echo "🌐 UI will open in your browser"
        echo ""
        echo "⚠️  Make sure the API is running in another terminal!"
        echo "    (Run option 1 in a separate terminal first)"
        echo ""
        echo "Press CTRL+C to stop"
        echo ""
        python src/app_gradio.py
        ;;
    3)
        echo ""
        echo "🧪 Running API Tests..."
        echo ""
        echo "⚠️  Make sure the API is running!"
        echo ""
        python test_api.py
        ;;
    4)
        echo ""
        echo "📚 Opening API Documentation..."
        echo ""
        if command -v xdg-open &> /dev/null; then
            xdg-open http://127.0.0.1:8000/docs
        elif command -v open &> /dev/null; then
            open http://127.0.0.1:8000/docs
        else
            echo "Please open this URL in your browser:"
            echo "http://127.0.0.1:8000/docs"
        fi
        ;;
    5)
        echo ""
        echo "📋 All Available Commands:"
        echo ""
        echo "Install dependencies:"
        echo "  uv pip install -e ."
        echo "  OR"
        echo "  pip install -e ."
        echo ""
        echo "Start API Backend:"
        echo "  uvicorn src.main:app --reload --port 8000"
        echo ""
        echo "Start Gradio UI:"
        echo "  python src/app_gradio.py"
        echo ""
        echo "Run Tests:"
        echo "  python test_api.py"
        echo ""
        echo "View API Docs:"
        echo "  Visit http://127.0.0.1:8000/docs (after starting API)"
        echo ""
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac
