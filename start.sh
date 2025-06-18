#!/bin/bash

echo "🎯 Starting TruthScore Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  Warning: FFmpeg is not installed!"
    echo "Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    echo ""
fi

# Check if API key file exists
if [ ! -f "eriks personliga api key" ]; then
    echo "⚠️  Warning: OpenAI API key file not found!"
    echo "Please make sure 'eriks personliga api key' file exists with your OpenAI API key."
    echo ""
fi

echo "🚀 Starting Flask application..."
echo "Access the application at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py 