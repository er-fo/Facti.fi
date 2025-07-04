#!/bin/bash

echo "🎬 Starting Video Editing Module..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in the video-editing-module directory
if [[ ! -f "app.py" ]]; then
    echo "❌ Error: app.py not found in $SCRIPT_DIR"
    exit 1
fi

# Set environment variables
export VIDEO_MODULE_HOST=0.0.0.0
export VIDEO_MODULE_PORT=9000
export TRUTHSCORE_URL=http://localhost:8000

# Use the parent venv if it exists
if [[ -f "../venv/bin/activate" ]]; then
    echo "🔧 Using parent virtual environment..."
    source ../venv/bin/activate
else
    echo "❌ Error: Parent virtual environment not found. Please run from TruthScore root directory."
    exit 1
fi

# Install video module specific dependencies
echo "📦 Installing video module dependencies..."
pip install -r requirements.txt

# Load OpenAI API key from the file
if [[ -f "../eriks personliga api key" ]]; then
    export OPENAI_API_KEY=$(cat "../eriks personliga api key")
    echo "✅ OpenAI API key loaded successfully"
elif [[ -z "$OPENAI_API_KEY" ]]; then
    echo "❌ Error: OpenAI API key not found. Please ensure '../eriks personliga api key' file exists."
    echo "   Video module requires OpenAI API key for TTS generation."
    exit 1
fi

# Create output directories
mkdir -p "/tmp/video_module"
mkdir -p "/tmp/clips"

echo "🚀 Starting Video Editing Module on port 9000..."
echo "📡 TruthScore integration: http://localhost:8000"
echo "🔗 Video module health check: http://localhost:9000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python app.py 