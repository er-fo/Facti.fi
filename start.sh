#!/bin/bash

# TruthScore Complete System Startup Script
# Launches both main application (port 8000) and video editing module (port 9000)

echo "🎯 Starting Complete TruthScore System..."
echo "🔄 Killing any existing processes and launching fresh instances"

# Function to kill processes on a specific port
kill_port_processes() {
    local port=$1
    local service_name=$2
    
    echo "🔍 Checking for existing processes on port $port ($service_name)..."
    if lsof -ti:$port > /dev/null 2>&1; then
        echo "💀 Killing existing processes on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
        echo "✅ Port $port cleared"
    else
        echo "✅ Port $port is free"
    fi
}

# Kill any existing processes
kill_port_processes 8000 "TruthScore Main App"
kill_port_processes 9000 "Video Editing Module"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Set environment variables for safe startup
export DISABLE_SPEAKER_DIARIZATION=""  # Don't explicitly disable, just defer loading
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE  # Fix OpenMP runtime conflict
export OMP_NUM_THREADS=1  # Limit OpenMP threads to prevent conflicts

# Video module specific environment variables
export VIDEO_MODULE_HOST=0.0.0.0
export VIDEO_MODULE_PORT=9000
export TRUTHSCORE_URL=http://localhost:8000

# Install dependencies for main app
echo "📦 Installing main application dependencies..."
pip install -r requirements.txt

# Install video module dependencies
echo "📦 Installing video module dependencies..."
cd video-editing-module
pip install -r requirements.txt
cd ..

# Create necessary directories
mkdir -p logs
mkdir -p /tmp/video_module
mkdir -p /tmp/clips

# Pre-flight checks
echo "🔍 Running pre-flight checks..."

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

# Check if OpenAI API key is set for video module
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable not set!"
    echo "Video module may fail during TTS generation."
    echo ""
fi

echo "🚀 Starting TruthScore Complete System..."
echo ""
echo "📡 Services:"
echo "  - TruthScore Main App: http://localhost:8000"
echo "  - Video Editing Module: http://localhost:9000"
echo ""
echo "🎬 Features Available:"
echo "  - Credibility Analysis"
echo "  - Speaker Diarization"
echo "  - Web Research Integration"
echo "  - Analysis History Database"
echo "  - Professional Video Generation"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Create a function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down TruthScore system..."
    kill_port_processes 8000 "TruthScore Main App"
    kill_port_processes 9000 "Video Editing Module"
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM

# Start the video editing module in background
echo "🎬 Starting Video Editing Module (port 9000)..."
cd video-editing-module
python app.py > ../logs/video_module.log 2>&1 &
VIDEO_MODULE_PID=$!
cd ..

# Wait a moment for video module to start
sleep 3

# Check if video module started successfully
if kill -0 $VIDEO_MODULE_PID 2>/dev/null; then
    echo "✅ Video Editing Module started successfully (PID: $VIDEO_MODULE_PID)"
else
    echo "❌ Video Editing Module failed to start"
    echo "📋 Check logs/video_module.log for details"
fi

# Start the main TruthScore application in foreground
echo "🎯 Starting TruthScore Main Application (port 8000)..."
echo "Speaker diarization will be loaded on first use to prevent startup crashes."
echo ""

# Start the main application (this will run in foreground)
python app.py

# If we reach here, the main app has stopped - cleanup the video module
echo ""
echo "🛑 Main application stopped, cleaning up..."
cleanup 