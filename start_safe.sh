#!/bin/bash

# TruthScore Safe Startup Script - Prevents segfaults during initialization
# This script ensures the application starts reliably by deferring speaker diarization loading

echo "🛡️  Starting TruthScore Application (Safe Mode)..."

# Check for existing processes on port 8000
echo "Checking for existing processes on port 8000..."
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "Killing existing processes on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi
echo "Port 8000 cleared."

# Activate virtual environment
echo "Activating virtual environment..."
source ./venv/bin/activate

# Set environment variables for safe startup
export DISABLE_SPEAKER_DIARIZATION=""  # Don't explicitly disable, just defer loading
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE  # Fix OpenMP runtime conflict
export OMP_NUM_THREADS=1  # Limit OpenMP threads to prevent conflicts

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🚀 Starting Flask application in safe mode..."
echo "Speaker diarization will be loaded on first use to prevent startup crashes."
echo "Access the application at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
python app.py 