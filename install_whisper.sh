#!/bin/bash

# TruthScore Local Whisper Installation Script
# This script installs the required dependencies for local Whisper functionality

echo "🎵 TruthScore Local Whisper Installation"
echo "========================================"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python3"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
    if [[ "$PYTHON_VERSION" == "3."* ]]; then
        PYTHON_CMD="python"
    fi
fi

echo "✅ Using Python command: $PYTHON_CMD"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Not in a virtual environment. It's recommended to use one."
    echo "   You can create one with: python -m venv venv && source venv/bin/activate"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
fi

# Install requirements
echo
echo "📦 Installing requirements..."
if [[ -f "requirements.txt" ]]; then
    $PYTHON_CMD -m pip install -r requirements.txt
    if [[ $? -eq 0 ]]; then
        echo "✅ Requirements installed successfully"
    else
        echo "❌ Failed to install requirements"
        exit 1
    fi
else
    echo "❌ requirements.txt not found. Make sure you're in the TruthScore directory."
    exit 1
fi

# Test Whisper installation
echo
echo "🧪 Testing Whisper installation..."
if $PYTHON_CMD test_whisper.py; then
    echo "✅ Whisper installation test passed!"
else
    echo "❌ Whisper installation test failed. Please check the error messages above."
    exit 1
fi

echo
echo "🎉 Installation completed successfully!"
echo
echo "💡 Tips:"
echo "  • The default Whisper model is 'tiny' (fastest processing)"
echo "  • Set WHISPER_MODEL_SIZE environment variable to change model:"
echo "    export WHISPER_MODEL_SIZE=tiny    # Default - Fastest"
echo "    export WHISPER_MODEL_SIZE=small   # Better accuracy"
echo "    export WHISPER_MODEL_SIZE=base    # Good balance"
echo "    export WHISPER_MODEL_SIZE=medium  # High accuracy"
echo "    export WHISPER_MODEL_SIZE=large   # Best accuracy (slowest)"
echo
echo "  • Start the application with: python app.py"
echo "  • Or use the start script: ./start.sh" 