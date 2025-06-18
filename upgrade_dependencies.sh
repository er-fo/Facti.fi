#!/bin/bash
# TruthScore Dependencies Upgrade Script
# Fixes Python 3.13 compatibility and performance issues

set -e

echo "🔧 TruthScore Dependencies Upgrade"
echo "=================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated. Please run: source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "🐍 Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "⚠️  Python 3.13 detected - upgrading dependencies for compatibility"
fi

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with proper version handling for Python 3.13
echo "🔥 Installing PyTorch..."
pytorch_installed=false

if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected - installing CUDA-enabled PyTorch"
    if pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121; then
        pytorch_installed=true
    fi
else
    echo "💻 CPU-only PyTorch installation"
    # For Python 3.13, try multiple approaches
    if [[ "$PYTHON_VERSION" == "3.13" ]]; then
        echo "🐍 Python 3.13 detected - trying compatible PyTorch installation..."
        
        # Try nightly builds first
        if pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu; then
            pytorch_installed=true
            echo "✅ Nightly PyTorch builds installed successfully"
        else
            echo "⚠️  Nightly builds failed, trying direct pip install..."
            # Fallback to direct pip install (latest available)
            if pip install torch torchaudio; then
                pytorch_installed=true
                echo "✅ PyTorch installed via direct pip"
            fi
        fi
    else
        if pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            pytorch_installed=true
        fi
    fi
fi

if [ "$pytorch_installed" = false ]; then
    echo "❌ PyTorch installation failed. Continuing with other dependencies..."
    echo "💡 You may need to install PyTorch manually:"
    echo "   For Python 3.13: pip install torch torchaudio"
    echo "   Or wait for official Python 3.13 support from PyTorch"
fi

# Uninstall old pyannote versions to prevent conflicts
echo "🧹 Cleaning old pyannote installations..."
pip uninstall -y pyannote.audio pyannote.pipeline pyannote.core pyannote.database || true

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Try to install speaker diarization dependencies if PyTorch is available
echo "🎤 Attempting to install speaker diarization dependencies..."
if [ "$pytorch_installed" = true ]; then
    echo "✅ PyTorch available - installing speaker diarization packages..."
    pip install pyannote.audio>=3.3.0 transformers>=4.30.0 huggingface_hub>=0.19.0 librosa>=0.10.0 soundfile>=0.12.0 || {
        echo "⚠️  Some speaker diarization dependencies failed to install"
        echo "💡 The application will work with transcription-only functionality"
    }
else
    echo "⚠️  PyTorch not available - skipping speaker diarization dependencies"
    echo "💡 Application will run in transcription-only mode"
fi

# Verify installations
echo "✅ Verifying installations..."
python -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
except ImportError as e:
    print(f'⚠️  PyTorch not available: {e}')
    print('💡 Speaker diarization will be disabled, but basic transcription should work')

try:
    import faster_whisper
    print(f'✅ faster-whisper available')
except ImportError as e:
    print(f'❌ faster-whisper: {e}')

try:
    from pyannote.audio import Pipeline
    print('✅ pyannote.audio Pipeline import successful')
except ImportError as e1:
    try:
        from pyannote.pipeline import Pipeline
        print('✅ pyannote.pipeline Pipeline import successful (legacy)')
    except ImportError as e2:
        print(f'❌ pyannote Pipeline: {e1}, {e2}')

try:
    import librosa
    print(f'✅ librosa {librosa.__version__} available')
except ImportError as e:
    print(f'❌ librosa: {e}')

try:
    import soundfile
    print(f'✅ soundfile {soundfile.__version__} available')
except ImportError as e:
    print(f'❌ soundfile: {e}')
"

echo ""
echo "🎉 Dependencies upgrade completed!"
echo ""
if [ "$pytorch_installed" = true ]; then
    echo "✅ Full installation with speaker diarization support"
    echo "💡 Next steps:"
    echo "   1. Set HF_TOKEN environment variable for speaker diarization"
    echo "   2. Run the application with: python app.py"
    echo "   3. Check logs/truthscore.log for any issues"
else
    echo "⚠️  Limited installation - transcription-only mode"
    echo "💡 Next steps:"
    echo "   1. Run the application with: python app.py"
    echo "   2. Consider upgrading to Python 3.11 or 3.12 for full speaker diarization support"
    echo "   3. Check logs/truthscore.log for any issues"
fi 