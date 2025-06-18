#!/bin/bash

echo "🎤 Installing Speaker Diarization Dependencies for TruthScore"
echo "============================================================"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Please run: python -m venv venv"
        exit 1
    fi
fi

echo ""
echo "📦 Installing PyTorch and audio processing dependencies..."

# Install PyTorch (CPU version - adjust for GPU if needed)
pip install torch>=2.0.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "📦 Installing speaker diarization libraries..."

# Install pyannote and related dependencies
pip install pyannote.audio>=3.1.0
pip install transformers>=4.0.0
pip install huggingface_hub>=0.19.0
pip install librosa>=0.10.0
pip install soundfile>=0.12.0

echo ""
echo "🔧 Testing installation..."

# Test the installation
python -c "
import sys
try:
    import torch
    print('✅ PyTorch installed')
    import pyannote.audio
    print('✅ pyannote.audio installed') 
    import librosa
    print('✅ librosa installed')
    import soundfile
    print('✅ soundfile installed')
    print('✅ All dependencies installed successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎯 Next Steps:"
    echo "=============="
    echo "1. Get a Hugging Face token: https://huggingface.co/settings/tokens"
    echo "2. Accept the model license: https://huggingface.co/pyannote/speaker-diarization"
    echo "3. Set environment variable: export HF_TOKEN='your_token_here'"
    echo "4. Test setup: python test_speaker_diarization.py"
    echo "5. Start application: python app.py"
    echo ""
    echo "✅ Speaker diarization setup complete!"
else
    echo ""
    echo "❌ Installation failed. Please check error messages above."
    exit 1
fi 