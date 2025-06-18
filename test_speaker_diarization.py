#!/usr/bin/env python3
"""
Test script for speaker diarization functionality.
This script demonstrates how to use the enhanced TruthScore analyzer with speaker diarization.
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_speaker_diarization():
    """Test the speaker diarization functionality"""
    print("🎤 Testing Speaker Diarization Integration")
    print("=" * 50)
    
    # Import the analyzer
    try:
        from app import TruthScoreAnalyzer
        print("✅ Successfully imported TruthScoreAnalyzer")
    except ImportError as e:
        print(f"❌ Failed to import TruthScoreAnalyzer: {e}")
        return False
    
    # Check dependencies
    print("\n📋 Checking Dependencies:")
    
    try:
        import pyannote.audio
        print("✅ pyannote.audio available")
    except ImportError:
        print("❌ pyannote.audio not available")
        print("   Install with: pip install pyannote.audio")
        return False
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("❌ PyTorch not available")
        print("   Install with: pip install torch")
        return False
    
    try:
        import librosa
        print("✅ librosa available")
    except ImportError:
        print("❌ librosa not available")
        print("   Install with: pip install librosa")
        return False
    
    # Check Hugging Face token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("✅ HF_TOKEN environment variable found")
    else:
        print("⚠️  HF_TOKEN not found - speaker diarization will be disabled")
        print("   Set with: export HF_TOKEN=your_huggingface_token")
    
    # Initialize analyzer
    print("\n🔧 Initializing TruthScoreAnalyzer...")
    try:
        analyzer = TruthScoreAnalyzer()
        print("✅ TruthScoreAnalyzer initialized successfully")
        
        # Check speaker diarization availability
        if analyzer.diarization_model:
            print("✅ Speaker diarization model loaded successfully")
        else:
            print("⚠️  Speaker diarization model not loaded")
            if not hf_token:
                print("   Reason: Missing HF_TOKEN")
            else:
                print("   Reason: Model loading failed")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False
    
    # Print example usage
    print("\n📖 Usage Examples:")
    print("=" * 50)
    
    print("\n1. Using the enhanced analyze endpoint (automatic):")
    print("   POST /analyze")
    print("   Body: {'url': 'https://youtube.com/watch?v=...'}")
    print("   - Extracts audio from URL")
    print("   - Performs transcription with speaker diarization")
    print("   - Saves enhanced transcript files")
    
    print("\n2. Using the dedicated speaker diarization endpoint:")
    print("   POST /transcribe_speakers")
    print("   Files: {'audio': audio_file}")
    print("   - Upload WAV/MP3/M4A file directly")
    print("   - Returns JSON with speaker-labeled transcript")
    
    print("\n3. Programmatic usage:")
    print("   from app import TruthScoreAnalyzer")
    print("   analyzer = TruthScoreAnalyzer()")
    print("   transcript = analyzer.transcribe_with_speakers('audio.wav')")
    
    # Print expected output format
    print("\n📄 Expected Output Format:")
    print("=" * 50)
    
    example_output = {
        "full_text": "Hello, how are you? I'm good, thank you!",
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 4.2,
                "text": "Hello, how are you?",
                "confidence": 0.95
            },
            {
                "speaker": "SPEAKER_01", 
                "start": 4.3,
                "end": 7.1,
                "text": "I'm good, thank you!",
                "confidence": 0.92
            }
        ],
        "language": "en",
        "speaker_count": 2,
        "has_speaker_diarization": True
    }
    
    print(json.dumps(example_output, indent=2))
    
    print("\n🔗 Next Steps:")
    print("=" * 50)
    print("1. Set HF_TOKEN environment variable with your Hugging Face token")
    print("2. Accept the pyannote/speaker-diarization license on Hugging Face")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Start the application: python app.py")
    print("5. Test with /health endpoint to verify speaker diarization status")
    
    return True

if __name__ == "__main__":
    success = test_speaker_diarization()
    if success:
        print("\n✅ Speaker diarization integration test completed!")
    else:
        print("\n❌ Speaker diarization integration test failed!")
        sys.exit(1) 