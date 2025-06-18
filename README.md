# 🎯 TruthScore - AI-Powered Credibility Analysis

⚡ A sleek, modern web application that uses AI to analyze the credibility of audio and video content.

## ✨ Features

### 🎯 Core Functionality
- **URL Analysis**: Support for YouTube, Vimeo, SoundCloud, and many other platforms
- **AI-Powered Transcription**: Uses local OpenAI Whisper model for accurate speech-to-text with timestamps
- **🎤 Speaker Diarization**: NEW! Automatically identifies who spoke when using pyannote.audio
- **Credibility Scoring**: Advanced AI analysis providing 0-100 credibility scores
- **Research Integration**: Automated fact-checking and claim verification
- **Real-time Processing**: Live updates during analysis with visual progress tracking
- **Progress Visualization**: Beautiful progress bars and step indicators showing transcription, analysis, and research progress
- **Multiple Output Formats**: Enhanced transcripts with speaker identification in JSON and text formats

### 🚀 Sleek Tech UI
- **Dark Theme**: Modern dark interface with tech-inspired color palette
- **Glassmorphism**: Translucent glass cards with blur effects
- **Smooth Animations**: Fluid transitions and hover effects  
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Monospace Typography**: Tech-focused fonts for a professional look
- **Neon Accents**: Cyan and purple gradients with glow effects

## 🎨 Design Principles

The UI follows these modern design principles:

- **Simplicity**: Clean, minimalist interface without clutter
- **Tech Aesthetic**: Dark theme with neon accents and monospace fonts
- **Visual Hierarchy**: Clear organization of information
- **Accessibility**: High contrast and readable typography
- **Performance**: Optimized CSS with hardware acceleration

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher (Python 3.13 supported)
- FFmpeg (for audio processing)
- OpenAI API key
- yt-dlp **2025.06.09 or newer** (installed automatically from `requirements.txt`)
- **Optional**: Hugging Face token for speaker diarization (recommended)

⚠️ **Important**: If you're having issues with Python 3.13 or pyannote.audio imports, see the [Troubleshooting](#troubleshooting) section below.

### 1. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add to PATH

### 2. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd truthscore

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies (including local Whisper model)
# For Python 3.13 users: use the automated upgrade script
./upgrade_dependencies.sh

# OR manually install:
pip install -r requirements.txt
```

### 3. Install Local Whisper Model

**Quick Setup (Recommended):**
```bash
./install_whisper.sh
```

**Manual Setup:**
```bash
# Test Whisper installation
python test_whisper.py

# Optional: Configure Whisper model size
export WHISPER_MODEL_SIZE=tiny  # Options: tiny, base, small, medium, large
```

**Model Recommendations:**
- `tiny` - **Default** - Fastest processing, good accuracy for most use cases
- `base` - Good balance of speed and accuracy 
- `small` - Better accuracy, slightly slower
- `medium` - High accuracy, noticeably slower
- `large` - Best accuracy, slowest processing

### 4. Configure API Keys

**OpenAI API Key:**
Make sure your OpenAI API key is in the file `eriks personliga api key` in the project root, or update the code to read from an environment variable.

**Speaker Diarization (Optional but Recommended):**
```bash
# Get a free Hugging Face token at https://huggingface.co/
export HF_TOKEN="your_huggingface_token_here"

# Accept the license at: https://huggingface.co/pyannote/speaker-diarization
```

**Test Setup:**
```bash
python test_speaker_diarization.py
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:8000`

**Note:** The first run will download the selected Whisper model (~39MB for tiny model, ~74MB for base model).

## 🎯 Usage

### Main Analysis (with Speaker Diarization)
1. **Enter URL**: Paste a link to video/audio content
2. **Analyze**: Click the analyze button to start processing
3. **Review Results**: Get detailed credibility analysis including:
   - Overall credibility score (0-100)
   - **Speaker-labeled transcript** with "who spoke when"
   - Key claims identification
   - Red flags and bias indicators
   - Factual accuracy assessment
   - Evidence quality evaluation
   - Full transcript with research findings

### Direct Audio Upload
```bash
# Upload audio file directly for speaker diarization
curl -X POST http://localhost:8000/transcribe_speakers \
  -F "audio=@your_audio_file.wav"
```

### Health Check
```bash
# Check if speaker diarization is available
curl http://localhost:8000/health
```

## 🧠 AI Models

- **Transcription**: Local OpenAI Whisper model (state-of-the-art speech recognition with timestamps)
- **🎤 Speaker Diarization**: PyAnnote.audio speaker-diarization-3.1 (identifies who spoke when)
- **Analysis**: OpenAI o3-mini (configurable via environment variable)
- **Research**: Comprehensive AI-powered fact-checking and claim verification system

## 📊 Analysis Output

Each analysis provides comprehensive insights:

### Credibility Score
- **70-100**: High credibility (green indicator)
- **40-69**: Medium credibility (yellow indicator)  
- **0-39**: Low credibility (red indicator)

### Detailed Breakdown
- **Key Claims**: Main assertions made in the content
- **Red Flags**: Potential misleading or concerning statements
- **Bias Indicators**: Signs of emotional manipulation or bias
- **Evidence Quality**: Assessment of supporting evidence
- **Research Results**: Comprehensive fact-checking with detailed verification outcomes

### Research & Verification Features
Each claim receives detailed analysis including:
- **Verification Status**: VERIFIED ✅ | PARTIALLY_VERIFIED ⚠️ | DISPUTED ❌ | UNVERIFIABLE ❓ | FALSE ❌
- **Truthfulness Score**: 0-100 numerical assessment
- **Evidence Quality**: STRONG | MODERATE | WEAK | INSUFFICIENT
- **Supporting Evidence**: Documented facts that support the claim
- **Contradicting Evidence**: Information that disputes the claim
- **Verification Notes**: Detailed explanation of research findings
- **Recommendation**: ACCEPT | ACCEPT_WITH_CAUTION | QUESTION | REJECT
- **Confidence Level**: HIGH | MEDIUM | LOW based on evidence quality

## 🔧 Configuration

Set environment variables to customize behavior:

```bash
export TRUTHSCORE_MODEL="gpt-4"     # Change AI model
export FLASK_ENV="production"       # Production mode
export HF_TOKEN="your_hf_token"     # Enable speaker diarization
export WHISPER_MODEL_SIZE="tiny"    # Whisper model size (tiny/small/base/medium/large)
```

## 🔧 Troubleshooting

### Python 3.13 Compatibility Issues

If you encounter import errors like `cannot import name 'Pipeline' from 'pyannote.audio'`:

1. **Use the automated upgrade script** (recommended):
   ```bash
   source venv/bin/activate
   ./upgrade_dependencies.sh
   ```

2. **Manual fix**:
   ```bash
   pip install --upgrade pyannote.audio>=3.3.0 torch>=2.1.0
   ```

### Transcription Timeout Issues

If transcriptions are timing out:

1. **Use a smaller model**:
   ```bash
   export WHISPER_MODEL_SIZE=tiny  # Fastest option
   ```

2. **Check system resources** - close unnecessary applications

3. **For very long audio files** - the timeout has been extended to 15 minutes

### Language Detection Issues

If Whisper incorrectly detects the language:
- The system now forces English detection for better accuracy
- Previous Welsh ('cy') detection issues have been resolved

### Speaker Diarization Issues

If speaker diarization fails:

1. **Set HuggingFace token**:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

2. **Accept model license** at: https://huggingface.co/pyannote/speaker-diarization-3.1

3. **Check internet connection** for model downloads

### Performance Optimization

For better performance:

1. **GPU acceleration** (automatic if CUDA available)
2. **Adjust model size** based on your needs:
   - `tiny`: **Default** - Fastest, good accuracy for most use cases
   - `small`: Better accuracy, slightly slower
   - `base`: Good balance, moderate speed
   - `medium`/`large`: High accuracy, very slow

## 📋 Recent Fixes Applied

See [FIXES_APPLIED.md](FIXES_APPLIED.md) for detailed information about recent improvements including:
- Python 3.13 compatibility fixes
- Transcription timeout resolution  
- Language detection improvements
- Performance optimizations

## 🎤 Speaker Diarization Features

### Enhanced Output Format

When speaker diarization is enabled, transcripts include speaker identification:

```json
{
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
  "has_speaker_diarization": true
}
```

### File Outputs

The system automatically saves enhanced transcripts in multiple formats:
- `transcript_REQUESTID_with_speakers.json` - Structured data
- `transcript_REQUESTID_with_speakers.txt` - Human-readable format

### Use Cases

Perfect for:
- **Interviews**: Distinguish interviewer from interviewee
- **Meetings**: Track individual participants
- **Podcasts**: Separate hosts and guests
- **Phone Calls**: Identify different speakers
- **Educational Content**: Distinguish instructor from student questions

## 📝 Logging

Comprehensive logging system tracks:
- Application events
- API usage and performance
- Error tracking and debugging
- User interactions

Logs are stored in the `logs/` directory with automatic rotation.

## 🚀 Production Deployment

For production use:

1. Set up proper environment variables
2. Configure reverse proxy (nginx recommended)
3. Use production WSGI server (gunicorn)
4. Set up SSL certificates
5. Configure monitoring and alerting

## 🔧 Troubleshooting

### Common Issues

#### "Failed to load" Error in Web Interface

**Symptoms**: Analysis request hangs or fails with "failed to load" message

**Possible Causes & Solutions**:

1. **Long Video Transcription Timeout**
   - **Issue**: Videos longer than ~10 minutes may timeout during transcription
   - **Solution**: The application now includes a 5-minute timeout protection
   - **Recommendation**: For very long videos, consider using a smaller Whisper model (`tiny` or `base`) for faster processing

2. **OpenAI API Model Compatibility**
   - **Issue**: Some models (like `o3-mini`) don't support all parameters
   - **Solution**: The `temperature` parameter has been removed for better compatibility
   - **Fix Applied**: Updated to use model without temperature parameter

3. **Memory Issues with Large Audio Files**
   - **Issue**: Large audio files can cause memory exhaustion
   - **Solution**: The system now has better error handling and cleanup
   - **Recommendation**: Use `base` or `small` Whisper model for better memory efficiency

#### API Errors

- **Model not found**: Check your OpenAI API key and model access
- **Quota exceeded**: Verify your OpenAI billing and usage limits
- **Unsupported parameter**: Model configuration has been updated for better compatibility

#### Audio Processing Issues

- **Audio extraction fails**: Ensure `yt-dlp` and `ffmpeg` are properly installed
- **Unsupported URL**: Only video/audio platforms supported by `yt-dlp` work
- **File format issues**: The system automatically converts to MP3 for compatibility

#### Display Issues

- **"[object Object]" in results**: Fixed with automatic data format conversion
- **Key claims not displaying properly**: System now handles both string and object formats from AI responses
- **Missing timestamps**: Timestamps are automatically extracted and displayed when available

### Performance Optimization

For better performance with large files:
- Use smaller Whisper models (`base` instead of `large`)
- Ensure sufficient RAM is available
- Consider shorter audio clips for testing

## 🔒 Privacy & Security

- No user data is stored permanently
- Temporary audio files are automatically cleaned up after processing
- All analyses are public and transparent
- API key should be kept secure and not shared

## 🛠️ Technical Architecture

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Audio Processing**: yt-dlp + FFmpeg
- **AI Services**: Local OpenAI Whisper model + configurable LLM (default **o3-mini**)
- **Logging**: Comprehensive multi-file logging system
- **Deployment**: Ready for local or cloud deployment

## 📋 Logging System

TruthScore includes a comprehensive logging system with multiple specialized log files:

### Log Files Created

- **`logs/truthscore.log`** - Main application log with all general activities
- **`logs/analysis.log`** - Detailed analysis workflow tracking
- **`logs/api_usage.log`** - OpenAI API calls and usage monitoring
- **`logs/errors.log`** - Error tracking and debugging information
- **`logs/performance.log`** - Performance metrics and timing data

### Log Features

- **Automatic Rotation**: Log files rotate when they reach 10MB (errors: 10 files, others: 5 files)
- **Request Tracking**: Each analysis request gets a unique ID for easy tracking
- **Performance Monitoring**: Detailed timing information for each operation
- **API Usage Tracking**: Monitor OpenAI API calls, response times, and costs
- **Error Details**: Comprehensive error logging with stack traces
- **Console Output**: Important logs also appear in the console during development

### Monitoring Usage

```bash
# View real-time main application logs
tail -f logs/truthscore.log

# Monitor API usage and costs
tail -f logs/api_usage.log

# Watch for errors
tail -f logs/errors.log

# Check analysis performance
tail -f logs/analysis.log

# View all logs together
tail -f logs/*.log
```

### Log Levels

- **INFO**: Normal operations, request tracking, successful completions
- **WARNING**: Non-critical issues, validation failures, cleanup problems
- **ERROR**: Critical failures, API errors, analysis failures
- **DEBUG**: Detailed debugging information (disabled by default)

## 🤝 Contributing

This is a basic implementation focusing on core functionality. Potential improvements:

- Enhanced web research integration
- Support for more content types
- User authentication for private analyses
- Database storage for analysis history
- Advanced bias detection algorithms
- Multi-language support

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for open-source Whisper model and GPT models
- yt-dlp for media extraction
- Flask for the web framework
- All contributors and testers

---

**TruthScore** - Promoting transparency and critical thinking through AI-powered analysis.

## 🚨 Troubleshooting

### Common Issues

1. **FFmpeg not found**: Make sure FFmpeg is installed and in your system PATH
2. **OpenAI API errors**: Check your API key and account billing status
3. **URL not supported**: Verify the URL is accessible and supported by yt-dlp
4. **Long processing times**: Large files take more time; be patient
5. **Memory issues**: Very long content may require more system resources

### Error Messages

- "Failed to extract audio": URL may be invalid or unsupported
- "Failed to transcribe audio": Audio file may be corrupted or too long
- "Analysis failed": OpenAI API issue or transcript too long for analysis

### Using Logs for Troubleshooting

1. **Check the main log**: `tail -f logs/truthscore.log` for general application issues
2. **Monitor API usage**: `logs/api_usage.log` to check OpenAI API calls and quotas
3. **Review errors**: `logs/errors.log` for detailed error information with stack traces
4. **Track performance**: `logs/performance.log` for timing and performance issues
5. **Follow analysis workflow**: `logs/analysis.log` for step-by-step analysis tracking

Each request gets a unique ID (e.g., `[20241201_143022_123456]`) making it easy to track a specific analysis through all log files.

For support, check the relevant log files for detailed error messages and timing information. 