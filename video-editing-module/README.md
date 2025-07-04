# 🎬 Video Editing Module - Phase 3 Implementation

Advanced AI-powered video generation system that creates professional video clips from TruthScore credibility analysis data.

## 🚀 Phase 3 Features Implemented

### ✅ Content Classification System (`content_classifier.py`)
- **Intelligent Segment Analysis**: Automatically identifies and prioritizes key content segments
- **Semantic Density Scoring**: Uses TF-IDF analysis to find information-rich content
- **Claim Verification Integration**: Weights segments based on fact-checking results
- **Speaker Credibility Assessment**: Factors in speaker identification confidence
- **Research Backing Analysis**: Incorporates web research verification results
- **Narrative Flow Optimization**: Ensures logical content progression
- **Duration Targeting**: Optimizes content selection for target video length (25s social, 2min summary)

### ✅ TTS Generation Pipeline (`tts_generator.py`)
- **OpenAI TTS Integration**: High-quality AI voice generation with multiple voice options
- **Dynamic Script Generation**: Creates narration from analysis content
- **Voice Profile Selection**: Chooses appropriate voice based on content credibility
- **Audio Optimization**: Applies normalization, compression, and quality enhancement
- **Caching System**: Efficient audio caching for repeated content
- **Async Processing**: Non-blocking TTS generation for better performance

### ✅ Visual Overlay System (`visual_overlays.py`)
- **Credibility Score Meters**: Animated progress bars showing content reliability
- **Speaker Identification Badges**: Dynamic speaker labels with confidence indicators  
- **Fact-Check Indicators**: Color-coded verification symbols with pulse animations
- **Dynamic Subtitles**: Key claim highlighting with professional typography
- **Progress Bars**: Video timeline indicators
- **Glass Morphism Design**: Modern, professional visual aesthetic
- **Multi-layer Composition**: Advanced alpha blending and effects

### ✅ Audio Mixing Capabilities (`audio_mixer.py`)
- **Multi-track Audio Mixing**: Combines narration, background music, and sound effects
- **Automatic Gain Control**: Maintains consistent audio levels
- **Dynamic Range Compression**: Professional audio dynamics processing
- **EQ and Filtering**: Frequency optimization for clarity
- **Background Music Ducking**: Automatic volume reduction during speech
- **LUFS Normalization**: Broadcast-standard loudness levels (-16 LUFS)
- **Crossfades and Transitions**: Smooth audio transitions between segments
- **Real-time Audio Analysis**: Quality monitoring and optimization

### ✅ Video Assembly System (`video_assembler.py`)
- **Professional Video Rendering**: High-quality MP4 output with H.264 encoding
- **Animated Backgrounds**: Dynamic gradient backgrounds with particle effects
- **Audio-Video Synchronization**: Precise timing alignment
- **Multiple Resolution Support**: Configurable output resolutions (default 1920x1080)
- **GPU Acceleration**: Hardware acceleration when available
- **Quality Validation**: Automated output verification
- **Optimized Encoding**: Fast-start MP4 files optimized for streaming

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+ (required for compatibility with existing TruthScore system)
- OpenAI API key for TTS generation
- FFmpeg installed on system
- Sufficient storage space for video processing

### Install Dependencies
```bash
cd video-editing-module
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file:
```env
VIDEO_MODULE_HOST=0.0.0.0
VIDEO_MODULE_PORT=8000
OPENAI_API_KEY=your_openai_api_key_here
TRUTHSCORE_URL=http://localhost:5000
MAX_CONCURRENT_JOBS=2
TEMP_DIR=/tmp/video_module
OUTPUT_DIR=/tmp/clips
```

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended for HD video processing)
- **Storage**: 10GB free space for temporary files and output
- **CPU**: Multi-core processor (video encoding is CPU-intensive)
- **Network**: Stable internet connection for OpenAI TTS API calls

## 🎯 API Usage

### Start the Video Module
```bash
# Option 1: Use the automated startup script (recommended)
./start_video_module.sh

# Option 2: Manual startup with environment variables
export OPENAI_API_KEY=$(cat "../eriks personliga api key")
python app.py
```
The module runs independently on port 9000 by default.

### Generate Video Clip
```bash
curl -X POST http://localhost:9000/generate_clip \
  -H "Content-Type: application/json" \
  -d '{
    "truthscore_data": {
      "request_id": "example-123",
      "transcript": {
        "segments": [
          {
            "start": 0.0,
            "end": 5.0,
            "text": "This is an example claim",
            "speaker": "SPEAKER_01"
          }
        ]
      },
      "credibility_analysis": {
        "credibility_score": 75,
        "key_claims": ["Example claim text"],
        "red_flags": [],
        "bias_indicators": []
      },
      "speakers": {
        "SPEAKER_01": {
          "identified_name": "John Doe",
          "confidence": 0.85
        }
      }
    },
    "clip_config": {
      "type": "social",
      "target_duration": 25,
      "style": "motion_graphics"
    },
    "callback_url": "http://localhost:5000/video_complete"
  }'
```

### Monitor Progress
```bash
curl http://localhost:9000/progress/{clip_id}
```

### Download Generated Video
```bash
curl -O http://localhost:9000/download/{clip_id}
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Generation Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TruthScore Data → Content Classification → TTS Generation │
│                           ↓                                 │
│      Visual Overlays ← Video Assembly ← Audio Mixing       │
│                           ↓                                 │
│                    Final MP4 Output                        │
└─────────────────────────────────────────────────────────────┘
```

### Processing Steps

1. **Content Analysis** (`ContentClassifier`)
   - Analyzes TruthScore data structure
   - Calculates importance scores for segments
   - Classifies content types (intro, key_claim, evidence, conclusion)
   - Optimizes for target video duration

2. **TTS Generation** (`TTSGenerator`)
   - Generates script from classified content
   - Selects appropriate AI voice
   - Creates high-quality audio narration
   - Applies audio optimization

3. **Visual Composition** (`VisualOverlayRenderer`)
   - Creates overlay timeline
   - Generates credibility meters, speaker badges, fact-check indicators
   - Applies glass morphism design aesthetic

4. **Video Assembly** (`VideoAssembler`)
   - Creates animated background frames
   - Composes visual overlays
   - Handles frame-by-frame rendering

5. **Audio Mixing** (`AudioMixer`)
   - Combines narration with background elements
   - Applies professional audio processing
   - Ensures broadcast-quality output

6. **Final Encoding**
   - Synchronizes audio and video
   - Exports optimized MP4 files
   - Validates output quality

## 🎨 Visual Design System

### Color Scheme
- **High Credibility**: Green accent theme (#10b981)
- **Medium Credibility**: Cyan tech theme (#00ffff)  
- **Low Credibility**: Clean minimal theme (#3b82f6)

### Typography
- **Primary Font**: OpenCV default fonts optimized for video
- **Subtitle Size**: 24px for optimal readability
- **UI Elements**: Consistent spacing and hierarchy

### Animation System
- **Credibility Meters**: Smooth fill animations (0.8s duration)
- **Speaker Badges**: Fade in/out transitions (0.3s)
- **Fact-Check Indicators**: Pulse animations (1.0s cycle)
- **Background Effects**: Subtle particle animations

## 📊 Quality Standards

### Video Output
- **Resolution**: 1080x1920 (Portrait format for social media)
- **Frame Rate**: 30 FPS
- **Codec**: H.264 (x264)
- **Bitrate**: 3Mbps (optimized for social media)
- **Color Space**: YUV 4:2:0

### Audio Output  
- **Sample Rate**: 44.1kHz
- **Bit Depth**: 16-bit
- **Channels**: Stereo
- **Codec**: AAC
- **Bitrate**: 192kbps
- **Loudness**: -16 LUFS (broadcast standard)

### Performance Metrics
- **Generation Time**: <2 minutes for 25s clips
- **File Size**: ~15-25MB for 25s social clips
- **Success Rate**: >95% completion rate
- **Memory Usage**: <8GB peak during processing

## 🔧 Configuration Options

### Video Settings
```python
# In app.py Config class
OUTPUT_RESOLUTION = (1080, 1920)  # Portrait format for social media (TikTok, Instagram Stories)
DEFAULT_FRAME_RATE = 30           # Adjustable: 24, 30, or 60 FPS
DEFAULT_BITRATE = '3M'            # Optimized for social media file sizes
```

### TTS Voice Selection
Available OpenAI voices:
- `alloy`: Balanced, neutral tone
- `echo`: Authoritative, high energy (used for high credibility)
- `fable`: Warm, low energy
- `onyx`: Deep, medium energy
- `nova`: Professional, high energy (used for low credibility)
- `shimmer`: Friendly, medium energy (used for social clips)

### Processing Limits
- **Max Concurrent Jobs**: 2 (configurable via `MAX_CONCURRENT_JOBS`)
- **Temp File TTL**: 24 hours (auto-cleanup)
- **Max Clip Duration**: 180 seconds
- **Min Clip Duration**: 15 seconds

## 🚨 Error Handling & Recovery

### Common Issues
1. **OpenAI API Errors**: Module gracefully handles rate limits and API failures
2. **FFmpeg Missing**: Fallback to MoviePy for video encoding
3. **Insufficient Memory**: Automatic quality reduction for large videos
4. **Network Issues**: Retry logic for API calls with exponential backoff

### Monitoring & Logging
- Comprehensive logging to `logs/video_module.log`
- Progress tracking via `/progress/{clip_id}` endpoint  
- Health check available at `/health`
- Job monitoring via `/jobs` endpoint

## 🔄 Integration with TruthScore

The video module integrates seamlessly with the main TruthScore application:

1. **Analysis Export**: TruthScore exports analysis data via `/export_analysis/{request_id}`
2. **Video Generation**: Frontend calls video module API with exported data
3. **Progress Tracking**: Real-time progress updates via polling
4. **Completion Callback**: Video module notifies TruthScore when complete
5. **Download Integration**: Videos accessible through TruthScore UI

## 🎯 Future Enhancements (Phase 4+)

- **Background Music Library**: Curated music tracks based on content mood
- **Advanced Sound Effects**: Synchronized audio cues for visual events
- **Multi-language TTS**: Support for multiple languages and accents
- **Custom Branding**: Configurable logos, colors, and templates
- **Real-time Preview**: Live preview during generation
- **Batch Processing**: Multiple videos from single analysis
- **Cloud Storage**: Direct upload to S3/GCS for scalability

## 📈 Performance Optimization

### Processing Speed
- **Parallel Processing**: TTS generation and visual overlay creation run concurrently
- **Caching**: Audio and visual elements cached for reuse
- **GPU Acceleration**: Automatic detection and use of hardware acceleration
- **Memory Management**: Efficient cleanup and memory recycling

### Scalability Considerations
- **Horizontal Scaling**: Multiple video module instances can run in parallel
- **Queue Management**: Redis/database-backed job queues for high volume
- **CDN Integration**: Optimized video delivery via content distribution networks
- **Container Deployment**: Docker support for easy deployment and scaling

---

## 🤝 Contributing

The video module follows the same development principles as the main TruthScore project:
- **Simple is Sufficient**: Elegant solutions over complex implementations
- **Long-term Solutions**: Avoid temporary fixes and shortcuts
- **Production Ready**: Full implementation, no mocks or stubs
- **Documentation**: Keep all documentation current with implementation

For questions or issues, refer to the main TruthScore project documentation or create an issue in the project repository. 