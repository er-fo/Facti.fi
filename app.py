from flask import Flask, request, render_template, jsonify
import os
import tempfile
import yt_dlp
import openai
import requests
from datetime import datetime, timedelta
import json
import re
import logging
import sys
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler
from faster_whisper import WhisperModel
import threading
import queue

# Web research imports
try:
    from serpapi import GoogleSearch
    WEB_SEARCH_AVAILABLE = True
    logging.getLogger('app').info("SerpAPI web search available")
except ImportError as e:
    WEB_SEARCH_AVAILABLE = False
    logging.getLogger('app').warning(f"SerpAPI web search not available: {e}")
    logging.getLogger('app').info("Install 'google-search-results' package to enable real web research")

# Speaker diarization imports - Updated for compatibility
DIARIZATION_AVAILABLE = False
try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.pipeline import Pipeline as LegacyPipeline
    import librosa
    import soundfile as sf
    DIARIZATION_AVAILABLE = True
    logging.getLogger('app').info("Speaker diarization dependencies loaded successfully")
except ImportError as e:
    try:
        # Fallback for older versions
        import torch
        from pyannote.pipeline import Pipeline
        from pyannote.pipeline import Pipeline as LegacyPipeline
        import librosa
        import soundfile as sf
        DIARIZATION_AVAILABLE = True
        logging.getLogger('app').info("Speaker diarization dependencies loaded successfully (legacy)")
    except ImportError as e2:
        logging.getLogger('app').warning(f"Speaker diarization not available: {e} (fallback also failed: {e2})")
        logging.getLogger('app').info("Application will continue with transcription-only functionality")
        DIARIZATION_AVAILABLE = False

app = Flask(__name__)

# Import and setup comprehensive logging
from logging_config import setup_logging
setup_logging()

# Get application logger
app_logger = logging.getLogger('app')

# Load API key
try:
    with open('eriks personliga api key', 'r') as f:
        api_key = f.read().strip()
    app_logger.info("OpenAI API key loaded successfully")
except FileNotFoundError:
    app_logger.error("OpenAI API key file not found: 'eriks personliga api key'")
    raise
except Exception as e:
    app_logger.error(f"Error loading API key: {e}")
    raise

# Configure OpenAI client
try:
    client = openai.OpenAI(api_key=api_key)
    app_logger.info("OpenAI client initialized successfully")
except Exception as e:
    app_logger.error(f"Error initializing OpenAI client: {e}")
    raise

# Global progress tracking
progress_store = {}
results_store = {}  # Add results storage

# Global request tracking persistence file path
REQUEST_TRACKING_FILE = "request_tracking.json"

# SerpAPI configuration and usage tracking
SERPAPI_USAGE_FILE = "serpapi_usage.json"

def load_serpapi_usage():
    """Load SerpAPI usage tracking data"""
    try:
        if os.path.exists(SERPAPI_USAGE_FILE):
            with open(SERPAPI_USAGE_FILE, 'r') as f:
                return json.load(f)
        else:
            return {"daily_usage": {}, "monthly_usage": {}}
    except Exception as e:
        app_logger.warning(f"Failed to load SerpAPI usage data: {e}")
        return {"daily_usage": {}, "monthly_usage": {}}

def save_serpapi_usage(usage_data):
    """Save SerpAPI usage tracking data"""
    try:
        with open(SERPAPI_USAGE_FILE, 'w') as f:
            json.dump(usage_data, f, indent=2)
    except Exception as e:
        app_logger.warning(f"Failed to save SerpAPI usage data: {e}")

def check_serpapi_quota():
    """Check if we're within SerpAPI quota limits"""
    usage_data = load_serpapi_usage()
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.now().strftime("%Y-%m")
    
    daily_limit = int(os.getenv("SERPAPI_DAILY_LIMIT", "10"))  # Conservative daily limit
    monthly_limit = int(os.getenv("SERPAPI_MONTHLY_LIMIT", "90"))  # Leave some buffer from 100/month
    
    daily_usage = usage_data.get("daily_usage", {}).get(current_date, 0)
    monthly_usage = usage_data.get("monthly_usage", {}).get(current_month, 0)
    
    if daily_usage >= daily_limit:
        return False, f"Daily quota exceeded ({daily_usage}/{daily_limit})"
    if monthly_usage >= monthly_limit:
        return False, f"Monthly quota exceeded ({monthly_usage}/{monthly_limit})"
    
    return True, f"Quota OK (Daily: {daily_usage}/{daily_limit}, Monthly: {monthly_usage}/{monthly_limit})"

def increment_serpapi_usage():
    """Increment SerpAPI usage counters"""
    usage_data = load_serpapi_usage()
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.now().strftime("%Y-%m")
    
    # Increment daily usage
    if "daily_usage" not in usage_data:
        usage_data["daily_usage"] = {}
    usage_data["daily_usage"][current_date] = usage_data["daily_usage"].get(current_date, 0) + 1
    
    # Increment monthly usage
    if "monthly_usage" not in usage_data:
        usage_data["monthly_usage"] = {}
    usage_data["monthly_usage"][current_month] = usage_data["monthly_usage"].get(current_month, 0) + 1
    
    save_serpapi_usage(usage_data)
    
    # Clean up old daily data (keep only last 7 days)
    cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    usage_data["daily_usage"] = {k: v for k, v in usage_data["daily_usage"].items() if k >= cutoff_date}
    
    # Clean up old monthly data (keep only last 3 months)
    cutoff_month = (datetime.now() - timedelta(days=90)).strftime("%Y-%m")
    usage_data["monthly_usage"] = {k: v for k, v in usage_data["monthly_usage"].items() if k >= cutoff_month}
    
    save_serpapi_usage(usage_data)

def load_request_tracking():
    """Load request tracking data from file to persist across restarts"""
    global progress_store, results_store
    try:
        if os.path.exists(REQUEST_TRACKING_FILE):
            with open(REQUEST_TRACKING_FILE, 'r') as f:
                data = json.load(f)
                progress_store = data.get('progress_store', {})
                results_store = data.get('results_store', {})
                app_logger.info(f"Loaded request tracking data: {len(progress_store)} progress entries, {len(results_store)} result entries")
        else:
            app_logger.info("No existing request tracking file found - starting fresh")
    except Exception as e:
        app_logger.warning(f"Failed to load request tracking data: {e}")
        progress_store = {}
        results_store = {}

def save_request_tracking():
    """Save request tracking data to file"""
    try:
        data = {
            'progress_store': progress_store,
            'results_store': results_store
        }
        with open(REQUEST_TRACKING_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        app_logger.warning(f"Failed to save request tracking data: {e}")

def cleanup_old_requests():
    """Clean up requests older than 1 hour to prevent memory buildup"""
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=1)
    
    # Clean up old progress entries
    to_remove_progress = []
    for request_id, progress_data in progress_store.items():
        try:
            timestamp_str = progress_data.get('timestamp', '')
            timestamp = datetime.fromisoformat(timestamp_str)
            if timestamp < cutoff_time:
                to_remove_progress.append(request_id)
        except:
            # If timestamp parsing fails, remove the entry
            to_remove_progress.append(request_id)
    
    for request_id in to_remove_progress:
        del progress_store[request_id]
    
    # Clean up old result entries
    to_remove_results = []
    for request_id, result_data in results_store.items():
        try:
            timestamp_str = result_data.get('timestamp', '')
            timestamp = datetime.fromisoformat(timestamp_str)
            if timestamp < cutoff_time:
                to_remove_results.append(request_id)
        except:
            # If timestamp parsing fails, remove the entry
            to_remove_results.append(request_id)
    
    for request_id in to_remove_results:
        del results_store[request_id]
    
    if to_remove_progress or to_remove_results:
        app_logger.info(f"Cleaned up {len(to_remove_progress)} old progress entries and {len(to_remove_results)} old result entries")
        save_request_tracking()

# Load existing request tracking data on startup
load_request_tracking()
cleanup_old_requests()

class TruthScoreAnalyzer:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger('analysis')
        self.api_logger = logging.getLogger('api_usage')
        self.error_logger = logging.getLogger('errors')
        # Use a configurable model name so deployments without GPT-4 access can still operate
        # Default to the lightweight "o3-mini" model, but allow override via environment variable
        self.model_name = os.getenv("TRUTHSCORE_MODEL", "o3-mini")
        self.current_request_id = None
        
        # Speaker identification knowledge base
        self.speaker_patterns = {
            "Donald Trump": {
                "keywords": ["tremendous", "incredible", "beautiful", "fake news", "witch hunt", "make america great", "believe me"],
                "speech_patterns": ["superlatives", "repetition", "crowd references"],
                "policy_positions": ["immigration restrictions", "america first", "trade deals", "border wall"],
                "distinctive_phrases": ["nobody knows more than me", "it's true", "many people are saying"]
            },
            "Joe Biden": {
                "keywords": ["folks", "malarkey", "senate", "barack", "democracy", "build back better"],
                "speech_patterns": ["folksy expressions", "senate references", "obama mentions"],
                "policy_positions": ["infrastructure", "climate change", "healthcare", "unity"],
                "distinctive_phrases": ["here's the deal", "not a joke", "come on, man"]
            },
            "Nancy Pelosi": {
                "keywords": ["speaker", "house", "impeachment", "democracy", "constitution"],
                "speech_patterns": ["parliamentary language", "constitutional references"],
                "policy_positions": ["democratic agenda", "trump investigations"],
                "distinctive_phrases": ["for the people", "when we win"]
            },
            "Mitch McConnell": {
                "keywords": ["senate majority", "conservative", "judges", "kentucky"],
                "speech_patterns": ["senate procedure", "conservative talking points"],
                "policy_positions": ["judicial appointments", "conservative agenda"],
                "distinctive_phrases": ["my democratic colleagues", "senate tradition"]
            },
            "Alexandria Ocasio-Cortez": {
                "keywords": ["green new deal", "climate", "progressive", "bronx", "medicare for all"],
                "speech_patterns": ["progressive rhetoric", "social media references"],
                "policy_positions": ["climate action", "social justice", "wealth inequality"],
                "distinctive_phrases": ["let's be clear", "working families"]
            },
            "Bernie Sanders": {
                "keywords": ["billionaire", "political revolution", "medicare for all", "wall street"],
                "speech_patterns": ["passionate delivery", "repetitive emphasis"],
                "policy_positions": ["wealth inequality", "healthcare", "education"],
                "distinctive_phrases": ["let me be clear", "enough is enough", "political revolution"]
            }
        }
        
        # Initialize faster-whisper model with optimized settings
        whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")  # Use tiny for fastest results while maintaining reasonable accuracy
        try:
            self.logger.info(f"Loading Faster-Whisper model: {whisper_model_size}")
            # Use optimized compute type and device settings
            if DIARIZATION_AVAILABLE and 'torch' in globals():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "float32"
            else:
                device = "cpu"
                compute_type = "float32"
                self.logger.info("PyTorch not available - using CPU-only settings")
            
            self.whisper_model = WhisperModel(
                whisper_model_size, 
                device=device, 
                compute_type=compute_type,
                cpu_threads=4  # Limit CPU threads to prevent overload
            )
            self.logger.info(f"Faster-Whisper model {whisper_model_size} loaded successfully on {device} with {compute_type}")
        except Exception as e:
            self.error_logger.error(f"Failed to load Faster-Whisper model {whisper_model_size}: {e}")
            # Fallback to tiny model with safe settings
            try:
                self.logger.info("Falling back to tiny Faster-Whisper model with safe settings")
                self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32", cpu_threads=2)
                self.logger.info("Faster-Whisper tiny model loaded successfully as fallback")
            except Exception as fallback_e:
                self.error_logger.error(f"Failed to load fallback Faster-Whisper model: {fallback_e}")
                raise
        
        # Initialize speaker diarization model if available
        self.diarization_model = None
        if DIARIZATION_AVAILABLE:
            try:
                # Check for HuggingFace token
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    self.logger.info("Loading speaker diarization model...")
                    # Try to use the newer Pipeline import first
                    try:
                        self.diarization_model = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token
                        )
                        self.logger.info("Speaker diarization model loaded successfully with pyannote.audio Pipeline")
                    except Exception as pipeline_e:
                        # Fallback to legacy Pipeline if available
                        try:
                            self.diarization_model = LegacyPipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                use_auth_token=hf_token
                            )
                            self.logger.info("Speaker diarization model loaded successfully with legacy Pipeline")
                        except Exception as legacy_e:
                            self.error_logger.error(f"Failed to load diarization model with both Pipeline types: {pipeline_e}, {legacy_e}")
                            raise legacy_e
                else:
                    self.logger.warning("HF_TOKEN not found. Speaker diarization will be disabled.")
                    self.logger.warning("Set HF_TOKEN environment variable with your Hugging Face token to enable speaker diarization.")
            except Exception as e:
                self.error_logger.error(f"Failed to load speaker diarization model: {e}")
                self.logger.warning("Speaker diarization will be disabled for this session.")
        
        self.logger.info(f"TruthScoreAnalyzer initialized with temp directory: {self.temp_dir}")
    
    def set_progress(self, step, percentage, message):
        """Update progress for the current request with cumulative progress across all stages"""
        if self.current_request_id:
            # Define stage weights for overall progress calculation
            stage_weights = {
                'extraction': (0, 25),      # 0-25% of total
                'transcription': (25, 50),  # 25-50% of total  
                'analysis': (50, 75),       # 50-75% of total
                'research': (75, 100),      # 75-100% of total
                'cleanup': (95, 98),        # 95-98% of total
                'complete': (100, 100)      # 100% of total
            }
            
            # Calculate overall progress
            if step in stage_weights:
                start_pct, end_pct = stage_weights[step]
                # Scale the stage percentage to the overall range
                overall_percentage = start_pct + (percentage / 100.0) * (end_pct - start_pct)
                overall_percentage = min(100, max(0, int(overall_percentage)))
            else:
                # Fallback for unknown steps
                overall_percentage = percentage
            
            progress_store[self.current_request_id] = {
                'step': step,
                'percentage': overall_percentage,
                'stage_percentage': percentage,  # Keep original stage percentage for logging
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            # Save progress data persistently
            save_request_tracking()
            self.logger.info(f"[{self.current_request_id}] Progress: {step} - {percentage}% (overall: {overall_percentage}%) - {message}")
    
    def set_request_id(self, request_id):
        """Set the current request ID for progress tracking"""
        self.current_request_id = request_id
    
    def is_valid_url(self, url):
        """Validate if the provided string is a valid URL"""
        try:
            result = urlparse(url)
            is_valid = all([result.scheme, result.netloc])
            self.logger.info(f"URL validation for '{url}': {'valid' if is_valid else 'invalid'}")
            return is_valid
        except Exception as e:
            self.logger.warning(f"Error validating URL '{url}': {e}")
            return False
    
    def extract_audio(self, url):
        """Extract audio from video/speech content using yt-dlp"""
        self.logger.info(f"Starting audio extraction from URL: {url}")
        start_time = datetime.now()
        
        try:
            self.set_progress("extraction", 0, "Initializing audio extraction...")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,  # Reduce yt-dlp output noise
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.set_progress("extraction", 20, "Extracting video info...")
                self.logger.info("Extracting video info and downloading audio...")
                
                self.set_progress("extraction", 40, "Downloading audio...")
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 'Unknown')
                
                self.set_progress("extraction", 80, "Processing audio file...")
                self.logger.info(f"Content info - Title: '{title}', Duration: {duration}s")
                
                # Find the downloaded audio file
                for file in os.listdir(self.temp_dir):
                    if file.endswith('.mp3'):
                        file_path = os.path.join(self.temp_dir, file)
                        file_size = os.path.getsize(file_path)
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        
                        self.set_progress("extraction", 100, f"Audio extraction complete - {file}")
                        self.logger.info(f"Audio extraction completed in {elapsed_time:.2f}s - File: {file}, Size: {file_size} bytes")
                        return file_path, title
                        
            self.logger.error("No MP3 file found after extraction")
            self.set_progress("extraction", 0, "No MP3 file found after extraction")
            return None, None
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Audio extraction failed after {elapsed_time:.2f}s for URL '{url}': {e}")
            self.set_progress("extraction", 0, f"Audio extraction failed: {str(e)}")
            return None, None
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using local OpenAI Whisper model with timestamps"""
        self.logger.info(f"Starting audio transcription for file: {audio_path}")
        start_time = datetime.now()
        
        try:
            file_size = os.path.getsize(audio_path)
            self.logger.info(f"Transcribing audio file - Size: {file_size} bytes")
            
            self.set_progress("transcription", 0, "Initializing transcription...")
            self.api_logger.info("Using local Faster-Whisper model for transcription with timestamps")
            
            # Use threading for timeout protection instead of signals (Flask compatibility)
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            progress_queue = queue.Queue()
            
            def transcribe_worker():
                try:
                    self.set_progress("transcription", 10, "Loading audio file...")
                    
                    # Transcribe with faster-whisper (returns segments and info)
                    # Use optimized settings for better performance and reliability
                    try:
                        segments, info = self.whisper_model.transcribe(
                            audio_path, 
                            word_timestamps=True,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            beam_size=3,  # Further reduce beam size for faster processing
                            best_of=3,    # Further reduce best_of for faster processing
                            temperature=0.0,  # Use deterministic output
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                            no_speech_threshold=0.6,
                            language="en"  # Force English to prevent Welsh misdetection
                        )
                    except Exception as transcribe_error:
                        self.logger.warning(f"Primary transcription failed: {transcribe_error}, trying fallback settings...")
                        # Fallback with minimal settings if primary fails
                        segments, info = self.whisper_model.transcribe(
                            audio_path,
                            word_timestamps=False,
                            vad_filter=False,
                            beam_size=1,
                            best_of=1,
                            temperature=0.0,
                            language="en"
                        )
                    
                    self.set_progress("transcription", 50, "Processing audio segments...")
                    
                    # Convert segments generator to list and track progress
                    segments_list = []
                    segment_count = 0
                    
                    for segment in segments:
                        segments_list.append(segment)
                        segment_count += 1
                        
                        # Update progress every 10 segments
                        if segment_count % 10 == 0:
                            progress_percentage = min(50 + (segment_count * 2), 90)
                            self.set_progress("transcription", progress_percentage, f"Processed {segment_count} segments...")
                    
                    self.set_progress("transcription", 95, "Finalizing transcription...")
                    result_queue.put((segments_list, info))
                    
                except Exception as e:
                    exception_queue.put(e)
            
            # Start transcription in a separate thread
            transcription_thread = threading.Thread(target=transcribe_worker)
            transcription_thread.daemon = True
            transcription_thread.start()
            
            # Wait for completion with dynamic timeout based on file size
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            # Base timeout of 10 minutes + 2 minutes per MB (more reasonable for long files)
            dynamic_timeout = max(600, int(600 + (file_size_mb * 120)))
            self.logger.info(f"Using dynamic timeout of {dynamic_timeout}s ({dynamic_timeout/60:.1f} minutes) for {file_size_mb:.1f}MB file")
            
            transcription_thread.join(timeout=dynamic_timeout)
            
            if transcription_thread.is_alive():
                # Timeout occurred
                elapsed_time = (datetime.now() - start_time).total_seconds()
                self.error_logger.error(f"Audio transcription timed out after {elapsed_time:.2f}s for file '{audio_path}': Timeout after {dynamic_timeout/60:.1f} minutes")
                self.set_progress("transcription", 0, f"Transcription timed out after {dynamic_timeout/60:.1f} minutes")
                return None
            
            # Check for exceptions
            if not exception_queue.empty():
                raise exception_queue.get()
            
            # Get results
            if result_queue.empty():
                self.error_logger.error(f"Audio transcription failed: No results returned")
                self.set_progress("transcription", 0, "Transcription failed")
                return None
                
            segments, info = result_queue.get()
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            self.set_progress("transcription", 98, "Building transcript...")
            
            # Extract segments with timestamps and build full text
            segments_with_timestamps = []
            full_text_parts = []
            
            for segment in segments:
                segment_info = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                segments_with_timestamps.append(segment_info)
                full_text_parts.append(segment.text.strip())
            
            transcript_text = " ".join(full_text_parts)
            transcript_length = len(transcript_text)
            
            # Create detailed transcript object
            detailed_transcript = {
                "full_text": transcript_text,
                "segments": segments_with_timestamps,
                "language": info.language,
                "duration": elapsed_time
            }
            
            self.set_progress("transcription", 100, f"Transcription complete - {transcript_length} characters")
            
            self.logger.info(f"Transcription completed in {elapsed_time:.2f}s - Length: {transcript_length} characters, Segments: {len(segments_with_timestamps)}")
            self.api_logger.info(f"Local Whisper transcription successful - Transcript length: {transcript_length} chars, Duration: {elapsed_time:.2f}s, Segments: {len(segments_with_timestamps)}")
            
            return detailed_transcript
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Audio transcription failed after {elapsed_time:.2f}s for file '{audio_path}': {e}")
            self.api_logger.error(f"Local Whisper transcription failed: {e}")
            self.set_progress("transcription", 0, f"Transcription failed: {str(e)}")
            return None
    
    def perform_speaker_diarization(self, audio_path):
        """Perform speaker diarization using pyannote.audio"""
        if not self.diarization_model:
            self.logger.warning("Speaker diarization model not available")
            return None
            
        self.logger.info(f"Starting speaker diarization for file: {audio_path}")
        start_time = datetime.now()
        
        try:
            self.set_progress("diarization", 0, "Initializing speaker diarization...")
            
            # Load audio file
            self.set_progress("diarization", 10, "Loading audio file...")
            
            # Ensure audio is in correct format for diarization
            # Load audio and ensure it's mono and at 16kHz sample rate
            audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Save processed audio temporarily for diarization
            temp_audio_path = os.path.join(self.temp_dir, "temp_diarization.wav")
            sf.write(temp_audio_path, audio_data, sample_rate)
            
            self.set_progress("diarization", 30, "Processing audio for speaker identification...")
            
            # Perform diarization
            diarization = self.diarization_model(temp_audio_path)
            
            self.set_progress("diarization", 80, "Processing speaker segments...")
            
            # Convert diarization results to structured format
            speaker_segments = []
            speaker_count = 0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segment = {
                    "speaker": speaker,
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "duration": float(turn.end - turn.start)
                }
                speaker_segments.append(speaker_segment)
                
                # Count unique speakers
                if speaker not in [seg["speaker"] for seg in speaker_segments[:-1]]:
                    speaker_count += 1
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            self.set_progress("diarization", 100, f"Speaker diarization complete - {speaker_count} speakers detected")
            self.logger.info(f"Speaker diarization completed in {elapsed_time:.2f}s - Detected {speaker_count} speakers, {len(speaker_segments)} segments")
            
            return {
                "segments": speaker_segments,
                "speaker_count": speaker_count,
                "duration": elapsed_time
            }
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Speaker diarization failed after {elapsed_time:.2f}s for file '{audio_path}': {e}")
            self.set_progress("diarization", 0, f"Diarization failed: {str(e)}")
            return None
    
    def align_transcript_with_speakers(self, transcript_data, diarization_data):
        """Align Whisper transcript segments with speaker diarization results"""
        if not transcript_data or not diarization_data:
            self.logger.warning("Missing transcript or diarization data for alignment")
            return transcript_data
            
        self.logger.info("Starting transcript-speaker alignment")
        start_time = datetime.now()
        
        try:
            self.set_progress("alignment", 0, "Aligning transcript with speakers...")
            
            transcript_segments = transcript_data.get("segments", [])
            speaker_segments = diarization_data.get("segments", [])
            
            aligned_segments = []
            total_segments = len(transcript_segments)
            
            for i, transcript_segment in enumerate(transcript_segments):
                transcript_start = transcript_segment["start"]
                transcript_end = transcript_segment["end"]
                transcript_text = transcript_segment["text"]
                
                # Find the speaker who was talking during this transcript segment
                best_speaker = None
                best_overlap = 0
                
                for speaker_segment in speaker_segments:
                    speaker_start = speaker_segment["start"]
                    speaker_end = speaker_segment["end"]
                    
                    # Calculate overlap between transcript and speaker segments
                    overlap_start = max(transcript_start, speaker_start)
                    overlap_end = min(transcript_end, speaker_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    # Calculate overlap percentage relative to transcript segment
                    transcript_duration = transcript_end - transcript_start
                    if transcript_duration > 0:
                        overlap_percentage = overlap_duration / transcript_duration
                        
                        # Use the speaker with the highest overlap
                        if overlap_percentage > best_overlap:
                            best_overlap = overlap_percentage
                            best_speaker = speaker_segment["speaker"]
                
                # Create aligned segment
                aligned_segment = {
                    "speaker": best_speaker if best_speaker else "Unknown",
                    "start": transcript_start,
                    "end": transcript_end,
                    "text": transcript_text,
                    "confidence": best_overlap  # How confident we are about speaker assignment
                }
                
                aligned_segments.append(aligned_segment)
                
                # Update progress
                progress = int((i + 1) / total_segments * 100)
                self.set_progress("alignment", progress, f"Aligned {i + 1}/{total_segments} segments")
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Create enhanced transcript with speaker information
            enhanced_transcript = {
                "full_text": transcript_data.get("full_text", ""),
                "segments": aligned_segments,
                "language": transcript_data.get("language", "unknown"),
                "speaker_count": diarization_data.get("speaker_count", 0),
                "alignment_duration": elapsed_time,
                "has_speaker_diarization": True
            }
            
            self.logger.info(f"Transcript-speaker alignment completed in {elapsed_time:.2f}s - {len(aligned_segments)} segments aligned")
            return enhanced_transcript
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Transcript-speaker alignment failed after {elapsed_time:.2f}s: {e}")
            self.set_progress("alignment", 0, f"Alignment failed: {str(e)}")
            return transcript_data  # Return original transcript if alignment fails
    
    def transcribe_with_speakers(self, audio_path):
        """Perform complete transcription with speaker diarization"""
        self.logger.info(f"Starting transcription with speaker diarization for: {audio_path}")
        start_time = datetime.now()
        
        try:
            # First, perform standard transcription with robust error handling
            self.logger.info("Step 1: Performing audio transcription...")
            transcript_data = self.transcribe_audio(audio_path)
            if not transcript_data:
                self.logger.error("Audio transcription failed completely")
                return None
            
            self.logger.info(f"Transcription successful: {len(transcript_data.get('full_text', ''))} characters, {len(transcript_data.get('segments', []))} segments")
            
                         # If diarization is available, perform speaker identification
            if self.diarization_model:
                self.logger.info("Step 2: Performing speaker diarization...")
                try:
                    # Use threading timeout for diarization to prevent hanging
                    import threading
                    import queue
                    
                    diarization_queue = queue.Queue()
                    diarization_exception_queue = queue.Queue()
                    
                    def diarization_worker():
                        try:
                            result = self.perform_speaker_diarization(audio_path)
                            diarization_queue.put(result)
                        except Exception as e:
                            diarization_exception_queue.put(e)
                    
                    # Start diarization in separate thread with timeout
                    diarization_thread = threading.Thread(target=diarization_worker)
                    diarization_thread.daemon = True
                    diarization_thread.start()
                    
                    # Wait with 5-minute timeout
                    diarization_thread.join(timeout=300)
                    
                    if diarization_thread.is_alive():
                        self.logger.warning("Speaker diarization timed out after 5 minutes, falling back to standard transcript")
                        transcript_data["has_speaker_diarization"] = False
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        self.logger.info(f"Transcription timeout fallback finished in {elapsed_time:.2f}s")
                        return transcript_data
                    
                    # Check for exceptions
                    if not diarization_exception_queue.empty():
                        raise diarization_exception_queue.get()
                    
                    # Get results
                    diarization_data = None
                    if not diarization_queue.empty():
                        diarization_data = diarization_queue.get()
                    
                    if diarization_data:
                        self.logger.info("Step 3: Aligning transcript with speakers...")
                        # Align transcript with speakers
                        enhanced_transcript = self.align_transcript_with_speakers(transcript_data, diarization_data)
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        self.logger.info(f"Complete transcription with speakers finished in {elapsed_time:.2f}s")
                        return enhanced_transcript
                    else:
                        self.logger.warning("Speaker diarization failed, returning transcript without speaker information")
                        transcript_data["has_speaker_diarization"] = False
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        self.logger.info(f"Transcription without speakers finished in {elapsed_time:.2f}s")
                        return transcript_data
                        
                except Exception as e:
                    self.logger.warning(f"Speaker diarization failed with error: {e}, falling back to standard transcript")
                    transcript_data["has_speaker_diarization"] = False
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    self.logger.info(f"Transcription fallback finished in {elapsed_time:.2f}s")
                    return transcript_data
            else:
                self.logger.info("Speaker diarization not available, returning standard transcript")
                transcript_data["has_speaker_diarization"] = False
                elapsed_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Standard transcription finished in {elapsed_time:.2f}s")
                return transcript_data
                
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Transcription with speakers failed after {elapsed_time:.2f}s: {e}")
            return None
    
    def save_transcript_with_speakers(self, transcript_data, output_path):
        """Save enhanced transcript to file in multiple formats"""
        if not transcript_data:
            return None
            
        try:
            # Create base filename without extension
            base_path = os.path.splitext(output_path)[0]
            
            # Save as JSON
            json_path = f"{base_path}_with_speakers.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
            # Save as readable text
            txt_path = f"{base_path}_with_speakers.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                if transcript_data.get("has_speaker_diarization", False):
                    f.write("TRANSCRIPT WITH SPEAKER DIARIZATION\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Group segments by speaker for readability
                    current_speaker = None
                    for segment in transcript_data.get("segments", []):
                        speaker = segment.get("speaker", "Unknown")
                        start_time = segment.get("start", 0)
                        end_time = segment.get("end", 0)
                        text = segment.get("text", "")
                        
                        if speaker != current_speaker:
                            if current_speaker is not None:
                                f.write("\n")
                            f.write(f"\n[{speaker}] ")
                            current_speaker = speaker
                        
                        f.write(f"({start_time:.1f}-{end_time:.1f}s) {text} ")
                    
                    f.write(f"\n\n--- Summary ---\n")
                    f.write(f"Total speakers detected: {transcript_data.get('speaker_count', 'Unknown')}\n")
                    f.write(f"Language: {transcript_data.get('language', 'Unknown')}\n")
                else:
                    f.write("TRANSCRIPT (NO SPEAKER DIARIZATION)\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(transcript_data.get("full_text", ""))
            
            self.logger.info(f"Enhanced transcript saved to: {json_path} and {txt_path}")
            return {"json": json_path, "txt": txt_path}
            
        except Exception as e:
            self.error_logger.error(f"Failed to save enhanced transcript: {e}")
            return None
    
    def perform_web_research(self, claims):
        """Perform comprehensive web research on claims using SerpAPI Google Search with AI analysis"""
        self.logger.info(f"Starting real web research for {len(claims)} claims")
        
        if not WEB_SEARCH_AVAILABLE:
            self.logger.warning("SerpAPI search not available, falling back to AI-only analysis")
            return self._perform_ai_only_research(claims)
        
        # Check API key availability
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            self.logger.warning("SERPAPI_KEY not found in environment variables, falling back to AI-only analysis")
            return self._perform_ai_only_research(claims)
        
        # Check quota before proceeding
        quota_ok, quota_message = check_serpapi_quota()
        if not quota_ok:
            self.logger.warning(f"SerpAPI quota exceeded: {quota_message}, falling back to AI-only analysis")
            return self._perform_ai_only_research(claims)
        
        self.logger.info(f"SerpAPI quota status: {quota_message}")
        
        research_results = []
        claims_to_research = claims[:2]  # Limit to first 2 claims to conserve quota
        
        self.logger.info(f"Processing {len(claims_to_research)} claims for web research")
        
        for i, claim in enumerate(claims_to_research, 1):
            # Handle different claim formats (string or dict)
            if isinstance(claim, dict):
                claim_text = str(claim.get('text', claim.get('claim', str(claim))))
            else:
                claim_text = str(claim)
            
            # Safely truncate the claim text for logging
            claim_preview = claim_text[:100] + "..." if len(claim_text) > 100 else claim_text
            
            self.logger.info(f"Web researching claim {i}: {claim_preview}")
            
            try:
                # Step 1: Generate search queries for the claim
                self.set_progress("research", (i-1)*50, f"Generating search queries for claim {i}...")
                
                search_queries = self._generate_search_queries(claim_text)
                # Limit to 2 queries per claim to conserve quota
                search_queries = search_queries[:2]
                self.logger.info(f"Generated {len(search_queries)} search queries for claim {i}")
                
                # Step 2: Perform web searches with SerpAPI
                self.set_progress("research", (i-1)*50 + 15, f"Searching web for evidence on claim {i}...")
                
                search_results = []
                for j, query in enumerate(search_queries):
                    try:
                        # Check quota before each search
                        quota_ok, quota_message = check_serpapi_quota()
                        if not quota_ok:
                            self.logger.warning(f"SerpAPI quota exceeded during search: {quota_message}")
                            break
                        
                        # Rate limiting: wait between requests
                        if j > 0:  # Don't wait before first search
                            import time
                            delay = float(os.getenv("SERPAPI_DELAY_SECONDS", "1.5"))
                            time.sleep(delay)
                        
                        # Perform SerpAPI search
                        search_params = {
                            "q": query,
                            "api_key": serpapi_key,
                            "engine": "google",
                            "num": 3,  # Limit results per query
                            "safe": "active",
                            "hl": "en",
                            "gl": "us"
                        }
                        
                        search = GoogleSearch(search_params)
                        search_data = search.get_dict()
                        
                        # Increment usage counter
                        increment_serpapi_usage()
                        
                        # Extract organic results
                        organic_results = search_data.get("organic_results", [])
                        
                        for result in organic_results:
                            search_results.append({
                                'title': result.get('title', 'No title'),
                                'body': result.get('snippet', 'No content'),
                                'href': result.get('link', 'No URL'),
                                'position': result.get('position', 0),
                                'source': self._extract_domain(result.get('link', ''))
                            })
                        
                        self.logger.info(f"Found {len(organic_results)} results for query: {query[:50]}...")
                        
                    except Exception as search_error:
                        self.error_logger.error(f"SerpAPI search failed for query '{query[:50]}...': {search_error}")
                        continue
                
                # Step 3: Analyze web results with AI
                self.set_progress("research", (i-1)*50 + 35, f"Analyzing web evidence for claim {i}...")
                
                if search_results:
                    research_result = self._analyze_web_results(claim_text, search_results)
                    research_result['web_sources_found'] = len(search_results)
                    research_result['search_queries_used'] = search_queries
                    research_result['research_method'] = 'SerpAPI Google Search + AI analysis'
                else:
                    # No web results found, fallback to AI-only analysis
                    self.logger.warning(f"No web results found for claim {i}, using AI-only analysis")
                    research_result = self._perform_single_ai_research(claim_text)
                    research_result['web_sources_found'] = 0
                    research_result['search_queries_used'] = search_queries
                    research_result['research_method'] = 'AI-only analysis (no web results)'
                
                research_results.append(research_result)
                
                # Log detailed research outcome
                verification_status = research_result.get('verification_status', 'UNKNOWN')
                truthfulness_score = research_result.get('truthfulness_score', 0)
                sources_count = research_result.get('web_sources_found', 0)
                research_method = research_result.get('research_method', 'Unknown')
                
                self.logger.info(f"Web research completed for claim {i}: {verification_status} "
                               f"(Score: {truthfulness_score}/100, Sources: {sources_count}, Method: {research_method})")
                
            except Exception as e:
                self.error_logger.error(f"Web research failed for claim {i}: {e}")
                research_results.append(self._create_fallback_research_result(claim_text, f"Web research error: {str(e)}"))
        
        # Log final quota status
        final_quota_ok, final_quota_message = check_serpapi_quota()
        self.logger.info(f"Final SerpAPI quota status: {final_quota_message}")
        
        self.logger.info(f"Real web research completed for {len(research_results)} claims using SerpAPI")
        return research_results
    
    def _generate_search_queries(self, claim_text):
        """Generate effective search queries for fact-checking a claim"""
        try:
            query_prompt = f"""
            Generate 2 specific, effective search queries to fact-check this claim:
            
            CLAIM: "{claim_text}"
            
            Return ONLY a JSON array of search query strings, like:
            ["search query 1", "search query 2"]
            
            Make queries:
            - Specific and factual
            - Include key names, dates, locations mentioned
            - Focus on verifiable facts
            - Avoid opinion-based terms
            - Keep queries concise (under 50 characters each)
            """
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": query_prompt}]
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                if isinstance(queries, list):
                    # Limit to 2 queries and truncate long ones
                    return [str(q)[:100] for q in queries[:2]]
            
            # Fallback: create simple queries from claim
            return [claim_text[:80]]  # Truncate long claims
            
        except Exception as e:
            self.logger.warning(f"Failed to generate search queries: {e}")
            # Fallback: use claim text directly
            return [claim_text[:80]]
    
    def _analyze_web_results(self, claim_text, search_results):
        """Analyze web search results to assess claim truthfulness"""
        try:
            # Prepare web evidence for analysis
            web_evidence = []
            for result in search_results[:10]:  # Limit to top 10 results
                evidence_item = {
                    'title': result.get('title', 'No title'),
                    'snippet': result.get('body', 'No content'),
                    'url': result.get('href', 'No URL'),
                    'source': self._extract_domain(result.get('href', ''))
                }
                web_evidence.append(evidence_item)
            
            # Create analysis prompt with web evidence
            analysis_prompt = f"""
            As an expert fact-checker, analyze this claim using the provided web search results:
            
            CLAIM: "{claim_text}"
            
            WEB SEARCH RESULTS:
            {json.dumps(web_evidence, indent=2)}
            
            Based on the web evidence above, provide a comprehensive fact-check assessment in JSON format:
            {{
                "verification_status": "<VERIFIED|PARTIALLY_VERIFIED|DISPUTED|UNVERIFIABLE|FALSE>",
                "truthfulness_score": <0-100 integer>,
                "evidence_quality": "<STRONG|MODERATE|WEAK|INSUFFICIENT>",
                "research_summary": "<2-3 sentence summary based on web sources>",
                "supporting_evidence": ["Evidence from web sources supporting the claim"],
                "contradicting_evidence": ["Evidence from web sources contradicting the claim"],
                "verification_notes": "<detailed analysis of web evidence and conclusion>",
                "source_reliability": "<assessment of source quality>",
                "recommendation": "<ACCEPT|ACCEPT_WITH_CAUTION|QUESTION|REJECT>",
                "key_sources": ["domain1.com", "domain2.com"]
            }}
            
            Focus on information from the web results provided. Be objective and cite specific sources.
            """
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Add web research metadata
                analysis.update({
                    'claim': claim_text,
                    'research_method': 'SerpAPI Google Search + AI analysis',
                    'confidence_level': self._calculate_confidence_level(analysis),
                    'web_sources': [result.get('href', '') for result in search_results[:5]],
                    'search_result_positions': [result.get('position', 0) for result in search_results[:5]]
                })
                
                # Create human-readable status message
                status_messages = {
                    'VERIFIED': '✅ VERIFIED - Web sources confirm this claim',
                    'PARTIALLY_VERIFIED': '⚠️ PARTIALLY_VERIFIED - Mixed evidence from web sources',
                    'DISPUTED': '❌ DISPUTED - Conflicting information found online',
                    'UNVERIFIABLE': '❓ UNVERIFIABLE - Insufficient web evidence',
                    'FALSE': '❌ FALSE - Web sources contradict this claim'
                }
                
                analysis['status_message'] = status_messages.get(
                    analysis.get('verification_status', 'UNVERIFIABLE'),
                    '❓ STATUS UNKNOWN'
                )
                
                return analysis
            else:
                return self._create_fallback_research_result(claim_text, "Failed to parse web analysis")
                
        except Exception as e:
            self.error_logger.error(f"Failed to analyze web results: {e}")
            return self._create_fallback_research_result(claim_text, f"Web analysis error: {str(e)}")
    
    def _extract_domain(self, url):
        """Extract domain name from URL for source attribution"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        except:
            return 'Unknown source'
    
    def _perform_ai_only_research(self, claims):
        """Fallback method for AI-only research when web search is unavailable"""
        self.logger.info("Performing AI-only research (no web search)")
        research_results = []
        claims_to_research = claims[:3]
        
        for i, claim in enumerate(claims_to_research, 1):
            if isinstance(claim, dict):
                claim_text = str(claim.get('text', claim.get('claim', str(claim))))
            else:
                claim_text = str(claim)
            
            research_result = self._perform_single_ai_research(claim_text)
            research_result['web_sources_found'] = 0
            research_result['search_queries_used'] = []
            research_results.append(research_result)
        
        return research_results
    
    def _perform_single_ai_research(self, claim_text):
        """Perform AI-only analysis of a single claim"""
        try:
            research_prompt = f"""
            As a fact-checking research assistant, analyze this claim using your knowledge base:
            
            CLAIM: "{claim_text}"
            
            Provide assessment in JSON format:
            {{
                "verification_status": "<VERIFIED|PARTIALLY_VERIFIED|DISPUTED|UNVERIFIABLE|FALSE>",
                "truthfulness_score": <0-100 integer>,
                "evidence_quality": "<STRONG|MODERATE|WEAK|INSUFFICIENT>",
                "research_summary": "<2-3 sentence summary of findings>",
                "supporting_evidence": ["Evidence point 1", "Evidence point 2"],
                "contradicting_evidence": ["Contradiction 1", "Contradiction 2"],
                "verification_notes": "<detailed explanation>",
                "reliability_factors": ["Factor 1", "Factor 2"],
                "recommendation": "<ACCEPT|ACCEPT_WITH_CAUTION|QUESTION|REJECT>"
            }}
            
            Note: This analysis is based on training data only, not current web search.
            """
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": research_prompt}]
            )
            
            research_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', research_text, re.DOTALL)
            
            if json_match:
                analysis = json.loads(json_match.group())
                analysis.update({
                    'claim': claim_text,
                    'research_method': 'AI-only analysis (no web search)',
                    'confidence_level': self._calculate_confidence_level(analysis)
                })
                return analysis
            else:
                return self._create_fallback_research_result(claim_text, "AI analysis parsing failed")
                
        except Exception as e:
            return self._create_fallback_research_result(claim_text, f"AI research error: {str(e)}")
    
    def _calculate_confidence_level(self, research_analysis):
        """Calculate confidence level based on evidence quality and verification status"""
        evidence_quality = research_analysis.get('evidence_quality', 'INSUFFICIENT')
        verification_status = research_analysis.get('verification_status', 'UNVERIFIABLE')
        truthfulness_score = research_analysis.get('truthfulness_score', 0)
        
        if evidence_quality == 'STRONG' and verification_status in ['VERIFIED', 'FALSE']:
            return 'HIGH'
        elif evidence_quality == 'MODERATE' and verification_status != 'UNVERIFIABLE':
            return 'MEDIUM'
        elif evidence_quality == 'WEAK' or verification_status == 'UNVERIFIABLE':
            return 'LOW'
        else:
            return 'UNCERTAIN'
    
    def _create_fallback_research_result(self, claim_text, reason):
        """Create a fallback research result when research fails"""
        return {
            'claim': claim_text,
            'verification_status': 'UNVERIFIABLE',
            'truthfulness_score': 0,
            'evidence_quality': 'INSUFFICIENT',
            'research_summary': f'Research could not be completed: {reason}',
            'supporting_evidence': [],
            'contradicting_evidence': [],
            'verification_notes': f'Unable to complete research due to: {reason}',
            'reliability_factors': ['Research system error'],
            'recommendation': 'QUESTION',
            'research_method': 'AI-powered fact-checking analysis (failed)',
            'confidence_level': 'LOW',
            'status_message': '❌ RESEARCH FAILED - Unable to verify this claim due to technical issues'
        }
    
    def analyze_transcript(self, transcript_data, title):
        """Analyze transcript for credibility using LLM"""
        self.logger.info(f"Starting transcript analysis for title: '{title}'")
        start_time = datetime.now()
        
        self.set_progress("analysis", 0, "Preparing transcript for analysis...")
        
        # Handle both old string format and new detailed format for backward compatibility
        if isinstance(transcript_data, dict):
            transcript_text = transcript_data.get("full_text", "")
            segments = transcript_data.get("segments", [])
            language = transcript_data.get("language", "unknown")
            
            # Format segments with timestamps for analysis
            formatted_segments = []
            for segment in segments:
                start_time_str = f"{segment['start']:.1f}s"
                end_time_str = f"{segment['end']:.1f}s"
                formatted_segments.append(f"[{start_time_str}-{end_time_str}] {segment['text']}")
            
            timestamped_transcript = "\n".join(formatted_segments)
        else:
            # Backward compatibility for string transcripts
            transcript_text = transcript_data
            timestamped_transcript = transcript_text
            language = "unknown"
        
        transcript_length = len(transcript_text)
        self.logger.info(f"Transcript length: {transcript_length} characters, Language: {language}")
        
        self.set_progress("analysis", 20, f"Sending transcript to {self.model_name} for analysis...")
        
        # Generate speaker context for the prompt
        speaker_context = "Known Speaker Patterns for Reference:\n"
        for speaker, patterns in self.speaker_patterns.items():
            speaker_context += f"- {speaker}: Keywords: {patterns['keywords'][:3]}, Phrases: {patterns['distinctive_phrases'][:2]}\n"

        try:
            prompt = f"""
            Role: You are an expert multi-stage content analyst, fact-checker, and credibility assessor specializing in systematic evaluation of spoken content.
            
            Intent: Perform a comprehensive 4-stage analysis of the provided transcript to ultimately determine its credibility score (1-100), using a systematic approach that examines who, what, why, and how of the content.
            
            Steps: You must work through ALL 4 stages in sequence, building upon each stage's findings:
            
            STAGE 1: Initial Classification – Understand the Input
            • WHO is speaking? (journalist, politician, scientist, influencer, anonymous narrator, etc.)
            • WHAT KIND of content is this? (public speech, educational video, political statement, podcast excerpt, advertisement, vlog, news report, etc.)
            
            STAGE 2: Context & Intent – Understand the Purpose  
            • WHY is this being said? What is the speaker trying to achieve? (persuade, inform, entertain, provoke, sell, rally support, clarify)
            • FOR WHOM? Who is the target audience? (general public, specific demographic, followers, voters, students, professionals)
            
            STAGE 3: Content Analysis – Evaluate the Core Message
            • What are the MAIN CLAIMS or messages? Summarize the key points or arguments made
            • Is there any BIAS, TONE, or EMOTION present? Analyze tone (neutral, emotional, sarcastic, urgent) and potential bias (political, ideological, corporate interest)
            
            STAGE 4: Output Framing – Decide the Best Response Style
            • What is the IDEAL RESPONSE TYPE based on the analysis? (factual summary, rebuttal, neutral explanation, tone analysis, question list, response script)
            • FINAL CREDIBILITY SCORE (1-100) based on all stages of analysis
            
            {speaker_context}
            
            Example Multi-Stage Process: 
            "STAGE 1: Based on vocabulary patterns like 'tremendous' and 'believe me', this appears to be Donald Trump giving a political rally speech.
            STAGE 2: Purpose is to persuade supporters and rally opposition to political opponents; target audience is his voter base.
            STAGE 3: Main claims focus on economic achievements; tone is highly emotional and partisan with clear political bias.
            STAGE 4: Given the political context and emotional delivery, a balanced fact-checking approach is needed. Score: 45/100."
            
            Notes: Each stage builds context for the final credibility assessment. Political content, advertising, and entertainment require different evaluation criteria than educational or journalistic content.
            
            Please analyze the following transcript using the 4-stage methodology:
            
            Title: {title}
            Language: {language}
            
            Transcript with timestamps:
            {timestamped_transcript}
            
            FIRST: Show your complete 4-stage analysis process:
            
            STAGE 1 - INITIAL CLASSIFICATION:
            [Identify WHO is speaking and WHAT KIND of content this is]
            
            STAGE 2 - CONTEXT & INTENT:
            [Determine WHY this is being said and FOR WHOM]
            
            STAGE 3 - CONTENT ANALYSIS:
            [Evaluate MAIN CLAIMS and identify BIAS/TONE/EMOTION]
            
            STAGE 4 - OUTPUT FRAMING:
            [Determine IDEAL RESPONSE TYPE and assign FINAL CREDIBILITY SCORE]
            
            THEN: Provide your structured JSON analysis based on all 4 stages:
            
            IMPORTANT: All array fields must contain simple text strings only, not objects.
            
            Format your response as:
            MULTI-STAGE ANALYSIS:
            [Your complete 4-stage analysis process here]
            
            JSON ANALYSIS:
            {{
                "stage_1_classification": {{
                    "speaker_identified": "<speaker name or category>",
                    "speaker_confidence": <number 0-100>,
                    "content_type": "<type of content>",
                    "classification_reasoning": "<explanation of classification>"
                }},
                "stage_2_context": {{
                    "speaker_intent": "<what speaker is trying to achieve>",
                    "target_audience": "<who this is intended for>",
                    "context_reasoning": "<explanation of purpose and audience>"
                }},
                "stage_3_content": {{
                    "main_claims": ["First main claim with timestamp", "Second main claim with timestamp"],
                    "tone_analysis": "<neutral, emotional, sarcastic, urgent, etc.>",
                    "bias_indicators": ["First bias indicator", "Second bias indicator"],
                    "emotional_elements": ["First emotional element", "Second emotional element"],
                    "content_reasoning": "<explanation of content analysis>"
                }},
                "stage_4_output": {{
                    "ideal_response_type": "<factual summary, rebuttal, neutral explanation, etc.>",
                    "credibility_score": <number 1-100>,
                    "credibility_reasoning": "<detailed explanation of why this score was assigned>",
                    "key_factors": ["Factor 1 affecting credibility", "Factor 2 affecting credibility"]
                }},
                "comprehensive_analysis": {{
                    "red_flags": ["First red flag description", "Second red flag description"],
                    "factual_accuracy": "<overall assessment text>",
                    "evidence_quality": "<assessment of evidence presented>",
                    "timeline_analysis": "<assessment of timeline consistency>",
                    "final_assessment": "<overall credibility summary>"
                }},
                "meta_analysis": {{
                    "analysis_confidence": <number 0-100>,
                    "limitations": "<what factors might affect this analysis>",
                    "recommended_follow_up": "<what additional verification might be helpful>"
                }}
            }}
            
            Ensure your credibility score reflects the systematic analysis through all 4 stages, considering content type, intent, audience, claims quality, and evidence strength.
            """
            
            prompt_length = len(prompt)
            self.logger.info(f"Sending analysis prompt to {self.model_name} - Prompt length: {prompt_length} characters")
            self.api_logger.info(f"LLM API call initiated - Model: {self.model_name}, Prompt length: {prompt_length}")
            
            self.set_progress("analysis", 50, f"Waiting for {self.model_name} response...")
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.set_progress("analysis", 80, "Processing analysis response...")
            
            # Parse JSON response
            analysis_text = response.choices[0].message.content
            response_length = len(analysis_text)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            self.api_logger.info(f"LLM API call completed - Model: {self.model_name}, Response length: {response_length} chars, Duration: {elapsed_time:.2f}s")
            self.logger.info(f"Received analysis response in {elapsed_time:.2f}s - Length: {response_length} characters")
            
            # Extract chain of thought and JSON from response
            chain_of_thought = ""
            if "MULTI-STAGE ANALYSIS:" in analysis_text:
                cot_start = analysis_text.find("MULTI-STAGE ANALYSIS:") + len("MULTI-STAGE ANALYSIS:")
                if "JSON ANALYSIS:" in analysis_text:
                    cot_end = analysis_text.find("JSON ANALYSIS:")
                    chain_of_thought = analysis_text[cot_start:cot_end].strip()
                else:
                    # Fallback if JSON ANALYSIS marker not found
                    json_start = analysis_text.find("{")
                    if json_start > cot_start:
                        chain_of_thought = analysis_text[cot_start:json_start].strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                    
                    # Clean up the analysis data to ensure all list items are strings
                    def clean_list_items(items):
                        """Ensure all list items are converted to readable strings"""
                        if not isinstance(items, list):
                            return items
                        cleaned_items = []
                        for item in items:
                            if isinstance(item, str):
                                cleaned_items.append(item)
                            elif isinstance(item, dict):
                                # Extract text from common object keys
                                if 'text' in item:
                                    cleaned_items.append(str(item['text']))
                                elif 'claim' in item:
                                    cleaned_items.append(str(item['claim']))
                                elif 'description' in item:
                                    cleaned_items.append(str(item['description']))
                                else:
                                    # Convert entire object to readable string
                                    cleaned_items.append(str(item))
                            else:
                                cleaned_items.append(str(item))
                        return cleaned_items
                    
                    # Clean up all list fields
                    if 'key_claims' in analysis:
                        analysis['key_claims'] = clean_list_items(analysis['key_claims'])
                    if 'red_flags' in analysis:
                        analysis['red_flags'] = clean_list_items(analysis['red_flags'])
                    if 'bias_indicators' in analysis:
                        analysis['bias_indicators'] = clean_list_items(analysis['bias_indicators'])
                    
                    # Add extracted chain of thought if it wasn't in the JSON (fallback)
                    if chain_of_thought and 'chain_of_thought' not in analysis:
                        analysis['chain_of_thought'] = chain_of_thought
                    elif not analysis.get('chain_of_thought') and chain_of_thought:
                        analysis['chain_of_thought'] = chain_of_thought
                    
                    # Backward compatibility: Add old structure fields for frontend compatibility
                    analysis['credibility_score'] = analysis.get('stage_4_output', {}).get('credibility_score', 0)
                    analysis['key_claims'] = analysis.get('stage_3_content', {}).get('main_claims', [])
                    analysis['red_flags'] = analysis.get('comprehensive_analysis', {}).get('red_flags', [])
                    analysis['bias_indicators'] = analysis.get('stage_3_content', {}).get('bias_indicators', [])
                    analysis['factual_accuracy'] = analysis.get('comprehensive_analysis', {}).get('factual_accuracy', 'Unable to assess')
                    analysis['evidence_quality'] = analysis.get('comprehensive_analysis', {}).get('evidence_quality', 'Unable to assess')
                    analysis['timeline_analysis'] = analysis.get('comprehensive_analysis', {}).get('timeline_analysis', 'Unable to assess')
                    analysis['analysis_summary'] = analysis.get('comprehensive_analysis', {}).get('final_assessment', 'Unable to assess')
                    
                    # Speaker analysis for backward compatibility
                    analysis['speaker_analysis'] = {
                        'identified_speaker': analysis.get('stage_1_classification', {}).get('speaker_identified', 'Unknown'),
                        'confidence_score': analysis.get('stage_1_classification', {}).get('speaker_confidence', 0),
                        'identification_reasoning': analysis.get('stage_1_classification', {}).get('classification_reasoning', 'Unable to determine')
                    }
                    
                    # Handle speaker analysis logging
                    speaker_info = analysis.get('stage_1_classification', {})
                    identified_speaker = speaker_info.get('speaker_identified', 'Unknown')
                    speaker_confidence = speaker_info.get('speaker_confidence', 0)
                    
                    credibility_score = analysis.get('stage_4_output', {}).get('credibility_score', 'unknown')
                    main_claims = analysis.get('stage_3_content', {}).get('main_claims', [])
                    content_type = speaker_info.get('content_type', 'Unknown')
                    
                    self.logger.info(f"Successfully parsed multi-stage analysis JSON - Credibility score: {credibility_score}")
                    self.logger.info(f"Speaker identified: {identified_speaker} (confidence: {speaker_confidence}%)")
                    self.logger.info(f"Content type: {content_type}")
                    self.logger.info(f"Main claims sample: {main_claims[:2]}")  # Log first 2 claims for debugging
                    self.logger.info(f"Multi-stage analysis length: {len(analysis.get('meta_analysis', {}).get('limitations', ''))}")
                    self.set_progress("analysis", 100, f"Analysis complete - Speaker: {identified_speaker} ({speaker_confidence}%), Type: {content_type}, Score: {credibility_score}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parsing failed: {e}")
                    analysis = self._create_fallback_analysis(analysis_text, "JSON parsing error")
                    self.set_progress("analysis", 100, "Analysis complete (with parsing issues)")
            else:
                self.logger.warning(f"No valid JSON found in {self.model_name} response")
                analysis = self._create_fallback_analysis(analysis_text, "No JSON found")
                self.set_progress("analysis", 100, "Analysis complete (with formatting issues)")
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Transcript analysis completed in {total_time:.2f}s")
            return analysis
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Transcript analysis failed after {elapsed_time:.2f}s for title '{title}': {e}")
            self.api_logger.error(f"LLM API call failed - Model: {self.model_name}: {e}")
            self.set_progress("analysis", 0, f"Analysis failed: {str(e)}")
            return self._create_fallback_analysis(str(e), f"Analysis error: {str(e)}")
    
    def _create_fallback_analysis(self, text, reason):
        """Create a fallback analysis when parsing fails"""
        self.logger.warning(f"Creating fallback analysis due to: {reason}")
        return {
            "stage_1_classification": {
                "speaker_identified": "Unknown",
                "speaker_confidence": 0,
                "content_type": "Unknown",
                "classification_reasoning": f"Analysis failed: {reason}"
            },
            "stage_2_context": {
                "speaker_intent": "Unknown",
                "target_audience": "Unknown",
                "context_reasoning": f"Analysis failed: {reason}"
            },
            "stage_3_content": {
                "main_claims": ["Analysis failed"],
                "tone_analysis": "Unknown",
                "bias_indicators": ["Analysis error"],
                "emotional_elements": ["Analysis error"],
                "content_reasoning": f"Analysis failed: {reason}"
            },
            "stage_4_output": {
                "ideal_response_type": "Unknown",
                "credibility_score": 0,
                "credibility_reasoning": f"Analysis failed: {reason}",
                "key_factors": ["Analysis error"]
            },
            "comprehensive_analysis": {
                "red_flags": ["Analysis error"],
                "factual_accuracy": "Unable to assess",
                "evidence_quality": "Unable to assess",
                "timeline_analysis": "Unable to assess",
                "final_assessment": "Unable to assess"
            },
            "meta_analysis": {
                "analysis_confidence": 0,
                "limitations": f"Analysis could not be completed: {reason}. Raw response: {text[:200]}...",
                "recommended_follow_up": f"No additional verification recommended. Raw response preview: {text[:300]}..."
            },
            # Backward compatibility fields
            "credibility_score": 0,
            "key_claims": ["Analysis failed"],
            "red_flags": ["Analysis error"],
            "bias_indicators": ["Analysis error"],
            "factual_accuracy": "Unable to assess",
            "evidence_quality": "Unable to assess",
            "timeline_analysis": "Unable to assess",
            "analysis_summary": f"Analysis could not be completed: {reason}. Raw response: {text[:200]}...",
            "speaker_analysis": {
                "identified_speaker": "Unknown",
                "confidence_score": 0,
                "identification_reasoning": f"Analysis failed: {reason}"
            }
        }

analyzer = TruthScoreAnalyzer()

@app.route('/')
def index():
    app_logger.info(f"Home page accessed from {request.remote_addr}")
    return render_template('index.html')

def background_analyze(request_id, url):
    """Background function to perform the analysis"""
    analyzer.set_request_id(request_id)
    
    try:
        # Step 1: Extract audio
        app_logger.info(f"[{request_id}] Step 1: Starting audio extraction")
        audio_path, title = analyzer.extract_audio(url)
        if not audio_path:
            app_logger.error(f"[{request_id}] Step 1 failed: Audio extraction failed")
            results_store[request_id] = {'error': 'Failed to extract audio from URL', 'status': 'error', 'timestamp': datetime.now().isoformat()}
            save_request_tracking()
            return
        
        # Step 2: Transcribe audio with speaker diarization
        app_logger.info(f"[{request_id}] Step 2: Starting audio transcription with speaker diarization")
        transcript = analyzer.transcribe_with_speakers(audio_path)
        if not transcript:
            app_logger.error(f"[{request_id}] Step 2 failed: Audio transcription failed")
            results_store[request_id] = {'error': 'Failed to transcribe audio. This may be due to a timeout for very long videos (>5 minutes processing time) or unsupported audio format.', 'status': 'error', 'timestamp': datetime.now().isoformat()}
            save_request_tracking()
            return
        
        # Step 3: Analyze transcript
        app_logger.info(f"[{request_id}] Step 3: Starting transcript analysis")
        analysis = analyzer.analyze_transcript(transcript, title)
        
        # Step 4: Perform comprehensive web research
        app_logger.info(f"[{request_id}] Step 4: Starting comprehensive web research")
        analyzer.set_progress("research", 0, "Starting detailed web research...")
        # Extract claims from the new multi-stage structure
        main_claims = analysis.get('stage_3_content', {}).get('main_claims', [])
        research = analyzer.perform_web_research(main_claims)
        
        # Log detailed research outcomes
        if research:
            app_logger.info(f"[{request_id}] Research completed for {len(research)} claims:")
            for i, result in enumerate(research, 1):
                verification_status = result.get('verification_status', 'UNKNOWN')
                truthfulness_score = result.get('truthfulness_score', 0)
                evidence_quality = result.get('evidence_quality', 'UNKNOWN')
                recommendation = result.get('recommendation', 'UNKNOWN')
                web_sources_count = result.get('web_sources_found', 0)
                research_method = result.get('research_method', 'Unknown')
                claim_preview = str(result.get('claim', ''))[:80] + "..." if len(str(result.get('claim', ''))) > 80 else str(result.get('claim', ''))
                
                app_logger.info(f"[{request_id}] Claim {i}: {verification_status} (Score: {truthfulness_score}/100, Evidence: {evidence_quality}, Rec: {recommendation})")
                app_logger.info(f"[{request_id}] Claim {i} method: {research_method}, Web sources: {web_sources_count}")
                app_logger.info(f"[{request_id}] Claim {i} text: {claim_preview}")
                
                if result.get('research_summary'):
                    summary_preview = result.get('research_summary', '')[:120] + "..." if len(result.get('research_summary', '')) > 120 else result.get('research_summary', '')
                    app_logger.info(f"[{request_id}] Claim {i} summary: {summary_preview}")
        
        analyzer.set_progress("research", 100, "Detailed web research complete")
        
        # Save enhanced transcript if speaker diarization was performed
        if transcript.get("has_speaker_diarization", False):
            app_logger.info(f"[{request_id}] Saving enhanced transcript with speaker information")
            try:
                base_filename = f"transcript_{request_id}"
                output_path = os.path.join(analyzer.temp_dir, base_filename)
                saved_files = analyzer.save_transcript_with_speakers(transcript, output_path)
                if saved_files:
                    app_logger.info(f"[{request_id}] Enhanced transcript saved to: {saved_files}")
            except Exception as save_error:
                app_logger.warning(f"[{request_id}] Failed to save enhanced transcript: {save_error}")
        
        # Clean up temp files
        app_logger.info(f"[{request_id}] Cleaning up temporary files")
        analyzer.set_progress("cleanup", 50, "Cleaning up temporary files...")
        try:
            os.remove(audio_path)
            app_logger.info(f"[{request_id}] Successfully removed temporary file: {audio_path}")
            analyzer.set_progress("cleanup", 100, "Cleanup complete")
        except Exception as cleanup_error:
            app_logger.warning(f"[{request_id}] Failed to remove temporary file: {cleanup_error}")
            analyzer.set_progress("cleanup", 100, "Cleanup finished (with warnings)")
        
        analyzer.set_progress("complete", 100, "Analysis complete!")
        
        # Prepare transcript data for response (handle both formats)
        if isinstance(transcript, dict):
            transcript_text = transcript.get("full_text", "")
            transcript_data = transcript
        else:
            transcript_text = transcript
            transcript_data = {"full_text": transcript, "segments": [], "language": "unknown"}
        
        result = {
            'title': title,
            'transcript': transcript_data,  # Include full transcript data with timestamps
            'analysis': analysis,
            'research': research,
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'request_id': request_id,
            'status': 'completed'
        }
        
        # Store results
        results_store[request_id] = result
        # Save results data persistently
        save_request_tracking()
        
        # Log successful completion
        credibility_score = analysis.get('stage_4_output', {}).get('credibility_score', 'unknown')
        transcript_length = len(transcript_text)
        segments_count = len(transcript_data.get('segments', []))
        content_type = analysis.get('stage_1_classification', {}).get('content_type', 'Unknown')
        app_logger.info(f"[{request_id}] Analysis completed successfully - Title: '{title}', Score: {credibility_score}, Type: {content_type}, Transcript length: {transcript_length}, Segments: {segments_count}")
        
    except Exception as e:
        app_logger.error(f"[{request_id}] Analysis failed with exception: {str(e)}")
        error_logger = logging.getLogger('errors')
        error_logger.error(f"[{request_id}] Unhandled exception in analyze route: {str(e)}", exc_info=True)
        
        # Store error result
        results_store[request_id] = {'error': f'Analysis failed: {str(e)}', 'status': 'error', 'timestamp': datetime.now().isoformat()}
        # Save error result persistently
        save_request_tracking()
        
        # Clean up progress data on error
        if request_id in progress_store:
            del progress_store[request_id]
            save_request_tracking()

@app.route('/analyze', methods=['POST'])
def analyze():
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    client_ip = request.remote_addr
    app_logger.info(f"[{request_id}] Analysis request received from {client_ip}")
    
    try:
        data = request.get_json()
        url = data.get('url')
        
        app_logger.info(f"[{request_id}] Analysis requested for URL: {url}")
        
        if not url:
            app_logger.warning(f"[{request_id}] No URL provided in request")
            return jsonify({'error': 'No URL provided'}), 400
        
        if not analyzer.is_valid_url(url):
            app_logger.warning(f"[{request_id}] Invalid URL format: {url}")
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Start background processing
        thread = threading.Thread(target=background_analyze, args=(request_id, url))
        thread.daemon = True
        thread.start()
        
        # Return immediately with request ID
        return jsonify({
            'request_id': request_id,
            'status': 'processing',
            'message': 'Analysis started. Use the request_id to poll for progress and results.'
        })
        
    except Exception as e:
        app_logger.error(f"[{request_id}] Failed to start analysis: {str(e)}")
        return jsonify({'error': f'Failed to start analysis: {str(e)}'}), 500

@app.route('/results/<request_id>')
def get_results(request_id):
    """Get results for a specific request"""
    if request_id in results_store:
        result = results_store[request_id]
        # Return results but DON'T delete them immediately to avoid race conditions
        # Results will be cleaned up after 1 hour by cleanup_old_requests()
        return jsonify(result)
    else:
        # Check if we have any progress data to indicate it's still processing
        if request_id in progress_store:
            return jsonify({'status': 'processing', 'message': 'Analysis in progress'}), 202
        else:
            return jsonify({'status': 'unknown', 'message': 'Request not found - may have timed out or failed'}), 404

@app.route('/progress/<request_id>')
def get_progress(request_id):
    """Get progress for a specific request"""
    if request_id in progress_store:
        return jsonify(progress_store[request_id])
    elif request_id in results_store:
        # Analysis is complete
        result = results_store[request_id]
        if result.get('status') == 'completed':
            return jsonify({'step': 'complete', 'percentage': 100, 'message': 'Analysis complete!', 'timestamp': result.get('timestamp')})
        elif result.get('status') == 'error':
            return jsonify({'step': 'error', 'percentage': 0, 'message': f'Analysis failed: {result.get("error", "Unknown error")}', 'timestamp': result.get('timestamp')})
        else:
            return jsonify({'step': 'unknown', 'percentage': 50, 'message': 'Analysis status unknown'})
    else:
        # Be more lenient - return processing status instead of 404 to avoid frontend errors
        return jsonify({'step': 'processing', 'percentage': 0, 'message': 'Request not found in progress tracking - may be initializing or completed'}), 200

@app.route('/transcribe_speakers', methods=['POST'])
def transcribe_with_speaker_diarization():
    """Dedicated endpoint for transcription with speaker diarization from uploaded audio files"""
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    client_ip = request.remote_addr
    app_logger.info(f"[{request_id}] Speaker diarization request received from {client_ip}")
    
    analyzer.set_request_id(request_id)
    
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            app_logger.warning(f"[{request_id}] No audio file provided in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            app_logger.warning(f"[{request_id}] Empty filename provided")
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if speaker diarization is available
        if not analyzer.diarization_model:
            app_logger.warning(f"[{request_id}] Speaker diarization not available")
            return jsonify({
                'error': 'Speaker diarization not available. Please set HF_TOKEN environment variable.'
            }), 503
        
        # Save uploaded file
        filename = f"upload_{request_id}_{audio_file.filename}"
        audio_path = os.path.join(analyzer.temp_dir, filename)
        audio_file.save(audio_path)
        
        app_logger.info(f"[{request_id}] Audio file saved: {filename}")
        
        # Perform transcription with speaker diarization
        app_logger.info(f"[{request_id}] Starting transcription with speaker diarization")
        transcript = analyzer.transcribe_with_speakers(audio_path)
        
        if not transcript:
            app_logger.error(f"[{request_id}] Transcription failed")
            return jsonify({'error': 'Failed to transcribe audio file'}), 400
        
        # Save enhanced transcript
        base_filename = f"transcript_{request_id}"
        output_path = os.path.join(analyzer.temp_dir, base_filename)
        saved_files = analyzer.save_transcript_with_speakers(transcript, output_path)
        
        # Clean up uploaded file
        try:
            os.remove(audio_path)
            app_logger.info(f"[{request_id}] Successfully removed uploaded file: {audio_path}")
        except Exception as cleanup_error:
            app_logger.warning(f"[{request_id}] Failed to remove uploaded file: {cleanup_error}")
        
        # Prepare response
        result = {
            'transcript': transcript,
            'saved_files': saved_files,
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'speaker_count': transcript.get('speaker_count', 0),
            'has_speaker_diarization': transcript.get('has_speaker_diarization', False)
        }
        
        app_logger.info(f"[{request_id}] Speaker diarization completed successfully - Speakers: {transcript.get('speaker_count', 0)}")
        
        return jsonify(result)
        
    except Exception as e:
        app_logger.error(f"[{request_id}] Speaker diarization failed: {e}")
        return jsonify({'error': f'Speaker diarization failed: {str(e)}'}), 500

@app.route('/health')
def health():
    app_logger.info(f"Health check accessed from {request.remote_addr}")
    
    # Check SerpAPI status
    serpapi_status = 'unavailable'
    serpapi_quota = 'unknown'
    if WEB_SEARCH_AVAILABLE:
        serpapi_key = os.getenv("SERPAPI_KEY")
        if serpapi_key:
            quota_ok, quota_message = check_serpapi_quota()
            serpapi_status = 'available' if quota_ok else 'quota_exceeded'
            serpapi_quota = quota_message
        else:
            serpapi_status = 'no_api_key'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'whisper_model': 'available',
        'openai_api': 'configured',
        'speaker_diarization': 'available' if analyzer.diarization_model else 'unavailable',
        'web_search': serpapi_status,
        'web_search_quota': serpapi_quota,
        'model_name': analyzer.model_name
    })

if __name__ == '__main__':
    app_logger.info("Starting TruthScore Flask application")
    app_logger.info("Application will be available at: http://localhost:8000")
    try:
        app.run(debug=True, host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        app_logger.info("Application stopped by user")
    except Exception as e:
        app_logger.error(f"Failed to start application: {e}")
        raise 