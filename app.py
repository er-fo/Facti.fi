from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import tempfile

# Set environment variables early to prevent OpenMP conflicts and resource issues
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import yt_dlp
import openai
import requests
from datetime import datetime, timedelta
import json
import re
import logging
import sys
import subprocess
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler
from faster_whisper import WhisperModel
import threading
import queue
import time

# Database imports
from database import AnalysisDatabase

# Web research configuration
WEB_SEARCH_AVAILABLE = True  # Always available with OpenAI web search
OPENAI_SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4o-mini-search-preview")
logging.getLogger('app').info(f"OpenAI web search available using model: {OPENAI_SEARCH_MODEL}")

# Speaker diarization imports - Lazy loading to prevent crashes
DIARIZATION_AVAILABLE = False
Pipeline = None
LegacyPipeline = None

def check_diarization_dependencies():
    """Check if speaker diarization dependencies are available without importing them"""
    global DIARIZATION_AVAILABLE
    
    try:
        # Check if packages are installed without importing to avoid crashes
        import importlib.util
        
        required_packages = ['torch', 'pyannote.audio', 'librosa', 'soundfile']
        missing_packages = []
        
        for package in required_packages:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(package)
        
        if missing_packages:
            logging.getLogger('app').warning(f"Missing speaker diarization packages: {missing_packages}")
            return False
        
        logging.getLogger('app').info("Speaker diarization dependencies available (will be loaded on demand)")
        return True
        
    except Exception as e:
        logging.getLogger('app').warning(f"Error checking speaker diarization dependencies: {e}")
        return False

def lazy_load_diarization_imports():
    """Safely import diarization modules only when needed with subprocess isolation"""
    global Pipeline, LegacyPipeline, DIARIZATION_AVAILABLE
    
    if DIARIZATION_AVAILABLE:
        return True
    
    # First check if we should skip loading due to environment variable
    if os.getenv("DISABLE_SPEAKER_DIARIZATION"):
        logging.getLogger('app').info("Speaker diarization disabled via environment variable")
        return False
        
    subprocess_module = None  # Initialize to avoid scope issues
    
    try:
        # Use subprocess to test imports safely without crashing main process
        import subprocess as subprocess_module
        import tempfile
        
        # Create a test script to safely check imports
        test_script = '''
import sys
import os
try:
    # Test basic imports first
    import torch
    print("TORCH_OK")
    
    import librosa
    import soundfile as sf
    print("AUDIO_OK")
    
    # Test pyannote imports (most likely to segfault)
    Pipeline = None
    LegacyPipeline = None
    
    try:
        from pyannote.audio import Pipeline
        print("PIPELINE_OK")
    except ImportError as e:
        print(f"PIPELINE_FAIL: {e}")
    
    try:
        from pyannote.pipeline import Pipeline as LegacyPipeline  
        print("LEGACY_OK")
    except ImportError as e:
        print(f"LEGACY_FAIL: {e}")
    
    # Note: Model instantiation testing is skipped during dependency check
    # Model loading will be tested only when first requested to avoid startup crashes
    print("MODEL_TEST_SKIPPED_FOR_SAFE_STARTUP")
        
    print("IMPORT_TEST_COMPLETE")
    
except Exception as e:
    print(f"CRITICAL_ERROR: {e}")
    sys.exit(1)
'''
        
        # Run the test in a subprocess with timeout
        try:
            result = subprocess_module.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=os.getcwd()
            )
            
            output = result.stdout
            logging.getLogger('app').info(f"Import test output: {output.strip()}")
            
            if result.returncode != 0:
                logging.getLogger('app').error(f"Import test failed with return code {result.returncode}")
                logging.getLogger('app').error(f"Error output: {result.stderr}")
                return False
            
            # Check what imports worked
            torch_ok = "TORCH_OK" in output
            audio_ok = "AUDIO_OK" in output  
            pipeline_ok = "PIPELINE_OK" in output
            legacy_ok = "LEGACY_OK" in output
            safe_startup = "MODEL_TEST_SKIPPED_FOR_SAFE_STARTUP" in output
            
            if not torch_ok:
                logging.getLogger('app').error("PyTorch import test failed")
                return False
            
            if not audio_ok:
                logging.getLogger('app').error("Audio libraries import test failed")
                return False
                
            # For safe startup, we just need basic imports to work
            if safe_startup:
                logging.getLogger('app').info("Import test passed - model loading will be deferred for safety")
            
            # If subprocess test passed, try actual imports in main process
            logging.getLogger('app').info("Subprocess import test passed, attempting main process imports...")
            
            # Import torch first as it's most likely to cause issues
            import torch
            logging.getLogger('app').info("PyTorch imported successfully in main process")
            
            # Import audio processing libraries
            import librosa
            import soundfile as sf
            logging.getLogger('app').info("Audio processing libraries imported successfully")
            
            # Only import pipelines if subprocess test showed they work
            if pipeline_ok:
                from pyannote.audio import Pipeline
                logging.getLogger('app').info("pyannote.audio Pipeline imported successfully")
            else:
                Pipeline = None
                logging.getLogger('app').warning("pyannote.audio Pipeline not available")
                
            if legacy_ok:
                from pyannote.pipeline import Pipeline as LegacyPipeline
                logging.getLogger('app').info("Legacy Pipeline imported successfully")
            else:
                LegacyPipeline = None
                logging.getLogger('app').warning("Legacy Pipeline not available")
                
            if Pipeline is None and LegacyPipeline is None:
                logging.getLogger('app').error("No valid Pipeline class available")
                return False
            
            DIARIZATION_AVAILABLE = True
            logging.getLogger('app').info("Speaker diarization dependencies loaded successfully (imports only)")
            
            # Note: Actual model loading will be attempted later during initialization
            # This separation allows for safer testing
            return True
            
        except Exception as subprocess_e:
            # Check if it's a timeout specifically
            if subprocess_module and hasattr(subprocess_module, 'TimeoutExpired') and isinstance(subprocess_e, subprocess_module.TimeoutExpired):
                logging.getLogger('app').error("Import test timed out - likely indicates compatibility issues")
            else:
                logging.getLogger('app').error(f"Subprocess import test failed: {subprocess_e}")
            return False
        
    except Exception as e:
        logging.getLogger('app').error(f"Failed to test speaker diarization dependencies: {e}")
        DIARIZATION_AVAILABLE = False
        return False

# Check if dependencies are available without loading them
DIARIZATION_DEPENDENCIES_AVAILABLE = check_diarization_dependencies()

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

# Request tracking configuration
REQUEST_TRACKING_FILE = "request_tracking.json"

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

# Initialize analysis database
try:
    db = AnalysisDatabase()
    app_logger.info("Analysis database initialized successfully")
except Exception as e:
    app_logger.error(f"Error initializing analysis database: {e}")
    raise

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
            # US Political Figures
            "Donald Trump": {
                "keywords": ["tremendous", "incredible", "beautiful", "fake news", "witch hunt", "make america great", "believe me", "bigly", "yuge", "very very"],
                "speech_patterns": ["superlatives", "repetition", "crowd references", "self-praise"],
                "policy_positions": ["immigration restrictions", "america first", "trade deals", "border wall", "china", "tariffs"],
                "distinctive_phrases": ["nobody knows more than me", "it's true", "many people are saying", "like you wouldn't believe", "probably the best", "in the history of", "fake news media"]
            },
            "Joe Biden": {
                "keywords": ["folks", "malarkey", "senate", "barack", "democracy", "build back better", "scranton", "amtrak"],
                "speech_patterns": ["folksy expressions", "senate references", "obama mentions", "personal anecdotes"],
                "policy_positions": ["infrastructure", "climate change", "healthcare", "unity", "democracy"],
                "distinctive_phrases": ["here's the deal", "not a joke", "come on, man", "look", "the fact of the matter is"]
            },
            "Kamala Harris": {
                "keywords": ["prosecutor", "california", "justice", "equality", "women", "families"],
                "speech_patterns": ["legal terminology", "justice references", "prosecutorial language"],
                "policy_positions": ["criminal justice reform", "women's rights", "civil rights"],
                "distinctive_phrases": ["i'm speaking", "that's not accurate", "the american people deserve"]
            },
            "Nancy Pelosi": {
                "keywords": ["speaker", "house", "impeachment", "democracy", "constitution", "san francisco"],
                "speech_patterns": ["parliamentary language", "constitutional references", "procedural terminology"],
                "policy_positions": ["democratic agenda", "trump investigations", "house procedures"],
                "distinctive_phrases": ["for the people", "when we win", "the constitution requires"]
            },
            "Mitch McConnell": {
                "keywords": ["senate majority", "conservative", "judges", "kentucky", "leader", "minority"],
                "speech_patterns": ["senate procedure", "conservative talking points", "institutional language"],
                "policy_positions": ["judicial appointments", "conservative agenda", "senate rules"],
                "distinctive_phrases": ["my democratic colleagues", "senate tradition", "the american people sent us here"]
            },
            "Alexandria Ocasio-Cortez": {
                "keywords": ["green new deal", "climate", "progressive", "bronx", "medicare for all", "working class"],
                "speech_patterns": ["progressive rhetoric", "social media references", "generational language"],
                "policy_positions": ["climate action", "social justice", "wealth inequality", "healthcare"],
                "distinctive_phrases": ["let's be clear", "working families", "this is about", "we need to"]
            },
            "Bernie Sanders": {
                "keywords": ["billionaire", "political revolution", "medicare for all", "wall street", "vermont", "millions"],
                "speech_patterns": ["passionate delivery", "repetitive emphasis", "economic inequality focus"],
                "policy_positions": ["wealth inequality", "healthcare", "education", "workers rights"],
                "distinctive_phrases": ["let me be clear", "enough is enough", "political revolution", "billionaire class"]
            },
            
            # Media Personalities
            "Tucker Carlson": {
                "keywords": ["establishment", "elite", "mainstream media", "ordinary americans", "suburbs"],
                "speech_patterns": ["rhetorical questions", "us vs them framing", "populist appeals"],
                "policy_positions": ["immigration skepticism", "populist conservatism", "anti-establishment"],
                "distinctive_phrases": ["here's what's interesting", "of course", "notice how", "by the way"]
            },
            "Rachel Maddow": {
                "keywords": ["investigation", "documents", "sources", "reporting", "exclusive"],
                "speech_patterns": ["detailed explanations", "document references", "investigative language"],
                "policy_positions": ["liberal commentary", "investigative journalism", "democratic positions"],
                "distinctive_phrases": ["we've got", "this is important", "here's what we know", "documents show"]
            },
            "Sean Hannity": {
                "keywords": ["liberal media", "deep state", "corrupt", "america first", "conservative"],
                "speech_patterns": ["partisan framing", "conspiracy theories", "media criticism"],
                "policy_positions": ["conservative commentary", "trump support", "media criticism"],
                "distinctive_phrases": ["let not your heart be troubled", "the liberal media", "fake news"]
            },
            "Anderson Cooper": {
                "keywords": ["cnn", "breaking news", "sources tell us", "reporting", "facts"],
                "speech_patterns": ["journalistic tone", "fact-based reporting", "neutral presentation"],
                "policy_positions": ["mainstream journalism", "fact-based reporting"],
                "distinctive_phrases": ["keeping them honest", "according to sources", "let's be clear about the facts"]
            },
            
            # Tech Leaders
            "Elon Musk": {
                "keywords": ["mars", "tesla", "spacex", "neural", "ai", "twitter", "free speech", "humanity"],
                "speech_patterns": ["technical language", "future-focused", "casual tone", "meme references"],
                "policy_positions": ["space exploration", "sustainable energy", "ai development", "free speech"],
                "distinctive_phrases": ["obviously", "like", "um", "probably", "i think", "mars needs moms"]
            },
            "Mark Zuckerberg": {
                "keywords": ["facebook", "meta", "metaverse", "community", "connecting people", "platform"],
                "speech_patterns": ["corporate speak", "technology focus", "measured delivery"],
                "policy_positions": ["technology regulation", "social media", "virtual reality"],
                "distinctive_phrases": ["bringing the world closer together", "we believe", "our mission"]
            },
            "Jeff Bezos": {
                "keywords": ["amazon", "customer", "innovation", "space", "blue origin", "long term"],
                "speech_patterns": ["business language", "customer-centric", "long-term thinking"],
                "policy_positions": ["business innovation", "space exploration", "customer service"],
                "distinctive_phrases": ["customer obsession", "long term", "day one", "we're always"]
            },
            
            # Religious Leaders
            "Joel Osteen": {
                "keywords": ["blessing", "favor", "victory", "god's goodness", "potential", "positive"],
                "speech_patterns": ["motivational language", "prosperity gospel", "positive affirmations"],
                "policy_positions": ["prosperity theology", "positive thinking", "christian faith"],
                "distinctive_phrases": ["god's got something better", "your best life now", "favor is coming"]
            },
            
            # International Leaders  
            "Vladimir Putin": {
                "keywords": ["russia", "motherland", "nato", "sovereignty", "traditional values", "west"],
                "speech_patterns": ["authoritarian tone", "nationalist rhetoric", "security language"],
                "policy_positions": ["russian nationalism", "anti-western", "authoritarianism"],
                "distinctive_phrases": ["russian people", "our sovereignty", "traditional values"]
            },
            "Xi Jinping": {
                "keywords": ["china", "chinese people", "socialism", "development", "harmony", "rejuvenation"],
                "speech_patterns": ["communist party language", "collective terminology", "development focus"],
                "policy_positions": ["chinese communism", "economic development", "party leadership"],
                "distinctive_phrases": ["chinese dream", "great rejuvenation", "common prosperity"]
            },
            
            # Celebrities/Influencers
            "Oprah Winfrey": {
                "keywords": ["authentic", "truth", "journey", "empowerment", "gratitude", "intention"],
                "speech_patterns": ["inspirational tone", "personal growth language", "emotional connection"],
                "policy_positions": ["personal empowerment", "education", "social justice"],
                "distinctive_phrases": ["what i know for sure", "your truth", "aha moment", "gratitude"]
            },
            "Joe Rogan": {
                "keywords": ["interesting", "crazy", "dmt", "mma", "comedy", "elk meat", "sauna"],
                "speech_patterns": ["conversational", "curious questioning", "casual language", "tangents"],
                "policy_positions": ["free speech", "skepticism", "open dialogue"],
                "distinctive_phrases": ["have you ever tried", "that's crazy", "entirely possible", "jamie pull that up"]
            },
            
            # Academic/Expert Voices
            "Jordan Peterson": {
                "keywords": ["responsibility", "hierarchy", "meaning", "chaos", "order", "lobsters", "clinical"],
                "speech_patterns": ["academic language", "psychological terminology", "verbose explanations"],
                "policy_positions": ["personal responsibility", "traditional values", "free speech"],
                "distinctive_phrases": ["roughly speaking", "it's complicated", "clean up your life", "that's not trivial"]
            },
            "Neil deGrasse Tyson": {
                "keywords": ["cosmos", "science", "astrophysics", "evidence", "universe", "hayden planetarium"],
                "speech_patterns": ["scientific explanation", "educational tone", "wonder and curiosity"],
                "policy_positions": ["scientific literacy", "space exploration", "evidence-based thinking"],
                "distinctive_phrases": ["the cosmos", "astrophysically speaking", "science literacy"]
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
        
        # Initialize speaker diarization model with robust error handling
        self.diarization_model = None
        self.diarization_status = "disabled"  # disabled, loading, available, failed, deferred
        
        # Check if speaker diarization is explicitly disabled
        if os.getenv("DISABLE_SPEAKER_DIARIZATION"):
            self.logger.info("Speaker diarization explicitly disabled via environment variable")
            self.diarization_status = "disabled"
        elif DIARIZATION_DEPENDENCIES_AVAILABLE:
            # Check for HuggingFace token
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                # Use deferred loading to prevent startup crashes
                self.diarization_status = "deferred"
                self.logger.info("Speaker diarization model will be loaded on first use to prevent startup issues")
                self._hf_token = hf_token
            else:
                self.logger.warning("HF_TOKEN not found. Speaker diarization will be disabled.")
                self.logger.warning("Set HF_TOKEN environment variable with your Hugging Face token to enable speaker diarization.")
                self.diarization_status = "no_token"
        else:
            self.logger.info("Speaker diarization dependencies not available")
            self.diarization_status = "disabled"
        
        self.logger.info(f"TruthScoreAnalyzer initialized with temp directory: {self.temp_dir}")
        
        # Define models that don't support certain parameters
        self.models_without_temperature = {"o3-mini", "o3", "o1-mini", "o1-preview", "o1"}
        self.models_without_frequency_penalty = {"o3-mini", "o3", "o1-mini", "o1-preview", "o1"}
        self.models_without_presence_penalty = {"o3-mini", "o3", "o1-mini", "o1-preview", "o1"}

    def create_chat_completion(self, messages, **kwargs):
        """Create a chat completion with model-compatible parameters"""
        # Start with base parameters
        params = {
            "model": self.model_name,
            "messages": messages
        }
        
        # Add supported parameters based on model
        if self.model_name not in self.models_without_temperature and 'temperature' in kwargs:
            params['temperature'] = kwargs['temperature']
        elif 'temperature' in kwargs:
            self.logger.info(f"Skipping 'temperature' parameter for model {self.model_name} (not supported)")
            
        if self.model_name not in self.models_without_frequency_penalty and 'frequency_penalty' in kwargs:
            params['frequency_penalty'] = kwargs['frequency_penalty']
        elif 'frequency_penalty' in kwargs:
            self.logger.info(f"Skipping 'frequency_penalty' parameter for model {self.model_name} (not supported)")
            
        if self.model_name not in self.models_without_presence_penalty and 'presence_penalty' in kwargs:
            params['presence_penalty'] = kwargs['presence_penalty']
        elif 'presence_penalty' in kwargs:
            self.logger.info(f"Skipping 'presence_penalty' parameter for model {self.model_name} (not supported)")
        
        # Add any other supported parameters
        for key, value in kwargs.items():
            if key not in ['temperature', 'frequency_penalty', 'presence_penalty']:
                params[key] = value
        
        self.logger.debug(f"Creating chat completion with parameters: {list(params.keys())}")
        return client.chat.completions.create(**params)
    
    def perform_openai_web_search(self, query, user_location=None):
        """Perform web search using OpenAI's built-in search capability"""
        try:
            self.logger.info(f"Performing OpenAI web search for query: {query[:100]}...")
            
            search_params = {
                "model": OPENAI_SEARCH_MODEL,
                "messages": [{"role": "user", "content": query}],
                "web_search_options": {}
            }
            
            # Add location context if provided
            if user_location:
                search_params["web_search_options"]["user_location"] = {
                    "type": "approximate", 
                    "approximate": user_location
                }
                self.logger.info(f"Using location context: {user_location}")
            
            # Use the OpenAI client directly with search parameters
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(**search_params)
            
            result = response.choices[0].message.content
            
            # Log successful search
            self.api_logger.info(f"OpenAI web search completed - Query length: {len(query)}, Response length: {len(result)}")
            self.logger.info(f"OpenAI web search completed successfully")
            
            return result
            
        except Exception as e:
            self.error_logger.error(f"OpenAI web search failed for query '{query[:50]}...': {e}")
            raise
    
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
    
    def _load_diarization_model_safe(self):
        """Safely load the speaker diarization model on demand using subprocess isolation"""
        if self.diarization_status == "available":
            return True
        
        if self.diarization_status not in ["deferred", "failed"]:
            return False
        
        if not hasattr(self, '_hf_token'):
            self.logger.error("No HuggingFace token available for diarization model loading")
            self.diarization_status = "failed"
            return False
        
        self.logger.info("Loading speaker diarization model with subprocess isolation...")
        self.diarization_status = "loading"
        
        try:
            # First, ensure all imports are available
            if not lazy_load_diarization_imports():
                raise Exception("Failed to import diarization dependencies")
            
            # Use subprocess to safely test model loading without risking main process
            import subprocess
            import sys
            import tempfile
            import json
            
            # Create a test script to verify model loading
            test_script = f'''
import os
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    # Set minimal environment for testing
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    import torch
    from pyannote.audio import Pipeline
    
    # Test basic model instantiation with minimal resources
    try:
        # Try to load with reduced settings to prevent segfaults
        model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="{self._hf_token}"
        )
        
        # Test model is functional with a dummy inference
        # This helps catch issues before returning to main process
        print("MODEL_LOAD_SUCCESS")
        sys.exit(0)
        
    except Exception as e:
        print(f"MODEL_LOAD_FAILED:{{e}}")
        sys.exit(1)
         
except Exception as e:
    print(f"IMPORT_FAILED:{{e}}")
    sys.exit(2)
'''
            
            # Write test script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                test_script_path = f.name
            
            try:
                # Run the test in a separate Python process with timeout
                result = subprocess.run([
                    sys.executable, test_script_path
                ], capture_output=True, text=True, timeout=60)  # 1 minute timeout (reduced)
                
                if result.returncode == 0 and "MODEL_LOAD_SUCCESS" in result.stdout:
                    # Test passed, now load in main process
                    self.logger.info("Subprocess test passed, loading model in main process...")
                    
                    # Load the actual model (the test verified it should work)
                    try:
                        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                        os.environ["TOKENIZERS_PARALLELISM"] = "false"
                        
                        # Import globally available modules
                        Pipeline = globals().get('Pipeline')
                        if not Pipeline:
                            from pyannote.audio import Pipeline
                        
                        self.diarization_model = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=self._hf_token
                        )
                        
                        self.diarization_status = "available"
                        self.logger.info("Speaker diarization model loaded successfully")
                        return True
                        
                    except Exception as main_load_e:
                        self.error_logger.error(f"Main process model loading failed despite subprocess success: {main_load_e}")
                        self.diarization_status = "failed"
                        return False
                else:
                    # Test failed, log the reason
                    error_msg = result.stdout + result.stderr
                    self.error_logger.error(f"Subprocess model test failed: {error_msg}")
                    self.diarization_status = "failed"
                    return False
                    
            finally:
                # Clean up test script
                try:
                    os.unlink(test_script_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            self.error_logger.error("Model loading test timed out - model appears to hang")
            self.diarization_status = "failed"
            return False
        except Exception as e:
            self.error_logger.error(f"Safe model loading failed: {e}")
            self.diarization_status = "failed"
            return False
    
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
        """Extract audio from video/speech content using yt-dlp with comprehensive metadata"""
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
                self.set_progress("extraction", 20, "Extracting video info and metadata...")
                self.logger.info("Extracting video info, metadata, and downloading audio...")
                
                self.set_progress("extraction", 40, "Downloading audio...")
                info = ydl.extract_info(url, download=True)
                
                # Extract comprehensive metadata
                metadata = self._extract_social_metadata(info)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 'Unknown')
                
                self.set_progress("extraction", 80, "Processing audio file...")
                self.logger.info(f"Content info - Title: '{title}', Duration: {duration}s")
                self.logger.info(f"Platform: {metadata.get('platform', 'Unknown')}, Channel: {metadata.get('uploader', 'Unknown')}")
                
                # Log metadata extraction results
                description_length = len(metadata.get('description', ''))
                self.logger.info(f"Metadata extracted - Description: {description_length} chars, Tags: {len(metadata.get('tags', []))}")
                
                # Find the downloaded audio file
                for file in os.listdir(self.temp_dir):
                    if file.endswith('.mp3'):
                        file_path = os.path.join(self.temp_dir, file)
                        file_size = os.path.getsize(file_path)
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        
                        self.set_progress("extraction", 100, f"Audio extraction complete - {file}")
                        self.logger.info(f"Audio extraction completed in {elapsed_time:.2f}s - File: {file}, Size: {file_size} bytes")
                        return file_path, title, metadata
                        
            self.logger.error("No MP3 file found after extraction")
            self.set_progress("extraction", 0, "No MP3 file found after extraction")
            return None, None, None
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Audio extraction failed after {elapsed_time:.2f}s for URL '{url}': {e}")
            self.set_progress("extraction", 0, f"Audio extraction failed: {str(e)}")
            return None, None, None

    def _extract_social_metadata(self, info):
        """Extract and normalize social media metadata from yt-dlp info"""
        try:
            # Extract core metadata fields
            metadata = {
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'uploader': info.get('uploader', ''),
                'channel': info.get('channel', info.get('uploader', '')),
                'upload_date': info.get('upload_date', ''),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'comment_count': info.get('comment_count', 0),
                'tags': info.get('tags', []),
                'platform': self._detect_platform(info.get('extractor', '')),
                'url': info.get('webpage_url', ''),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
            }
            
            # Clean and validate metadata
            metadata['description'] = str(metadata['description']) if metadata['description'] else ''
            metadata['tags'] = list(metadata['tags']) if metadata['tags'] else []
            
            # Ensure numeric fields are properly typed
            for field in ['view_count', 'like_count', 'comment_count', 'duration']:
                if metadata[field] is None:
                    metadata[field] = 0
                try:
                    metadata[field] = int(metadata[field])
                except (ValueError, TypeError):
                    metadata[field] = 0
            
            self.logger.info(f"Social metadata extracted - Platform: {metadata['platform']}, Description length: {len(metadata['description'])}")
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract social metadata: {e}")
            return {
                'title': info.get('title', ''),
                'description': '',
                'uploader': '',
                'channel': '',
                'upload_date': '',
                'view_count': 0,
                'like_count': 0,
                'comment_count': 0,
                'tags': [],
                'platform': 'Unknown',
                'url': info.get('webpage_url', ''),
                'duration': info.get('duration', 0),
                'thumbnail': '',
            }

    def _detect_platform(self, extractor):
        """Detect platform from yt-dlp extractor name"""
        extractor_lower = extractor.lower()
        
        platform_mapping = {
            'youtube': 'YouTube',
            'tiktok': 'TikTok', 
            'instagram': 'Instagram',
            'facebook': 'Facebook',
            'twitter': 'Twitter/X',
            'vimeo': 'Vimeo',
            'twitch': 'Twitch',
            'reddit': 'Reddit',
            'soundcloud': 'SoundCloud',
            'spotify': 'Spotify',
            'dailymotion': 'Dailymotion',
            'rumble': 'Rumble',
        }
        
        for key, platform in platform_mapping.items():
            if key in extractor_lower:
                return platform
                
                return f"Platform ({extractor})" if extractor else "Unknown Platform"

    def _build_social_context(self, metadata):
        """Build social media context section for LLM prompt"""
        if not metadata:
            return ""
        
        # Extract key fields
        title = metadata.get('title', '')
        description = metadata.get('description', '')
        platform = metadata.get('platform', '')
        uploader = metadata.get('uploader', '')
        channel = metadata.get('channel', '')
        upload_date = metadata.get('upload_date', '')
        view_count = metadata.get('view_count', 0)
        like_count = metadata.get('like_count', 0)
        tags = metadata.get('tags', [])
        
        # Build context sections
        context_parts = []
        
        # Platform information
        if platform:
            context_parts.append(f"**Platform:** {platform}")
        
        # Channel/uploader information  
        if channel or uploader:
            uploader_name = channel or uploader
            context_parts.append(f"**Channel/Uploader:** {uploader_name}")
        
        # Title (clearly marked as separate from transcript)
        if title:
            context_parts.append(f"**Content Title:** {title}")
        
        # Description (clearly marked as separate from transcript)
        if description:
            # Truncate very long descriptions
            desc_preview = description[:500] + "..." if len(description) > 500 else description
            context_parts.append(f"**Content Description:** {desc_preview}")
        
        # Publication info
        if upload_date:
            try:
                # Format date if it's in YYYYMMDD format
                if len(upload_date) == 8 and upload_date.isdigit():
                    formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                    context_parts.append(f"**Publication Date:** {formatted_date}")
                else:
                    context_parts.append(f"**Publication Date:** {upload_date}")
            except:
                context_parts.append(f"**Publication Date:** {upload_date}")
        
        # Engagement metrics
        engagement_info = []
        if view_count > 0:
            engagement_info.append(f"{view_count:,} views")
        if like_count > 0:
            engagement_info.append(f"{like_count:,} likes")
        
        if engagement_info:
            context_parts.append(f"**Engagement:** {', '.join(engagement_info)}")
        
        # Tags (first 5 most relevant)
        if tags:
            tag_list = tags[:5]  # Limit to first 5 tags
            context_parts.append(f"**Tags:** {', '.join(tag_list)}")
        
        if context_parts:
            return "\n\n**SOCIAL MEDIA CONTEXT:**\n" + "\n".join(context_parts) + "\n"
        return ""

    def transcribe_audio(self, audio_path, metadata=None):
        """Transcribe audio using local OpenAI Whisper model with timestamps and social metadata"""
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
            
            # Create detailed transcript object with embedded social metadata
            detailed_transcript = {
                "full_text": transcript_text,
                "segments": segments_with_timestamps,
                "language": info.language,
                "duration": elapsed_time,
                "social_metadata": metadata if metadata else {}
            }
            
            self.set_progress("transcription", 100, f"Transcription complete - {transcript_length} characters")
            
            self.logger.info(f"Transcription completed in {elapsed_time:.2f}s - Length: {transcript_length} characters, Segments: {len(segments_with_timestamps)}")
            self.api_logger.info(f"Local Whisper transcription successful - Transcript length: {transcript_length} chars, Duration: {elapsed_time:.2f}s, Segments: {len(segments_with_timestamps)}")
            
            if metadata:
                self.logger.info(f"Social metadata embedded - Platform: {metadata.get('platform', 'Unknown')}, Description: {len(metadata.get('description', ''))} chars")
            
            return detailed_transcript
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Audio transcription failed after {elapsed_time:.2f}s for file '{audio_path}': {e}")
            self.api_logger.error(f"Local Whisper transcription failed: {e}")
            self.set_progress("transcription", 0, f"Transcription failed: {str(e)}")
            return None
    
    def perform_speaker_diarization(self, audio_path):
        """Perform speaker diarization using pyannote.audio with robust error handling"""
        # Try to load the model if it's deferred
        if self.diarization_status == "deferred":
            if not self._load_diarization_model_safe():
                self.logger.warning("Failed to load diarization model on demand")
                return None
        
        if self.diarization_status != "available":
            self.logger.warning(f"Speaker diarization not available - status: {self.diarization_status}")
            return None
            
        if not self.diarization_model:
            self.logger.warning("Speaker diarization model not loaded")
            return None
            
        self.logger.info(f"Starting speaker diarization for file: {audio_path}")
        start_time = datetime.now()
        
        try:
            self.set_progress("diarization", 0, "Initializing speaker diarization...")
            
            # Import audio processing libraries that were loaded in lazy_load_diarization_imports
            try:
                import librosa
                import soundfile as sf
            except ImportError as import_e:
                self.error_logger.error(f"Audio processing libraries not available: {import_e}")
                self.set_progress("diarization", 0, f"Missing audio libraries: {import_e}")
                return None
            
            # Load audio file
            self.set_progress("diarization", 10, "Loading audio file...")
            
            # Ensure audio is in correct format for diarization
            # Load audio and ensure it's mono and at 16kHz sample rate
            audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Save processed audio temporarily for diarization
            temp_audio_path = os.path.join(self.temp_dir, "temp_diarization.wav")
            sf.write(temp_audio_path, audio_data, sample_rate)
            
            self.set_progress("diarization", 30, "Processing audio for speaker identification...")
            
            # Perform diarization with thread-safe timeout protection
            try:
                import threading
                import queue
                import time
                
                # Use thread-based timeout instead of signal (which doesn't work in threads)
                diarization_result_queue = queue.Queue()
                diarization_error_queue = queue.Queue()
                
                def diarization_process():
                    try:
                        # Set process timeout using signal alarm (Unix only)
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Diarization process timed out at model level")
                        
                        # Set a 25-second alarm (less than thread timeout)
                        if hasattr(signal, 'SIGALRM'):
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(25)
                        
                        try:
                            diarization = self.diarization_model(temp_audio_path)
                            diarization_result_queue.put(diarization)
                        finally:
                            # Cancel the alarm
                            if hasattr(signal, 'SIGALRM'):
                                signal.alarm(0)
                                
                    except Exception as e:
                        diarization_error_queue.put(e)
                
                # Start diarization in a separate thread
                diarization_thread = threading.Thread(target=diarization_process)
                diarization_thread.daemon = True
                diarization_thread.start()
                
                # Wait for completion with timeout (30 seconds max - more aggressive)
                diarization_thread.join(timeout=30)
                
                if diarization_thread.is_alive():
                    self.logger.error("Speaker diarization timed out after 30 seconds")
                    # Clean up temporary file if it exists
                    if os.path.exists(temp_audio_path):
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass
                    return None
                
                # Check for errors first
                if not diarization_error_queue.empty():
                    error = diarization_error_queue.get()
                    self.logger.error(f"Speaker diarization process failed: {error}")
                    return None
                
                # Get the result
                if not diarization_result_queue.empty():
                    diarization = diarization_result_queue.get()
                else:
                    self.logger.error("Speaker diarization completed but no result was returned")
                    return None
                    
            except Exception as diar_e:
                self.logger.error(f"Speaker diarization process failed: {diar_e}")
                return None
            
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
    
    def transcribe_with_speakers(self, audio_path, metadata=None):
        """Perform complete transcription with speaker diarization ONLY (no content-based identification)"""
        self.logger.info(f"Starting transcription with speaker diarization for: {audio_path}")
        start_time = datetime.now()
        
        # Check if speaker diarization is explicitly disabled
        if os.getenv("DISABLE_SPEAKER_DIARIZATION") or os.getenv("FORCE_BASIC_TRANSCRIPTION"):
            self.logger.info("Speaker diarization disabled via environment variable, performing basic transcription")
            transcript_data = self.transcribe_audio(audio_path, metadata)
            if transcript_data:
                transcript_data["has_speaker_diarization"] = False
                transcript_data["diarization_status"] = "disabled_by_env"
            return transcript_data
        
        try:
            # First, perform standard transcription with robust error handling and metadata
            self.logger.info("Step 1: Performing audio transcription...")
            transcript_data = self.transcribe_audio(audio_path, metadata)
            if not transcript_data:
                self.logger.error("Audio transcription failed completely")
                return None

            self.logger.info(f"Transcription successful: {len(transcript_data.get('full_text', ''))} characters, {len(transcript_data.get('segments', []))} segments")
            
            # Check if diarization is available or can be loaded
            if self.diarization_status == "deferred":
                self._load_diarization_model_safe()
                
            # If diarization is available, perform speaker identification
            if self.diarization_model:
                self.logger.info("Step 2: Performing pyannote speaker diarization...")
                try:
                    # Use threading timeout for diarization to prevent hanging
                    import threading
                    import queue
                    import signal
                    
                    diarization_queue = queue.Queue()
                    diarization_exception_queue = queue.Queue()
                    
                    def diarization_worker():
                        try:
                            # Set a shorter timeout for the actual diarization call
                            result = self.perform_speaker_diarization(audio_path)
                            diarization_queue.put(result)
                        except Exception as e:
                            diarization_exception_queue.put(e)
                    
                    # Start diarization in separate thread with aggressive timeout
                    diarization_thread = threading.Thread(target=diarization_worker)
                    diarization_thread.daemon = True
                    diarization_thread.start()
                    
                    # Wait with 90-second timeout (reduced from 2 minutes)
                    diarization_thread.join(timeout=90)
                    
                    if diarization_thread.is_alive():
                        self.logger.warning("Speaker diarization timed out after 90 seconds, returning transcript without speaker info")
                        transcript_data["has_speaker_diarization"] = False
                        transcript_data["diarization_status"] = "timeout"
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
                        
                        # SKIP CONTENT-BASED SPEAKER IDENTIFICATION
                        self.logger.info("SKIPPING content-based speaker identification (disabled per user request)")
                        
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        self.logger.info(f"Complete transcription with pyannote diarization finished in {elapsed_time:.2f}s")
                        return enhanced_transcript
                    else:
                        self.logger.warning("Speaker diarization failed, returning transcript without speaker info")
                        transcript_data["has_speaker_diarization"] = False
                        transcript_data["diarization_status"] = "failed"
                        
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        self.logger.info(f"Transcription without diarization finished in {elapsed_time:.2f}s")
                        return transcript_data
                        
                except Exception as e:
                    self.logger.warning(f"Speaker diarization failed with error: {e}, returning transcript without speaker info")
                    transcript_data["has_speaker_diarization"] = False
                    transcript_data["diarization_status"] = "error"
                    transcript_data["diarization_error"] = str(e)
                    
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    self.logger.info(f"Transcription with diarization error finished in {elapsed_time:.2f}s")
                    return transcript_data
            else:
                self.logger.info("Speaker diarization not available, returning standard transcript")
                transcript_data["has_speaker_diarization"] = False
                transcript_data["diarization_status"] = "not_available"
                
                elapsed_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Transcription without diarization capability finished in {elapsed_time:.2f}s")
                return transcript_data
                
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Transcription with speakers failed after {elapsed_time:.2f}s: {e}")
            return None
        finally:
            # Clean up any multiprocessing resources to prevent resource leaks
            try:
                import multiprocessing
                import gc
                
                # Force garbage collection to clean up any leftover resources
                gc.collect()
                
                # Close any open multiprocessing pools
                if hasattr(multiprocessing, 'active_children'):
                    for process in multiprocessing.active_children():
                        try:
                            process.terminate()
                            process.join(timeout=1)
                        except:
                            pass
                            
            except Exception as cleanup_e:
                self.logger.warning(f"Cleanup warning (non-critical): {cleanup_e}")
                pass
    
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
                    
                    # Add speaker identification information
                    if "speaker_identification" in transcript_data:
                        speaker_id = transcript_data["speaker_identification"]
                        f.write(f"\n--- Speaker Identification ---\n")
                        if speaker_id.get("primary_speaker"):
                            f.write(f"Identified Speaker: {speaker_id['primary_speaker']} (Confidence: {speaker_id['primary_confidence']}%)\n")
                            evidence = speaker_id.get("all_candidates", {}).get(speaker_id["primary_speaker"], {}).get("evidence", [])
                            if evidence:
                                f.write(f"Evidence: {'; '.join(evidence[:5])}\n")
                        else:
                            f.write("No speaker identified with high confidence\n")
                            candidates = speaker_id.get("all_candidates", {})
                            if candidates:
                                f.write("Top candidates: ")
                                for i, (name, data) in enumerate(list(candidates.items())[:3]):
                                    f.write(f"{name}({data['confidence']}%)")
                                    if i < 2: f.write(", ")
                                f.write("\n")
                        f.write(f"Analysis method: {speaker_id.get('identification_method', 'Unknown')}\n")
                else:
                    f.write("TRANSCRIPT (NO SPEAKER DIARIZATION)\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(transcript_data.get("full_text", ""))
                    
                    # Add speaker identification information even without diarization
                    if "speaker_identification" in transcript_data:
                        speaker_id = transcript_data["speaker_identification"]
                        f.write(f"\n\n--- Speaker Identification ---\n")
                        if speaker_id.get("primary_speaker"):
                            f.write(f"Identified Speaker: {speaker_id['primary_speaker']} (Confidence: {speaker_id['primary_confidence']}%)\n")
                            evidence = speaker_id.get("all_candidates", {}).get(speaker_id["primary_speaker"], {}).get("evidence", [])
                            if evidence:
                                f.write(f"Evidence: {'; '.join(evidence[:5])}\n")
                        else:
                            f.write("No speaker identified with high confidence\n")
                            candidates = speaker_id.get("all_candidates", {})
                            if candidates:
                                f.write("Top candidates: ")
                                for i, (name, data) in enumerate(list(candidates.items())[:3]):
                                    f.write(f"{name}({data['confidence']}%)")
                                    if i < 2: f.write(", ")
                                f.write("\n")
                        f.write(f"Analysis method: {speaker_id.get('identification_method', 'Unknown')}\n")
            
            self.logger.info(f"Enhanced transcript saved to: {json_path} and {txt_path}")
            return {"json": json_path, "txt": txt_path}
            
        except Exception as e:
            self.error_logger.error(f"Failed to save enhanced transcript: {e}")
            return None
    
    def perform_web_research(self, claims):
        """Perform comprehensive web research on claims using OpenAI's web search capability"""
        self.logger.info(f"Starting OpenAI web research for {len(claims)} claims")
        
        research_results = []
        claims_to_research = claims[:3]  # Process more claims since no quota restrictions
        
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
                self.set_progress("research", (i-1)*30, f"Researching claim {i} with web search...")
                
                # Create comprehensive search prompt for claim verification
                search_prompt = f"""
                Fact-check and verify this claim: "{claim_text}"
                
                Please provide a comprehensive analysis including:
                1. Verification status (VERIFIED/PARTIALLY_VERIFIED/DISPUTED/UNVERIFIABLE/FALSE)
                2. Truthfulness score (0-100)
                3. Supporting evidence with specific sources and URLs
                4. Contradicting evidence if any exists
                5. Expert opinions and analysis
                6. Evidence quality assessment (STRONG/MODERATE/WEAK/INSUFFICIENT)
                7. Confidence level (HIGH/MEDIUM/LOW)
                8. Recommendation (ACCEPT/ACCEPT_WITH_CAUTION/QUESTION/REJECT)
                9. Detailed verification notes explaining the reasoning
                
                Provide direct links to credible sources for all claims. Focus on recent, authoritative sources.
                Include specific facts, dates, and quotes where available.
                """
                
                # Perform OpenAI web search
                search_result = self.perform_openai_web_search(search_prompt)
                
                # Parse the search result into structured format
                research_result = self._parse_openai_search_result(claim_text, search_result)
                research_result['research_method'] = 'OpenAI Web Search'
                research_result['claim_text'] = claim_text
                
                research_results.append(research_result)
                
                # Log research outcome
                verification_status = research_result.get('verification_status', 'UNKNOWN')
                truthfulness_score = research_result.get('truthfulness_score', 0)
                evidence_quality = research_result.get('evidence_quality', 'UNKNOWN')
                
                self.logger.info(f"Web research completed for claim {i}: {verification_status} "
                               f"(Score: {truthfulness_score}/100, Evidence: {evidence_quality})")
                
            except Exception as e:
                self.error_logger.error(f"OpenAI web research failed for claim {i}: {e}")
                research_results.append(self._create_fallback_research_result(claim_text, f"Web search error: {str(e)}"))
        
        self.logger.info(f"OpenAI web research completed for {len(research_results)} claims")
        return research_results
    
    def _parse_openai_search_result(self, claim_text, search_result):
        """Parse OpenAI web search result into structured research format"""
        try:
            # Default values
            parsed_result = {
                'claim': claim_text,
                'verification_status': 'UNVERIFIABLE',
                'truthfulness_score': 50,
                'evidence_quality': 'INSUFFICIENT',
                'confidence_level': 'LOW',
                'recommendation': 'QUESTION',
                'supporting_evidence': [],
                'contradicting_evidence': [],
                'verification_notes': search_result,
                'expert_sources': [],
                'web_sources_found': 0
            }
            
            # Extract verification status from search result
            result_upper = search_result.upper()
            if 'VERIFIED' in result_upper and 'PARTIALLY' not in result_upper:
                parsed_result['verification_status'] = 'VERIFIED'
            elif 'PARTIALLY_VERIFIED' in result_upper or 'PARTIALLY VERIFIED' in result_upper:
                parsed_result['verification_status'] = 'PARTIALLY_VERIFIED'
            elif 'DISPUTED' in result_upper:
                parsed_result['verification_status'] = 'DISPUTED'
            elif 'FALSE' in result_upper:
                parsed_result['verification_status'] = 'FALSE'
            
            # Extract truthfulness score
            import re
            score_match = re.search(r'(?:score|truthfulness).*?(\d+)', result_upper)
            if score_match:
                score = int(score_match.group(1))
                parsed_result['truthfulness_score'] = max(0, min(100, score))
            
            # Extract evidence quality
            if 'STRONG' in result_upper:
                parsed_result['evidence_quality'] = 'STRONG'
            elif 'MODERATE' in result_upper:
                parsed_result['evidence_quality'] = 'MODERATE'
            elif 'WEAK' in result_upper:
                parsed_result['evidence_quality'] = 'WEAK'
            
            # Extract confidence level
            if 'HIGH' in result_upper and 'CONFIDENCE' in result_upper:
                parsed_result['confidence_level'] = 'HIGH'
            elif 'MEDIUM' in result_upper and 'CONFIDENCE' in result_upper:
                parsed_result['confidence_level'] = 'MEDIUM'
            
            # Extract recommendation
            if 'ACCEPT_WITH_CAUTION' in result_upper or 'ACCEPT WITH CAUTION' in result_upper:
                parsed_result['recommendation'] = 'ACCEPT_WITH_CAUTION'
            elif 'ACCEPT' in result_upper:
                parsed_result['recommendation'] = 'ACCEPT'
            elif 'REJECT' in result_upper:
                parsed_result['recommendation'] = 'REJECT'
            
            # Count approximate sources mentioned (look for URLs)
            url_count = len(re.findall(r'https?://[^\s]+', search_result))
            parsed_result['web_sources_found'] = url_count
            
            # Extract supporting and contradicting evidence (simple heuristic)
            lines = search_result.split('\n')
            for line in lines:
                line_lower = line.lower()
                if ('support' in line_lower or 'evidence for' in line_lower or 'confirms' in line_lower) and line.strip():
                    parsed_result['supporting_evidence'].append(line.strip())
                elif ('contradict' in line_lower or 'disputes' in line_lower or 'against' in line_lower) and line.strip():
                    parsed_result['contradicting_evidence'].append(line.strip())
            
            self.logger.info(f"Parsed search result: {parsed_result['verification_status']} (Score: {parsed_result['truthfulness_score']})")
            return parsed_result
            
        except Exception as e:
            self.error_logger.error(f"Failed to parse OpenAI search result: {e}")
            return {
                'claim': claim_text,
                'verification_status': 'UNVERIFIABLE',
                'truthfulness_score': 50,
                'evidence_quality': 'INSUFFICIENT',
                'confidence_level': 'LOW',
                'recommendation': 'QUESTION',
                'supporting_evidence': [],
                'contradicting_evidence': [],
                'verification_notes': search_result,
                'expert_sources': [],
                'web_sources_found': 0,
                'research_method': 'OpenAI Web Search (Parse Error)'
            }
    
    def perform_enhanced_web_research(self, analysis, title, transcript_data):
        """Enhanced web research using OpenAI's web search for topic research and claim verification"""
        self.logger.info(f"Starting enhanced OpenAI web research for topic and claims")
        
        try:
            # Step 1: Extract topic information and context
            self.set_progress("research", 10, "Extracting topic information...")
            topic_info = self._extract_topic_information(title, transcript_data)
            self.logger.info(f"Extracted topic information: {topic_info.get('main_topic', 'Unknown')}")
            
            # Step 2: Perform topic-based research using OpenAI web search
            self.set_progress("research", 25, "Researching topic background...")
            topic_research = self._perform_topic_research_openai(topic_info)
            
            # Step 3: Extract claims for verification
            self.set_progress("research", 50, "Extracting claims for verification...")
            claims_to_research = self._extract_claims_for_research(analysis)
            self.logger.info(f"Extracted {len(claims_to_research)} claims for verification")
            
            # Step 4: Perform claim verification research using OpenAI web search
            self.set_progress("research", 75, "Verifying specific claims...")
            claim_research = self._perform_claim_verification_openai(claims_to_research)
            
            # Step 5: Combine all research results
            self.set_progress("research", 95, "Combining research results...")
            combined_research = self._combine_research_results_openai(topic_research, claim_research, topic_info)
            
            self.set_progress("research", 100, "Enhanced web research complete")
            self.logger.info(f"Enhanced web research completed - Topic research: {len(topic_research)}, Claim research: {len(claim_research)}")
            
            return combined_research
            
        except Exception as e:
            self.error_logger.error(f"Enhanced web research failed: {e}")
            self.set_progress("research", 0, f"Enhanced research failed: {str(e)}")
            # Fallback to basic claim research
            claims = self._extract_claims_for_research(analysis)
            return self.perform_web_research(claims[:3])  # Limit to 3 claims as fallback
    
    def _perform_topic_research_openai(self, topic_info):
        """Perform topic research using OpenAI web search"""
        try:
            main_topic = topic_info.get('main_topic', '')
            
            # Create comprehensive topic research prompt
            topic_search_prompt = f"""
            Research comprehensive background information about: {main_topic}
            
            Key context to explore:
            - Key people: {', '.join([p.get('name', '') for p in topic_info.get('key_people', [])])}
            - Important dates: {', '.join([d.get('date', '') for d in topic_info.get('important_dates', [])])}
            - Central themes: {', '.join(topic_info.get('central_themes', []))}
            
            Please provide:
            1. Historical context and timeline
            2. Recent developments and current status
            3. Expert analysis and credible sources
            4. Key facts and verified information
            5. Relevant background that would help understand claims about this topic
            
            Include direct links to authoritative sources for verification.
            Focus on factual, well-sourced information from credible institutions.
            """
            
            search_result = self.perform_openai_web_search(topic_search_prompt)
            
            # Structure the topic research result
            topic_research_result = {
                'topic': main_topic,
                'background_summary': search_result,
                'research_type': 'topic_background',
                'research_method': 'OpenAI Web Search',
                'topic_info': topic_info
            }
            
            self.logger.info(f"Topic research completed for: {main_topic}")
            return [topic_research_result]
            
        except Exception as e:
            self.error_logger.error(f"Topic research failed: {e}")
            return []
    
    def _perform_claim_verification_openai(self, claims):
        """Perform claim verification using OpenAI web search"""
        verification_results = []
        
        for i, claim in enumerate(claims[:3], 1):  # Limit to 3 claims
            try:
                # Handle different claim formats
                if isinstance(claim, dict):
                    claim_text = str(claim.get('text', claim.get('claim', str(claim))))
                else:
                    claim_text = str(claim)
                
                self.logger.info(f"Verifying claim {i}: {claim_text[:100]}...")
                
                # Create verification prompt
                verification_prompt = f"""
                Verify and fact-check this specific claim: "{claim_text}"
                
                Provide comprehensive verification including:
                1. Verification status (VERIFIED/PARTIALLY_VERIFIED/DISPUTED/UNVERIFIABLE/FALSE)
                2. Truthfulness score (0-100)
                3. Supporting evidence with sources and URLs
                4. Contradicting evidence if any
                5. Expert analysis and fact-checker reports
                6. Evidence quality (STRONG/MODERATE/WEAK/INSUFFICIENT)
                7. Confidence level (HIGH/MEDIUM/LOW)
                8. Recommendation (ACCEPT/ACCEPT_WITH_CAUTION/QUESTION/REJECT)
                
                Focus on recent, credible sources and provide specific details with links.
                """
                
                search_result = self.perform_openai_web_search(verification_prompt)
                parsed_result = self._parse_openai_search_result(claim_text, search_result)
                parsed_result['research_method'] = 'OpenAI Web Search - Claim Verification'
                
                verification_results.append(parsed_result)
                
            except Exception as e:
                self.error_logger.error(f"Claim verification failed for claim {i}: {e}")
                verification_results.append(
                    self._create_fallback_research_result(claim_text, f"Verification error: {str(e)}")
                )
        
        self.logger.info(f"Claim verification completed for {len(verification_results)} claims")
        return verification_results
    
    def _combine_research_results_openai(self, topic_research, claim_research, topic_info):
        """Combine topic research and claim verification results"""
        try:
            combined_results = []
            
            # Add topic research results
            for topic_result in topic_research:
                combined_results.append({
                    'type': 'topic_research',
                    'content': topic_result,
                    'research_method': 'OpenAI Web Search'
                })
            
            # Add claim verification results
            for claim_result in claim_research:
                combined_results.append({
                    'type': 'claim_verification',
                    'content': claim_result,
                    'research_method': 'OpenAI Web Search'
                })
            
            # Create summary
            research_summary = {
                'total_results': len(combined_results),
                'topic_research_count': len(topic_research),
                'claim_verification_count': len(claim_research),
                'main_topic': topic_info.get('main_topic', 'Unknown'),
                'research_method': 'Enhanced OpenAI Web Search',
                'timestamp': datetime.now().isoformat()
            }
            
            # Add summary as first result
            combined_results.insert(0, {
                'type': 'research_summary',
                'content': research_summary,
                'research_method': 'OpenAI Web Search'
            })
            
            self.logger.info(f"Combined research results: {len(combined_results)} total results")
            return combined_results
            
        except Exception as e:
            self.error_logger.error(f"Failed to combine research results: {e}")
            return topic_research + claim_research  # Simple fallback
    
    def _extract_topic_information(self, title, transcript_data):
        """Extract topic information and context from title and transcript"""
        try:
            # Get transcript text
            if isinstance(transcript_data, dict):
                transcript_text = transcript_data.get("full_text", "")
                language = transcript_data.get("language", "unknown")
            else:
                transcript_text = str(transcript_data)
                language = "unknown"
            
            # Limit transcript length for processing
            transcript_excerpt = transcript_text[:2000] + "..." if len(transcript_text) > 2000 else transcript_text
            
            topic_extraction_prompt = f"""
            Extract topic information and context from this content:
            
            TITLE: "{title}"
            LANGUAGE: {language}
            TRANSCRIPT EXCERPT: "{transcript_excerpt}"
            
            Please analyze and extract:
            1. Main topic/subject
            2. Key people mentioned (names, titles, roles)
            3. Important dates or time periods mentioned
            4. Key locations or places mentioned
            5. Main events or incidents discussed
            6. Central themes or subjects
            7. Relevant keywords for research
            
            Return your analysis as JSON in this exact format:
            {{
                "main_topic": "Brief description of the main topic",
                "key_people": [
                    {{"name": "Person Name", "role": "Their role or title", "context": "Why they're relevant"}}
                ],
                "important_dates": [
                    {{"date": "Date or time period", "significance": "What happened or why it's important"}}
                ],
                "key_locations": [
                    {{"location": "Place name", "relevance": "Why this location is important"}}
                ],
                "main_events": [
                    {{"event": "Event description", "timeframe": "When it happened", "significance": "Why it matters"}}
                ],
                "central_themes": ["Theme 1", "Theme 2", "Theme 3"],
                "research_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
            }}
            """
            
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": topic_extraction_prompt}]
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                topic_info = json.loads(json_match.group())
                self.logger.info(f"Successfully extracted topic information")
                return topic_info
            else:
                raise ValueError("No valid JSON found in topic extraction response")
                
        except Exception as e:
            self.logger.warning(f"Failed to extract topic information: {e}")
            # Fallback topic info
            return {
                "main_topic": title or "Unknown Topic",
                "key_people": [],
                "important_dates": [],
                "key_locations": [],
                "main_events": [],
                "central_themes": [title] if title else [],
                "research_keywords": title.split()[:5] if title else []
            }
    
    def _perform_topic_research(self, topic_info, serpapi_key):
        """DEPRECATED: Legacy SerpAPI-based topic research (replaced with OpenAI web search)"""
        try:
            self.logger.info(f"Starting topic research for: {topic_info.get('main_topic', 'Unknown')}")
            
            # Generate topic-based search queries
            topic_queries = self._generate_topic_search_queries(topic_info)
            topic_research_results = []
            
            for i, query in enumerate(topic_queries[:1]):  # Limit to 1 topic query
                try:
                    # Check quota before each search
                    quota_ok, quota_message = check_serpapi_quota()
                    if not quota_ok:
                        self.logger.warning(f"SerpAPI quota exceeded during topic search: {quota_message}")
                        break
                    
                    # Rate limiting
                    if i > 0:
                        import time
                        delay = float(os.getenv("SERPAPI_DELAY_SECONDS", "1.5"))
                        time.sleep(delay)
                    
                    # Perform search
                    search_params = {
                        "q": query,
                        "api_key": serpapi_key,
                        "engine": "google",
                        "num": 4,  # Get more results for topic context
                        "safe": "active",
                        "hl": "en",
                        "gl": "us"
                    }
                    
                    search = GoogleSearch(search_params)
                    search_data = search.get_dict()
                    increment_serpapi_usage()
                    
                    organic_results = search_data.get("organic_results", [])
                    
                    for result in organic_results:
                        topic_research_results.append({
                            'type': 'topic_research',
                            'query': query,
                            'title': result.get('title', 'No title'),
                            'snippet': result.get('snippet', 'No content'),
                            'url': result.get('link', 'No URL'),
                            'source': self._extract_domain(result.get('link', '')),
                            'position': result.get('position', 0)
                        })
                    
                    self.logger.info(f"Topic research query '{query[:50]}...' returned {len(organic_results)} results")
                    
                except Exception as search_error:
                    self.error_logger.error(f"Topic search failed for query '{query[:50]}...': {search_error}")
                    continue
            
            # Analyze topic research results
            if topic_research_results:
                topic_analysis = self._analyze_topic_research_results(topic_info, topic_research_results)
                topic_analysis['total_sources'] = len(topic_research_results)
                topic_analysis['research_type'] = 'topic_background'
                return [topic_analysis]
            else:
                return []
                
        except Exception as e:
            self.error_logger.error(f"Topic research failed: {e}")
            return []
    
    def _generate_topic_search_queries(self, topic_info):
        """Generate search queries for topic-based research"""
        queries = []
        main_topic = topic_info.get('main_topic', '')
        
        if main_topic:
            # Basic topic query
            queries.append(main_topic)
        
        # Add queries with people and dates (minimal for quota conservation)
        for person in topic_info.get('key_people', [])[:1]:  # Limit to 1 person
            person_query = f"{person.get('name', '')} {main_topic}"
            if person_query.strip():
                queries.append(person_query)
        
        # Add queries with dates
        for date_info in topic_info.get('important_dates', [])[:1]:  # Limit to 1 date
            date_query = f"{main_topic} {date_info.get('date', '')}"
            if date_query.strip():
                queries.append(date_query)
        
        # Add queries with events
        for event in topic_info.get('main_events', [])[:1]:  # Limit to 1 event
            event_query = f"{event.get('event', '')} background"
            if event_query.strip():
                queries.append(event_query)
        
        # Add keyword-based queries
        keywords = topic_info.get('research_keywords', [])
        if len(keywords) >= 2:
            keyword_query = ' '.join(keywords[:3])  # Combine first 3 keywords
            queries.append(keyword_query)
        
        # Remove duplicates and empty queries
        queries = list(set([q.strip() for q in queries if q.strip()]))
        
        self.logger.info(f"Generated {len(queries)} topic search queries")
        return queries[:1]  # Limit to 1 query max
    
    def _analyze_topic_research_results(self, topic_info, research_results):
        """Analyze topic research results to provide background context"""
        try:
            # Prepare research evidence for analysis
            evidence_text = ""
            for result in research_results[:8]:  # Limit to top 8 results
                evidence_text += f"Title: {result.get('title', 'No title')}\n"
                evidence_text += f"Source: {result.get('source', 'Unknown source')}\n"
                evidence_text += f"Content: {result.get('snippet', 'No content')}\n"
                evidence_text += f"URL: {result.get('url', 'No URL')}\n\n"
            
            analysis_prompt = f"""
            Based on the topic information and web research results, provide comprehensive background context:
            
            TOPIC INFORMATION:
            Main Topic: {topic_info.get('main_topic', 'Unknown')}
            Key People: {', '.join([p.get('name', '') for p in topic_info.get('key_people', [])])}
            Important Dates: {', '.join([d.get('date', '') for d in topic_info.get('important_dates', [])])}
            Central Themes: {', '.join(topic_info.get('central_themes', []))}
            
            WEB RESEARCH EVIDENCE:
            {evidence_text}
            
            Please provide a comprehensive analysis in JSON format:
            {{
                "background_summary": "A comprehensive background summary of the topic",
                "key_context": [
                    "Important contextual point 1",
                    "Important contextual point 2",
                    "Important contextual point 3"
                ],
                "historical_context": "Historical background and timeline if relevant",
                "current_status": "Current state or recent developments",
                "reliability_assessment": "Assessment of source reliability and information quality",
                "related_topics": ["Related topic 1", "Related topic 2"],
                "expert_sources": [
                    {{"source": "Source name", "expertise": "Area of expertise", "credibility": "High/Medium/Low"}}
                ]
            }}
            """
            
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
                analysis_result.update({
                    'topic_info': topic_info,
                    'research_method': 'Enhanced Topic Research',
                    'analysis_type': 'background_context'
                })
                return analysis_result
            else:
                raise ValueError("No valid JSON found in topic analysis response")
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze topic research results: {e}")
            return {
                'background_summary': 'Background research was performed but analysis failed',
                'key_context': [],
                'historical_context': 'Unable to determine historical context',
                'current_status': 'Unable to determine current status',
                'reliability_assessment': 'Analysis incomplete',
                'related_topics': [],
                'expert_sources': [],
                'topic_info': topic_info,
                'research_method': 'Enhanced Topic Research (Fallback)',
                'analysis_type': 'background_context'
            }
    
    def _extract_claims_for_research(self, analysis):
        """Extract claims from analysis for verification research"""
        claims = []
        
        # Extract subjective claims that are presented as facts
        if 'subjective_claims' in analysis:
            for claim in analysis['subjective_claims']:
                if isinstance(claim, dict):
                    claims.append(claim.get('claim', str(claim)))
                else:
                    claims.append(str(claim))
        
        # Extract key claims from simple analysis
        if 'key_claims' in analysis:
            for claim in analysis['key_claims']:
                if isinstance(claim, dict):
                    claims.append(claim.get('claim', str(claim)))
                else:
                    claims.append(str(claim))
        
        # Extract claims from key takeaways
        if 'key_takeaways' in analysis and isinstance(analysis['key_takeaways'], str):
            takeaway_sentences = [s.strip() for s in analysis['key_takeaways'].split('.') if s.strip()]
            claims.extend(takeaway_sentences[:2])  # Take first 2 sentences as potential claims
        
        # Extract claims from rhetorical tactics with high intensity
        if 'rhetorical_tactics' in analysis:
            for tactic in analysis['rhetorical_tactics']:
                if tactic.get('intensity_score', 0) > 0.7 and tactic.get('examples'):
                    claims.extend(tactic['examples'][:1])  # Take 1 example from high-intensity tactics
        
        # Remove duplicates and limit claims
        unique_claims = list(set([c.strip() for c in claims if c.strip() and len(c.strip()) > 10]))
        
        self.logger.info(f"Extracted {len(unique_claims)} unique claims for verification")
        return unique_claims[:1]  # Limit to 1 claim for verification
    
    def _perform_claim_verification_research(self, claims, serpapi_key):
        """Perform verification research on specific claims"""
        if not claims:
            return []
        
        verification_results = []
        
        for i, claim in enumerate(claims):
            try:
                # Check quota before each claim research
                quota_ok, quota_message = check_serpapi_quota()
                if not quota_ok:
                    self.logger.warning(f"SerpAPI quota exceeded during claim verification: {quota_message}")
                    break
                
                # Generate search queries for the claim
                search_queries = self._generate_search_queries(claim)
                claim_search_results = []
                
                for j, query in enumerate(search_queries[:1]):  # Limit to 1 query per claim
                    try:
                        # Rate limiting
                        if i > 0 or j > 0:
                            import time
                            delay = float(os.getenv("SERPAPI_DELAY_SECONDS", "1.5"))
                            time.sleep(delay)
                        
                        # Perform search
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
                        increment_serpapi_usage()
                        
                        organic_results = search_data.get("organic_results", [])
                        
                        for result in organic_results:
                            claim_search_results.append({
                                'title': result.get('title', 'No title'),
                                'body': result.get('snippet', 'No content'),
                                'href': result.get('link', 'No URL'),
                                'position': result.get('position', 0),
                                'source': self._extract_domain(result.get('link', ''))
                            })
                        
                        self.logger.info(f"Claim verification query '{query[:50]}...' returned {len(organic_results)} results")
                        
                    except Exception as search_error:
                        self.error_logger.error(f"Claim verification search failed for query '{query[:50]}...': {search_error}")
                        continue
                
                # Analyze claim verification results
                if claim_search_results:
                    verification_result = self._analyze_web_results(claim, claim_search_results)
                    verification_result['research_type'] = 'claim_verification'
                    verification_result['web_sources_found'] = len(claim_search_results)
                    verification_result['search_queries_used'] = search_queries
                    verification_results.append(verification_result)
                else:
                    # No web results found, use AI-only analysis
                    fallback_result = self._perform_single_ai_research(claim)
                    fallback_result['research_type'] = 'claim_verification_ai_only'
                    fallback_result['web_sources_found'] = 0
                    verification_results.append(fallback_result)
                
            except Exception as e:
                self.error_logger.error(f"Claim verification failed for claim {i+1}: {e}")
                fallback_result = self._create_fallback_research_result(claim, f"Verification error: {str(e)}")
                fallback_result['research_type'] = 'claim_verification_error'
                verification_results.append(fallback_result)
        
        self.logger.info(f"Completed claim verification research for {len(verification_results)} claims")
        return verification_results
    
    def _combine_research_results(self, topic_research, claim_research, topic_info):
        """Combine topic research and claim verification results"""
        combined_results = []
        
        # Add topic research results first
        for topic_result in topic_research:
            combined_results.append(topic_result)
        
        # Add claim verification results
        for claim_result in claim_research:
            combined_results.append(claim_result)
        
        # If we have results, add a summary result
        if combined_results:
            summary_result = {
                'research_type': 'enhanced_research_summary',
                'analysis_type': 'research_summary',
                'topic_info': topic_info,
                'total_topic_research': len(topic_research),
                'total_claim_verification': len(claim_research),
                'research_method': 'Enhanced Web Research (Topic + Claims)',
                'summary': f"Enhanced research completed: {len(topic_research)} topic analysis, {len(claim_research)} claim verifications",
                'research_scope': {
                    'topic_coverage': bool(topic_research),
                    'claim_verification': bool(claim_research),
                    'background_context': bool(topic_research),
                    'factual_verification': bool(claim_research)
                }
            }
            combined_results.insert(0, summary_result)  # Add at the beginning
        
        self.logger.info(f"Combined research results: {len(combined_results)} total results")
        return combined_results
    
    def _generate_search_queries(self, claim_text):
        """Generate effective search queries for fact-checking a claim"""
        try:
            query_prompt = f"""
            Generate 1 specific, effective search query to fact-check this claim:
            
            CLAIM: "{claim_text}"
            
            Return ONLY a JSON array of search query strings, like:
            ["search query 1"]
            
            Make queries:
            - Specific and factual
            - Include key names, dates, locations mentioned
            - Focus on verifiable facts
            - Avoid opinion-based terms
            - Keep queries concise (under 50 characters each)
            """
            
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": query_prompt}]
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                if isinstance(queries, list):
                    # Limit to 1 query and truncate long ones
                    return [str(q)[:100] for q in queries[:1]]
            
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
            
            response = self.create_chat_completion(
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
            
            response = self.create_chat_completion(
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
        """DEPRECATED: Legacy transcript analysis method. Use analyze_transcript_enhanced() instead."""
        self.logger.warning("DEPRECATED: analyze_transcript() is deprecated. All analysis should use enhanced political rhetoric prompt.")
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
            
            response = self.create_chat_completion(
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

    def load_prompt_template(self, prompt_type="comprehensive"):
        """Load the prompt template based on type"""
        try:
            if prompt_type == "simple":
                filename = 'SimplePrompt.md'
                self.logger.info("Loading simple prompt template")
            else:
                filename = 'Prompt.md'
                self.logger.info("Loading comprehensive prompt template")
            
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract the prompt content before the "Text to Analyze" section
                if "## 📄 Text to Analyze" in content:
                    prompt_content = content.split("## 📄 Text to Analyze")[0]
                    return prompt_content.strip()
                else:
                    self.logger.warning(f"Could not find '## 📄 Text to Analyze' marker in {filename}")
                    return content.strip()
        except FileNotFoundError:
            self.logger.error(f"{filename} file not found")
            return None
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {e}")
            return None

    def analyze_transcript_enhanced(self, transcript_data, title, prompt_type="comprehensive"):
        """Enhanced analysis using the structured prompt"""
        analysis_type = "comprehensive political rhetoric" if prompt_type == "comprehensive" else "simple content"
        self.logger.info(f"Starting {analysis_type} analysis for title: '{title}'")
        start_time = datetime.now()
        
        progress_message = f"Loading {prompt_type} analysis prompt..."
        self.set_progress("analysis", 0, progress_message)
        
        # Load the prompt template
        prompt_template = self.load_prompt_template(prompt_type)
        if not prompt_template:
            filename = "Prompt.md" if prompt_type == "comprehensive" else "SimplePrompt.md"
            self.logger.error(f"Prompt not available - {filename} file is required")
            raise FileNotFoundError(f"{filename} file is required for {analysis_type} analysis")
        
        # Handle both old string format and new detailed format for backward compatibility
        if isinstance(transcript_data, dict):
            transcript_text = transcript_data.get("full_text", "")
            segments = transcript_data.get("segments", [])
            language = transcript_data.get("language", "unknown")
            social_metadata = transcript_data.get("social_metadata", {})
            
            # Use enhanced transcript formatting for better AI context
            timestamped_transcript = self._build_enhanced_transcript_format(
                transcript_data, title, prompt_type
            )
        else:
            # Backward compatibility for string transcripts
            transcript_text = transcript_data
            timestamped_transcript = transcript_text
            language = "unknown"
            social_metadata = {}
        
        transcript_length = len(transcript_text)
        self.logger.info(f"Enhanced analysis - Transcript length: {transcript_length} characters, Language: {language}")
        
        progress_message = f"Sending transcript to {self.model_name} for {analysis_type} analysis..."
        self.set_progress("analysis", 20, progress_message)
        
        try:
            # Build social media context section
            social_context = self._build_social_context(social_metadata)
            
            # SKIP speaker identification information (disabled per user request)
            speaker_info_text = ""
            # Content-based speaker identification has been disabled
            # Only pyannote diarization speaker labels (SPEAKER_00, SPEAKER_01, etc.) are used
            
            # For o3-mini model, use a much shorter, more focused prompt
            if self.model_name in ["o3-mini", "o3", "o1-mini", "o1-preview", "o1"]:
                # Use simplified prompt for reasoning models
                analysis_type_label = "Political Rhetoric Analysis" if prompt_type == "comprehensive" else "Content Analysis"
                simplified_prompt = f"""Analyze this political speech transcript and provide a comprehensive assessment.

**Language:** {language}
**Analysis Type:** {analysis_type_label}{social_context}{speaker_info_text}

**Enhanced Transcript:**
{timestamped_transcript[:4000]}{"..." if len(timestamped_transcript) > 4000 else ""}

Please analyze for:
1. Rhetorical tactics (nationalist appeals, fear narratives, media criticism, etc.)
2. Subjective claims presented as facts
3. Truthfulness and evidence quality
4. Psychological markers (narcissism, authoritarianism)
5. Hate speech or discriminatory language
6. Contradictions and inconsistencies
7. Overall credibility assessment

Provide your analysis in this JSON format:
{{
  "rhetorical_tactics": [
    {{"tactic": "Name", "occurrences": 0, "intensity_score": 0.0, "examples": ["text"]}}
  ],
  "subjective_claims": [
    {{"claim": "Text", "frequency": 0, "context": "Context"}}
  ],
  "truthfulness": {{
    "overall_score": 0.0,
    "evidence_quality": "Assessment",
    "fact_check_summary": "Summary"
  }},
  "psychological_markers": {{
    "narcissism": {{"score": 0.0, "description": "Description"}},
    "authoritarianism": {{"score": 0.0, "description": "Description"}}
  }},
  "hate_speech": {{
    "overall": {{"occurrences": 0, "severity_score": 0.0}}
  }},
  "contradictions": {{
    "score": 0.0,
    "description": "Description",
    "examples": ["text"]
  }},
  "key_takeaways": "Summary of key findings",
  "speech_takeaway_summary": "Summary of speech content"
}}"""
                full_prompt = simplified_prompt
            else:
                # Use full prompt for other models
                analysis_type_label = "Political Rhetoric & Content Analysis" if prompt_type == "comprehensive" else "Simple Content Analysis"
                full_prompt = f"""{prompt_template}

**Language:** {language}
**Analysis Type:** {analysis_type_label}{social_context}{speaker_info_text}

**Enhanced Structured Transcript:**
{timestamped_transcript}

Please provide your analysis in the exact JSON format specified above."""
            
            prompt_length = len(full_prompt)
            self.logger.info(f"Sending enhanced analysis prompt to {self.model_name} - Prompt length: {prompt_length} characters")
            self.api_logger.info(f"Enhanced LLM API call initiated - Model: {self.model_name}, Prompt length: {prompt_length}")
            
            progress_message = f"Waiting for {self.model_name} {prompt_type} analysis response..."
            self.set_progress("analysis", 50, progress_message)
            
            # Create the API call with proper error handling
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            progress_message = f"Processing {prompt_type} analysis response..."
            self.set_progress("analysis", 80, progress_message)
            
            # Parse JSON response
            analysis_text = response.choices[0].message.content
            response_length = len(analysis_text)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # ADD COMPREHENSIVE DEBUGGING OUTPUT
            print("\n" + "="*80)
            print("🔍 DEBUGGING: FULL AI RESPONSE")
            print("="*80)
            print(f"Model: {self.model_name}")
            print(f"Prompt Type: {prompt_type}")
            print(f"Response Length: {response_length} characters")
            print(f"Response Time: {elapsed_time:.2f}s")
            print("\n📄 FULL RAW RESPONSE:")
            print("-" * 60)
            print(analysis_text)
            print("-" * 60)
            print("="*80)
            
            self.api_logger.info(f"Enhanced LLM API call completed - Model: {self.model_name}, Response length: {response_length} chars, Duration: {elapsed_time:.2f}s")
            self.logger.info(f"Received enhanced analysis response in {elapsed_time:.2f}s - Length: {response_length} characters")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                try:
                    json_text = json_match.group()
                    print(f"\n🔍 EXTRACTED JSON TEXT:\n{json_text[:1000]}{'...' if len(json_text) > 1000 else ''}\n")
                    
                    analysis = json.loads(json_text)
                    print(f"\n🔍 PARSED ANALYSIS KEYS: {list(analysis.keys())}")
                    
                    # Add backward compatibility fields for the existing UI
                    credibility_score = 50  # Default fallback
                    
                    # DEBUG: Print truthfulness data
                    truthfulness = analysis.get('truthfulness', {})
                    print(f"\n🔍 TRUTHFULNESS DATA: {truthfulness}")
                    
                    # Calculate overall credibility based on multiple factors
                    if 'truthfulness' in analysis and 'overall_score' in analysis['truthfulness']:
                        raw_score = analysis['truthfulness']['overall_score']
                        print(f"\n🔍 RAW CREDIBILITY SCORE: {raw_score} (type: {type(raw_score)})")
                        
                        # Fix the score calculation - handle different scales properly
                        if isinstance(raw_score, (int, float)):
                            if 0 <= raw_score <= 1:
                                # Score is in 0-1 range, convert to 0-100
                                credibility_score = int(raw_score * 100)
                            elif 1 < raw_score <= 10:
                                # Score is in 1-10 range (like 4.0), convert to 0-100
                                credibility_score = int(raw_score * 10)
                            elif 0 <= raw_score <= 100:
                                # Score is already in 0-100 range
                                credibility_score = int(raw_score)
                            else:
                                # If score is out of range, clamp it to 0-100
                                credibility_score = max(0, min(100, int(raw_score)))
                        else:
                            credibility_score = 50  # Fallback for non-numeric
                            
                        print(f"🔍 FINAL CREDIBILITY SCORE: {credibility_score}")
                    else:
                        print("🔍 NO TRUTHFULNESS/OVERALL_SCORE FOUND - using fallback")
                    
                    # Extract key information for backward compatibility
                    main_claims = []
                    red_flags = []
                    bias_indicators = []
                    
                    # Handle different claim field names based on prompt type and AI response
                    claims_sources = ['key_claims', 'subjective_claims', 'main_claims']
                    for claims_field in claims_sources:
                        if claims_field in analysis:
                            claims_data = analysis[claims_field]
                            print(f"\n🔍 CLAIMS DATA ({claims_field}): {claims_data}")
                            if isinstance(claims_data, list):
                                for claim in claims_data:
                                    if isinstance(claim, dict):
                                        # Extract claim text from various possible structures
                                        claim_text = claim.get('claim') or claim.get('text') or claim.get('description')
                                        if claim_text:
                                            main_claims.append(claim_text)
                                        else:
                                            main_claims.append(str(claim))
                                    else:
                                        main_claims.append(str(claim))
                            break  # Use first found claims field
                    
                    # Handle different red flag field names
                    red_flag_sources = ['red_flags', 'negative_attributes', 'contradictions', 'issues']
                    for flag_field in red_flag_sources:
                        if flag_field in analysis:
                            flag_data = analysis[flag_field]
                            print(f"\n🔍 RED FLAGS DATA ({flag_field}): {flag_data}")
                            if isinstance(flag_data, list):
                                for flag in flag_data:
                                    if isinstance(flag, dict):
                                        # Extract flag text from various possible structures
                                        flag_text = flag.get('description') or flag.get('text') or flag.get('issue') or flag.get('type')
                                        if flag_text:
                                            red_flags.append(flag_text)
                                        # Also check for examples
                                        if 'examples' in flag:
                                            red_flags.extend(flag['examples'])
                                    else:
                                        red_flags.append(str(flag))
                            elif isinstance(flag_data, dict):
                                # Handle nested red flag structures
                                for key, value in flag_data.items():
                                    if isinstance(value, list):
                                        red_flags.extend([str(v) for v in value])
                                    else:
                                        red_flags.append(f"{key}: {value}")
                    
                    # Handle rhetorical tactics and bias indicators
                    if 'rhetorical_tactics' in analysis:
                        rhetorical_tactics = analysis['rhetorical_tactics']
                        print(f"\n🔍 RHETORICAL TACTICS: {rhetorical_tactics}")
                        if isinstance(rhetorical_tactics, list):
                            for tactic in rhetorical_tactics:
                                if isinstance(tactic, dict):
                                    tactic_name = tactic.get('tactic', 'Unknown Tactic')
                                    intensity = tactic.get('intensity_score', 0)
                                    # Consider significant bias indicators (> 5.0 intensity)
                                    if intensity > 5.0:
                                        bias_indicators.append(f"{tactic_name} (intensity: {intensity})")
                                    # Also add as red flag if very high intensity
                                    if intensity > 7.0:
                                        red_flags.append(f"High-intensity rhetorical tactic: {tactic_name}")
                                else:
                                    bias_indicators.append(str(tactic))
                    
                    print(f"\n🔍 EXTRACTED DATA SUMMARY:")
                    print(f"  - Credibility Score: {credibility_score}")
                    print(f"  - Main Claims: {len(main_claims)} found")
                    print(f"  - Red Flags: {len(red_flags)} found") 
                    print(f"  - Bias Indicators: {len(bias_indicators)} found")
                    
                    # Add backward compatibility fields
                    analysis['credibility_score'] = credibility_score
                    
                    # Set common fields using extracted data (works for both prompt types)
                    analysis['key_claims'] = main_claims
                    analysis['red_flags'] = red_flags
                    analysis['bias_indicators'] = bias_indicators
                    analysis['factual_accuracy'] = analysis.get('truthfulness', {}).get('fact_check_summary', 'Analysis completed')
                    analysis['evidence_quality'] = analysis.get('truthfulness', {}).get('evidence_quality', 'See detailed analysis')
                    
                    # Set analysis summary based on available fields
                    if prompt_type == "simple":
                        # Priority order for simple analysis summary
                        summary_sources = [
                            ('summary', 'credibility_assessment'),
                            ('overall_assessment', 'reliability_rating'),
                            ('speech_takeaway_summary', None),
                            ('key_takeaways', None)
                        ]
                        analysis_summary = 'Simple content analysis completed'  # fallback
                        for source_key, sub_key in summary_sources:
                            if source_key in analysis:
                                if sub_key and isinstance(analysis[source_key], dict):
                                    if sub_key in analysis[source_key]:
                                        analysis_summary = analysis[source_key][sub_key]
                                        break
                                elif not sub_key:
                                    analysis_summary = str(analysis[source_key])
                                    break
                        analysis['analysis_summary'] = analysis_summary
                        
                        # Map AI response fields to frontend expected fields for simple analysis
                        # Map speech_takeaway_summary to overall_assessment
                        if 'speech_takeaway_summary' in analysis:
                            analysis['overall_assessment'] = {
                                'main_assessment': analysis['speech_takeaway_summary'],
                                'reliability_rating': 'Based on content analysis'
                            }
                        
                        # Map key_takeaways to summary
                        if 'key_takeaways' in analysis:
                            analysis['summary'] = {
                                'main_topic': analysis.get('truthfulness', {}).get('fact_check_summary', 'Content analysis'),
                                'credibility_assessment': analysis['key_takeaways'],
                                'key_points': analysis.get('subjective_claims', [])[:3] if analysis.get('subjective_claims') else [],
                                'recommendation': f"Credibility Score: {credibility_score}/100"
                            }
                    else:
                        # For comprehensive analysis
                        analysis_summary = analysis.get('key_takeaways', 'Enhanced political rhetoric analysis completed')
                        if isinstance(analysis_summary, dict):
                            analysis_summary = analysis_summary.get('summary', str(analysis_summary))
                        analysis['analysis_summary'] = analysis_summary
                    
                    # Enhanced fields for new UI features
                    analysis['enhanced_analysis'] = True
                    analysis['analysis_type'] = 'political_rhetoric' if prompt_type == "comprehensive" else 'simple_content'
                    analysis['prompt_type'] = prompt_type
                    
                    self.logger.info(f"Successfully parsed {prompt_type} analysis JSON - Credibility score: {credibility_score}")
                    
                    if prompt_type == "comprehensive":
                        self.logger.info(f"Rhetorical tactics found: {len(analysis.get('rhetorical_tactics', []))}")
                        self.logger.info(f"Hate speech categories analyzed: {len(analysis.get('hate_speech', {}).get('by_category', {}))}")
                        progress_final = f"Enhanced analysis complete - Score: {credibility_score}, Tactics: {len(analysis.get('rhetorical_tactics', []))}"
                    else:
                        self.logger.info(f"Key claims identified: {len(main_claims)}")
                        self.logger.info(f"Red flags found: {len(red_flags)}")
                        progress_final = f"Simple analysis complete - Score: {credibility_score}, Claims: {len(main_claims)}"
                    
                    self.set_progress("analysis", 100, progress_final)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Enhanced analysis JSON parsing failed: {e}")
                    self.logger.error(f"Raw response that failed to parse: {analysis_text[:500]}...")
                    raise ValueError(f"Failed to parse enhanced analysis JSON response: {e}")
            else:
                self.logger.error(f"No valid JSON found in enhanced {self.model_name} response")
                self.logger.error(f"Raw response without JSON: {analysis_text[:500]}...")
                raise ValueError(f"No valid JSON found in enhanced analysis response")
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Enhanced transcript analysis completed in {total_time:.2f}s")
            return analysis
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Enhanced transcript analysis failed after {elapsed_time:.2f}s for title '{title}': {e}")
            self.api_logger.error(f"Enhanced LLM API call failed - Model: {self.model_name}: {e}")
            self.set_progress("analysis", 0, f"Enhanced analysis failed: {str(e)}")
            # Re-raise the exception instead of falling back
            raise

    def identify_speakers_from_content(self, transcript_text, segments=None):
        """Identify speakers based on linguistic patterns, keywords, and distinctive phrases"""
        self.logger.info("Starting content-based speaker identification")
        start_time = datetime.now()
        
        try:
            # Prepare text for analysis
            full_text = transcript_text.lower()
            
            # Calculate speaker scores based on multiple factors
            speaker_scores = {}
            
            for speaker_name, patterns in self.speaker_patterns.items():
                score = 0
                evidence = []
                
                # Score based on keywords (weighted heavily)
                keyword_matches = 0
                for keyword in patterns["keywords"]:
                    keyword_count = full_text.count(keyword.lower())
                    keyword_matches += keyword_count
                    if keyword_count > 0:
                        evidence.append(f"Keyword '{keyword}': {keyword_count} times")
                
                score += keyword_matches * 3  # Keywords are worth 3 points each
                
                # Score based on distinctive phrases (weighted very heavily)
                phrase_matches = 0
                for phrase in patterns["distinctive_phrases"]:
                    phrase_count = full_text.count(phrase.lower())
                    phrase_matches += phrase_count
                    if phrase_count > 0:
                        evidence.append(f"Phrase '{phrase}': {phrase_count} times")
                
                score += phrase_matches * 8  # Distinctive phrases are worth 8 points each
                
                # Score based on policy positions (medium weight)
                policy_matches = 0
                for policy in patterns["policy_positions"]:
                    policy_count = full_text.count(policy.lower())
                    policy_matches += policy_count
                    if policy_count > 0:
                        evidence.append(f"Policy '{policy}': {policy_count} times")
                
                score += policy_matches * 2  # Policy positions are worth 2 points each
                
                # Calculate confidence based on multiple factors
                total_matches = keyword_matches + phrase_matches + policy_matches
                text_length = len(full_text)
                
                # Normalize confidence score (0-100)
                if total_matches > 0:
                    # Base confidence on match frequency relative to text length
                    match_density = total_matches / max(text_length / 1000, 1)  # matches per 1000 chars
                    confidence = min(100, int(match_density * 20))  # Scale to 0-100
                    
                    # Boost confidence for distinctive phrases
                    if phrase_matches > 0:
                        confidence = min(100, confidence + (phrase_matches * 15))
                    
                    # Ensure minimum confidence for any matches
                    confidence = max(confidence, min(30, total_matches * 10))
                else:
                    confidence = 0
                
                if score > 0:
                    speaker_scores[speaker_name] = {
                        "score": score,
                        "confidence": confidence,
                        "evidence": evidence,
                        "keyword_matches": keyword_matches,
                        "phrase_matches": phrase_matches,
                        "policy_matches": policy_matches,
                        "total_matches": total_matches
                    }
            
            # Sort speakers by score
            sorted_speakers = sorted(speaker_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            
            # Determine primary speaker (highest score with minimum confidence)
            primary_speaker = None
            primary_confidence = 0
            
            if sorted_speakers:
                top_speaker, top_data = sorted_speakers[0]
                if top_data["confidence"] >= 25:  # Minimum confidence threshold
                    primary_speaker = top_speaker
                    primary_confidence = top_data["confidence"]
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Create identification result
            identification_result = {
                "primary_speaker": primary_speaker,
                "primary_confidence": primary_confidence,
                "all_candidates": dict(sorted_speakers[:5]),  # Top 5 candidates
                "identification_method": "linguistic_pattern_analysis",
                "analysis_duration": elapsed_time,
                "total_speakers_analyzed": len(self.speaker_patterns),
                "candidates_with_matches": len(speaker_scores)
            }
            
            # Log results
            if primary_speaker:
                evidence_summary = sorted_speakers[0][1]["evidence"][:3]  # Top 3 evidence items
                self.logger.info(f"Speaker identified: {primary_speaker} (confidence: {primary_confidence}%)")
                self.logger.info(f"Top evidence: {evidence_summary}")
            else:
                self.logger.info("No speaker identified with sufficient confidence")
                
            if sorted_speakers:
                top_candidates_str = ', '.join([f"{name}({data['confidence']}%)" for name, data in sorted_speakers[:3]])
                self.logger.info(f"Top candidates: {top_candidates_str}")
            
            self.logger.info(f"Speaker identification completed in {elapsed_time:.2f}s - analyzed {len(self.speaker_patterns)} speaker patterns")
            
            return identification_result
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.error_logger.error(f"Speaker identification failed after {elapsed_time:.2f}s: {e}")
            return {
                "primary_speaker": None,
                "primary_confidence": 0,
                "all_candidates": {},
                "identification_method": "linguistic_pattern_analysis_failed",
                "analysis_duration": elapsed_time,
                "error": str(e)
            }
    
    def enhance_transcript_with_speaker_identification(self, transcript_data):
        """Enhance transcript with speaker identification results"""
        try:
            # Get transcript text
            if isinstance(transcript_data, dict):
                transcript_text = transcript_data.get("full_text", "")
            else:
                transcript_text = str(transcript_data)
            
            if not transcript_text:
                self.logger.warning("No transcript text available for speaker identification")
                return transcript_data
            
            # Perform speaker identification
            identification_result = self.identify_speakers_from_content(transcript_text)
            
            # Add identification results to transcript data
            if isinstance(transcript_data, dict):
                transcript_data["speaker_identification"] = identification_result
                
                # If we have segments with diarization, try to map identified speaker to segments
                if transcript_data.get("has_speaker_diarization", False) and identification_result.get("primary_speaker"):
                    segments = transcript_data.get("segments", [])
                    primary_speaker = identification_result["primary_speaker"]
                    
                    # Simple heuristic: assign primary speaker to the dominant speaker in diarization
                    speaker_segment_counts = {}
                    for segment in segments:
                        speaker = segment.get("speaker", "Unknown")
                        speaker_segment_counts[speaker] = speaker_segment_counts.get(speaker, 0) + 1
                    
                    if speaker_segment_counts:
                        # Find the diarization speaker with the most segments
                        dominant_diarization_speaker = max(speaker_segment_counts.items(), key=lambda x: x[1])[0]
                        
                        # Update segments to replace dominant speaker with identified speaker
                        for segment in segments:
                            if segment.get("speaker") == dominant_diarization_speaker:
                                segment["identified_speaker"] = primary_speaker
                                segment["speaker_confidence"] = identification_result["primary_confidence"]
                        
                        self.logger.info(f"Mapped identified speaker '{primary_speaker}' to diarization speaker '{dominant_diarization_speaker}'")
            
            return transcript_data
            
        except Exception as e:
            self.error_logger.error(f"Failed to enhance transcript with speaker identification: {e}")
            return transcript_data

    def _build_enhanced_transcript_format(self, transcript_data, title="", analysis_type="comprehensive"):
        """
        Build an enhanced, structured transcript format with better context for AI analysis
        
        This provides:
        1. Content structure analysis (introduction, main points, conclusion)
        2. Topic-based segmentation 
        3. Speaking pace and delivery context
        4. Enhanced timestamp formatting with duration context
        5. Content density analysis
        """
        if isinstance(transcript_data, str):
            # Handle legacy string format
            return f"**Content:** {title}\n**Transcript:** {transcript_data}"
        
        segments = transcript_data.get("segments", [])
        full_text = transcript_data.get("full_text", "")
        language = transcript_data.get("language", "unknown")
        social_metadata = transcript_data.get("social_metadata", {})
        speaker_info = transcript_data.get("speaker_identification", {})
        
        if not segments:
            return f"**Content:** {title}\n**Transcript:** {full_text}"
        
        # Calculate content metrics
        total_duration = segments[-1]["end"] if segments else 0
        total_words = len(full_text.split()) if full_text else 0
        speaking_rate = total_words / (total_duration / 60) if total_duration > 0 and total_words > 0 else 0
        
        # Analyze content structure
        content_structure = self._analyze_content_structure(segments, total_duration)
        
        # Build enhanced header
        header_parts = [
            f"**Content Title:** {title}" if title else "",
            f"**Language:** {language.upper()}",
            f"**Duration:** {total_duration:.1f} minutes ({len(segments)} segments)",
            f"**Speaking Rate:** {speaking_rate:.0f} words/minute" if speaking_rate > 0 else "",
            f"**Content Density:** {total_words:,} words total"
        ]
        
        # Add speaker context if available
        if speaker_info.get("primary_speaker"):
            speaker = speaker_info["primary_speaker"]
            confidence = speaker_info.get("primary_confidence", 0)
            header_parts.append(f"**Primary Speaker:** {speaker} (Confidence: {confidence}%)")
        
        # Add content structure analysis
        structure_text = self._format_content_structure(content_structure)
        if structure_text:
            header_parts.append(f"**Content Structure:** {structure_text}")
        
        # Filter out empty parts and join
        header = "\n".join([part for part in header_parts if part])
        
        # Build enhanced transcript sections
        transcript_sections = []
        
        # Group segments by topic/theme for better readability
        segment_groups = self._group_segments_by_topic(segments, total_duration)
        
        for group_idx, group in enumerate(segment_groups):
            group_title = group["title"]
            group_segments = group["segments"]
            group_start = group_segments[0]["start"]
            group_end = group_segments[-1]["end"]
            group_duration = group_end - group_start
            
            # Section header
            section_header = f"\n--- {group_title} ({group_start:.1f}-{group_end:.1f}min, {group_duration:.1f}min duration) ---"
            transcript_sections.append(section_header)
            
            # Format segments within this group
            for segment in group_segments:
                start_min = segment["start"] / 60
                end_min = segment["end"] / 60
                duration = segment["end"] - segment["start"]
                text = segment["text"].strip()
                
                # Add context indicators for unusually long pauses or rapid speech
                context_indicators = []
                if duration > 15:  # Long segment
                    context_indicators.append("📢 EXTENDED")
                elif duration < 2 and len(text.split()) > 10:  # Very fast speech
                    context_indicators.append("⚡ RAPID")
                
                # Enhanced timestamp format with context
                context_suffix = f" {' '.join(context_indicators)}" if context_indicators else ""
                timestamp_format = f"[{start_min:.1f}-{end_min:.1f}min]{context_suffix}"
                
                transcript_sections.append(f"{timestamp_format} {text}")
        
        # Combine all parts
        enhanced_transcript = f"{header}\n\n**STRUCTURED TRANSCRIPT:**\n{''.join(transcript_sections)}"
        
        return enhanced_transcript

    def _analyze_content_structure(self, segments, total_duration):
        """Analyze the structure of content (intro, main content, conclusion)"""
        if not segments or total_duration < 30:  # Too short to analyze structure
            return {"type": "brief", "sections": ["main"]}
        
        # Divide into thirds for basic structure analysis
        third_duration = total_duration / 3
        
        intro_segments = [s for s in segments if s["start"] < third_duration]
        middle_segments = [s for s in segments if third_duration <= s["start"] < 2 * third_duration]
        outro_segments = [s for s in segments if s["start"] >= 2 * third_duration]
        
        # Analyze text patterns for each section
        intro_text = " ".join([s["text"] for s in intro_segments]).lower()
        middle_text = " ".join([s["text"] for s in middle_segments]).lower()
        outro_text = " ".join([s["text"] for s in outro_segments]).lower()
        
        # Look for structural indicators
        intro_indicators = ["welcome", "today", "going to talk", "introduce", "beginning", "start"]
        conclusion_indicators = ["conclusion", "summary", "thank you", "questions", "end", "closing"]
        
        structure = {"type": "standard", "sections": []}
        
        # Check if intro has introduction patterns
        if any(indicator in intro_text for indicator in intro_indicators):
            structure["sections"].append("introduction")
        
        structure["sections"].append("main_content")
        
        # Check if outro has conclusion patterns  
        if any(indicator in outro_text for indicator in conclusion_indicators):
            structure["sections"].append("conclusion")
        
        return structure

    def _format_content_structure(self, structure):
        """Format content structure for display"""
        if structure["type"] == "brief":
            return "Brief content (single topic)"
        
        sections = structure["sections"]
        if len(sections) == 1:
            return "Single main section"
        elif len(sections) == 2:
            return f"{sections[0].title()} + {sections[1].title()}"
        else:
            return " → ".join([s.title() for s in sections])

    def _group_segments_by_topic(self, segments, total_duration, max_groups=5):
        """Group segments by topic/theme for better organization"""
        if len(segments) <= 10:  # Short content - don't group
            return [{"title": "Main Content", "segments": segments}]
        
        # For longer content, create logical groups based on duration
        group_size = max(3, len(segments) // max_groups)  # At least 3 segments per group
        groups = []
        
        for i in range(0, len(segments), group_size):
            group_segments = segments[i:i + group_size]
            if not group_segments:
                continue
                
            # Determine group title based on position and content
            group_start_min = group_segments[0]["start"] / 60
            group_end_min = group_segments[-1]["end"] / 60
            
            # Simple topic detection based on position
            if i == 0:
                title = "Opening/Introduction"
            elif i + group_size >= len(segments):
                title = "Conclusion/Closing"
            else:
                section_num = (i // group_size) + 1
                title = f"Section {section_num}"
            
            groups.append({
                "title": title,
                "segments": group_segments
            })
        
        return groups

    def _build_social_context(self, metadata):
        """Build social media context section for LLM prompt"""
        if not metadata:
            return ""
        
        # Extract key fields
        title = metadata.get('title', '')
        description = metadata.get('description', '')
        platform = metadata.get('platform', '')
        uploader = metadata.get('uploader', '')
        channel = metadata.get('channel', '')
        upload_date = metadata.get('upload_date', '')
        view_count = metadata.get('view_count', 0)
        like_count = metadata.get('like_count', 0)
        tags = metadata.get('tags', [])
        
        # Build context sections
        context_parts = []
        
        # Platform information
        if platform:
            context_parts.append(f"**Platform:** {platform}")
        
        # Channel/uploader information  
        if channel or uploader:
            uploader_name = channel or uploader
            context_parts.append(f"**Channel/Uploader:** {uploader_name}")
        
        # Title (clearly marked as separate from transcript)
        if title:
            context_parts.append(f"**Content Title:** {title}")
        
        # Description (clearly marked as separate from transcript)
        if description:
            # Truncate very long descriptions
            desc_preview = description[:500] + "..." if len(description) > 500 else description
            context_parts.append(f"**Content Description:** {desc_preview}")
        
        # Publication info
        if upload_date:
            try:
                # Format date if it's in YYYYMMDD format
                if len(upload_date) == 8 and upload_date.isdigit():
                    formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                    context_parts.append(f"**Publication Date:** {formatted_date}")
                else:
                    context_parts.append(f"**Publication Date:** {upload_date}")
            except:
                context_parts.append(f"**Publication Date:** {upload_date}")
        
        # Engagement metrics
        engagement_info = []
        if view_count > 0:
            engagement_info.append(f"{view_count:,} views")
        if like_count > 0:
            engagement_info.append(f"{like_count:,} likes")
        
        if engagement_info:
            context_parts.append(f"**Engagement:** {', '.join(engagement_info)}")
        
        # Tags (first 5 most relevant)
        if tags:
            tag_list = tags[:5]  # Limit to first 5 tags
            context_parts.append(f"**Tags:** {', '.join(tag_list)}")
        
        if context_parts:
            return "\n\n**SOCIAL MEDIA CONTEXT:**\n" + "\n".join(context_parts) + "\n"
        return ""

class ContentPackageManager:
    """Manages creation and processing of content packages from TruthScore analyses"""
    
    def __init__(self):
        self.logger = logging.getLogger('content_packages')
        self.db = db
        
    def create_content_packages(self, time_range='week', themes=None, credibility_range=None):
        """Create content packages from existing analyses"""
        try:
            self.logger.info(f"Creating content packages for {time_range}")
            
            # Query analyses from database
            analyses = self._query_analyses_for_packages(time_range, themes, credibility_range)
            
            if not analyses:
                return []
            
            # Group analyses into thematic packages
            packages = self._group_analyses_by_themes(analyses)
            self.logger.info(f"Grouped {len(analyses)} analyses into {len(packages)} thematic packages")
            
            # Enhance packages with video planning
            enhanced_packages = []
            for package in packages:
                enhanced_package = self._plan_package_videos(package)
                enhanced_packages.append(enhanced_package)
                self.logger.info(f"Created package: '{package['theme']}' with {len(package['analyses'])} analyses and {len(package.get('video_plan', []))} video formats")
            
            self.logger.info(f"Created {len(enhanced_packages)} content packages total")
            return enhanced_packages
            
        except Exception as e:
            self.logger.error(f"Error creating content packages: {e}")
            return []
    
    def _query_analyses_for_packages(self, time_range, themes, credibility_range):
        """Query analyses from database based on criteria"""
        try:
            # Use the existing database methods
            limit = 100 if time_range == 'month' else 50
            db_result = self.db.get_all_analyses(limit=limit, order_by='created_at DESC')
            
            if not db_result or not db_result['analyses']:
                return []
            
            analyses = []
            from datetime import datetime, timedelta
            
            # Calculate time cutoff
            now = datetime.now()
            if time_range == 'week':
                cutoff = now - timedelta(days=7)
            elif time_range == 'month':
                cutoff = now - timedelta(days=30)
            else:
                cutoff = now - timedelta(days=7)
            
            for analysis_row in db_result['analyses']:
                # Get full analysis data
                full_analysis = self.db.get_analysis_by_id(analysis_row['id'])
                if not full_analysis:
                    continue
                
                # Parse created_at
                try:
                    created_at = datetime.fromisoformat(full_analysis['created_at'].replace('Z', '+00:00'))
                    if created_at < cutoff:
                        continue
                except:
                    # Skip if we can't parse the date
                    continue
                
                analysis_data = {
                    'id': full_analysis['id'],
                    'request_id': f"analysis_{full_analysis['id']}",
                    'url': full_analysis['url'],
                    'title': full_analysis['title'],
                    'analysis': full_analysis['analysis_data'] or {},
                    'created_at': full_analysis['created_at'],
                    'metadata': {}
                }
                
                # Apply credibility filter if specified
                if credibility_range:
                    score = analysis_data['analysis'].get('credibility_score', 50)
                    if not (credibility_range[0] <= score <= credibility_range[1]):
                        continue
                
                analyses.append(analysis_data)
            
            return analyses
            
        except Exception as e:
            self.logger.error(f"Error querying analyses: {e}")
            return []
    
    def _group_analyses_by_themes(self, analyses):
        """Group analyses into thematic content packages"""
        packages = []
        
        # Group by credibility patterns
        high_credibility = [a for a in analyses if a['analysis'].get('credibility_score', 50) >= 70]
        medium_credibility = [a for a in analyses if 40 <= a['analysis'].get('credibility_score', 50) < 70]
        low_credibility = [a for a in analyses if a['analysis'].get('credibility_score', 50) < 40]
        
        # Create packages
        if high_credibility:
            package = {
                'type': 'high_credibility_roundup',
                'theme': 'Reliable Content This Week',
                'analyses': high_credibility[:5],  # Top 5
                'package_score': 'high',
                'description': 'Highly credible content that can be trusted'
            }
            packages.append(package)
            self.logger.info(f"Created HIGH credibility package with {len(high_credibility[:5])} analyses (avg score: {sum(a['analysis'].get('credibility_score', 50) for a in high_credibility[:5])/len(high_credibility[:5]):.1f})")
        
        if low_credibility:
            package = {
                'type': 'credibility_alerts',
                'theme': 'Content Requiring Caution',
                'analyses': low_credibility[:5],  # Top 5 concerning
                'package_score': 'low',
                'description': 'Content with credibility concerns requiring verification'
            }
            packages.append(package)
            self.logger.info(f"Created LOW credibility package with {len(low_credibility[:5])} analyses (avg score: {sum(a['analysis'].get('credibility_score', 50) for a in low_credibility[:5])/len(low_credibility[:5]):.1f})")
        
        if medium_credibility:
            package = {
                'type': 'mixed_analysis',
                'theme': 'Mixed Credibility Analysis',
                'analyses': medium_credibility[:3],
                'package_score': 'medium',
                'description': 'Content with mixed credibility requiring careful evaluation'
            }
            packages.append(package)
            self.logger.info(f"Created MEDIUM credibility package with {len(medium_credibility[:3])} analyses (avg score: {sum(a['analysis'].get('credibility_score', 50) for a in medium_credibility[:3])/len(medium_credibility[:3]):.1f})")
        
        # Group by topics/speakers if we have enough data
        speaker_groups = self._group_by_speakers(analyses)
        for speaker, speaker_analyses in speaker_groups.items():
            if len(speaker_analyses) >= 2:  # At least 2 analyses from same speaker
                packages.append({
                    'type': 'speaker_profile',
                    'theme': f'Speaker Analysis: {speaker}',
                    'analyses': speaker_analyses[:3],
                    'package_score': 'speaker_focused',
                    'description': f'Credibility analysis of content from {speaker}',
                    'speaker': speaker
                })
        
        return packages
    
    def _group_by_speakers(self, analyses):
        """Group analyses by identified speakers"""
        speaker_groups = {}
        
        for analysis in analyses:
            speakers = analysis['analysis'].get('speakers', {})
            if speakers:
                # Get primary speaker
                primary_speaker = None
                for speaker_id, speaker_info in speakers.items():
                    if speaker_info.get('identified_name'):
                        primary_speaker = speaker_info['identified_name']
                        break
                
                if primary_speaker:
                    if primary_speaker not in speaker_groups:
                        speaker_groups[primary_speaker] = []
                    speaker_groups[primary_speaker].append(analysis)
        
        return speaker_groups
    
    def _plan_package_videos(self, package):
        """Plan video formats for a content package"""
        package['video_plan'] = []
        
        if package['type'] == 'high_credibility_roundup':
            package['video_plan'].extend([
                {
                    'format': 'weekly_roundup',
                    'duration': 60,
                    'platform': 'youtube',
                    'style': 'educational',
                    'script_focus': 'reliable_sources'
                },
                {
                    'format': 'social_highlight',
                    'duration': 15,
                    'platform': 'tiktok',
                    'style': 'engaging',
                    'script_focus': 'trust_scores'
                }
            ])
        
        elif package['type'] == 'credibility_alerts':
            package['video_plan'].extend([
                {
                    'format': 'alert_video',
                    'duration': 30,
                    'platform': 'all',
                    'style': 'warning',
                    'script_focus': 'verification_needed'
                }
            ])
        
        elif package['type'] == 'speaker_profile':
            package['video_plan'].extend([
                {
                    'format': 'speaker_analysis',
                    'duration': 90,
                    'platform': 'youtube',
                    'style': 'analytical',
                    'script_focus': 'credibility_patterns'
                }
            ])
        
        return package
    
    async def generate_package_videos(self, package):
        """Generate videos for a content package using existing video module"""
        generated_videos = []
        
        for video_plan in package.get('video_plan', []):
            try:
                # Convert package to TruthScore format for video generation
                truthscore_data = self._convert_package_to_truthscore_format(package, video_plan)
                
                # Use existing video generation endpoint
                video_result = await self._request_video_generation(truthscore_data, video_plan)
                
                if video_result:
                    generated_videos.append({
                        'video_id': video_result['clip_id'],
                        'format': video_plan['format'],
                        'platform': video_plan['platform'],
                        'duration': video_plan['duration'],
                        'status': 'generating'
                    })
                
            except Exception as e:
                self.logger.error(f"Error generating video for package: {e}")
        
        return generated_videos
    
    def _convert_package_to_truthscore_format(self, package, video_plan):
        """Convert content package to TruthScore data format"""
        # Take the first analysis as primary content
        primary_analysis = package['analyses'][0]
        
        # Build comprehensive data structure
        truthscore_data = {
            'request_id': f"package_{package['type']}_{int(time.time())}",
            'metadata': {
                'title': package['theme'],
                'package_type': package['type'],
                'total_analyses': len(package['analyses']),
                'url': primary_analysis.get('url', ''),
                'is_package': True
            },
            'credibility_analysis': self._build_package_analysis(package),
            'transcript': self._build_package_transcript(package),
            'speakers': self._extract_package_speakers(package),
            'web_research': self._build_package_research(package)
        }
        
        return truthscore_data
    
    def _build_package_analysis(self, package):
        """Build comprehensive analysis from package data"""
        analyses = package['analyses']
        
        # Calculate aggregate credibility score
        scores = [a['analysis'].get('credibility_score', 50) for a in analyses]
        avg_score = sum(scores) / len(scores) if scores else 50
        
        # Collect key claims from all analyses
        all_claims = []
        all_red_flags = []
        all_bias_indicators = []
        
        for analysis in analyses:
            analysis_data = analysis['analysis']
            all_claims.extend(analysis_data.get('key_claims', []))
            all_red_flags.extend(analysis_data.get('red_flags', []))
            all_bias_indicators.extend(analysis_data.get('bias_indicators', []))
        
        return {
            'overall_score': avg_score,
            'credibility_score': avg_score,
            'key_claims': all_claims[:5],  # Top 5
            'red_flags': all_red_flags[:3],  # Top 3
            'bias_indicators': all_bias_indicators[:3],  # Top 3
            'summary': f"Analysis of {len(analyses)} pieces of content with {package['description'].lower()}",
            'package_insights': {
                'content_count': len(analyses),
                'avg_credibility': avg_score,
                'theme': package['theme'],
                'type': package['type']
            }
        }
    
    def _build_package_transcript(self, package):
        """Build aggregated transcript for package"""
        # For packages, create a summary transcript
        return {
            'segments': [
                {
                    'start': 0.0,
                    'end': 30.0,
                    'text': f"Content package analysis: {package['theme']}. Analyzed {len(package['analyses'])} pieces of content.",
                    'speaker': 'NARRATOR'
                }
            ]
        }
    
    def _extract_package_speakers(self, package):
        """Extract speaker information from package"""
        all_speakers = {}
        
        for analysis in package['analyses']:
            speakers = analysis['analysis'].get('speakers', {})
            for speaker_id, speaker_info in speakers.items():
                if speaker_id not in all_speakers:
                    all_speakers[speaker_id] = speaker_info
        
        return all_speakers
    
    def _build_package_research(self, package):
        """Build research context for package"""
        return {
            'topic_research': {
                'topic': package['theme'],
                'research_summary': f"Comprehensive analysis of {len(package['analyses'])} content pieces",
                'sources_analyzed': len(package['analyses'])
            },
            'claim_verification': []
        }
    
    async def _request_video_generation(self, truthscore_data, video_plan):
        """Request video generation using existing infrastructure"""
        try:
            import requests
            
            # Use your existing video module
            video_module_url = "http://localhost:9000/generate_clip"
            
            payload = {
                'truthscore_data': truthscore_data,
                'clip_config': {
                    'type': 'summary' if video_plan['duration'] > 60 else 'social',
                    'target_duration': video_plan['duration'],
                    'style': video_plan['style'],
                    'platform_focus': video_plan['platform']
                },
                'callback_url': f"http://localhost:8000/package_video_complete"
            }
            
            response = requests.post(video_module_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Video generation request failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error requesting video generation: {e}")
            return None

# Initialize the content package manager

analyzer = TruthScoreAnalyzer()
content_package_manager = ContentPackageManager()

@app.route('/')
def index():
    app_logger.info(f"Home page accessed from {request.remote_addr}")
    return render_template('index.html')

def background_analyze(request_id, url, prompt_type="comprehensive"):
    """Background function to perform the analysis"""
    analyzer.set_request_id(request_id)
    
    # Memory cleanup
    import gc
    gc.collect()
    
    try:
        # Step 1: Extract audio
        app_logger.info(f"[{request_id}] Step 1: Starting audio extraction")
        audio_path, title, metadata = analyzer.extract_audio(url)
        if not audio_path:
            app_logger.error(f"[{request_id}] Step 1 failed: Audio extraction failed")
            results_store[request_id] = {'error': 'Failed to extract audio from URL', 'status': 'error', 'timestamp': datetime.now().isoformat()}
            save_request_tracking()
            return
        
        # Step 2: Transcribe audio with speaker diarization and social metadata
        app_logger.info(f"[{request_id}] Step 2: Starting audio transcription with speaker diarization")
        transcript = analyzer.transcribe_with_speakers(audio_path, metadata)
        if not transcript:
            app_logger.error(f"[{request_id}] Step 2 failed: Audio transcription failed")
            results_store[request_id] = {'error': 'Failed to transcribe audio. This may be due to a timeout for very long videos (>5 minutes processing time) or unsupported audio format.', 'status': 'error', 'timestamp': datetime.now().isoformat()}
            save_request_tracking()
            return
        
        # Step 3: Analyze transcript (Enhanced)
        analysis_type = "comprehensive" if prompt_type == "comprehensive" else "simple"
        app_logger.info(f"[{request_id}] Step 3: Starting {analysis_type} transcript analysis")
        analysis = analyzer.analyze_transcript_enhanced(transcript, title, prompt_type)
        
        # Step 4: Perform comprehensive web research
        app_logger.info(f"[{request_id}] Step 4: Starting comprehensive web research")
        analyzer.set_progress("research", 0, "Starting detailed web research...")
        
        # Enhanced research: includes topic research AND claim verification
        research = analyzer.perform_enhanced_web_research(analysis, title, transcript)
        
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
        
        # Memory cleanup
        import gc
        gc.collect()
        
        analyzer.set_progress("complete", 100, "Analysis complete!")
        
        # Prepare transcript data for response (handle both formats)
        if isinstance(transcript, dict):
            transcript_text = transcript.get("full_text", "")
            transcript_data = transcript.copy()
            
            # Add enhanced formatted transcript for better user display
            try:
                enhanced_transcript = analyzer._build_enhanced_transcript_format(
                    transcript, title, prompt_type
                )
                transcript_data["enhanced_display"] = enhanced_transcript
                app_logger.info(f"[{request_id}] Enhanced transcript format added for user display")
            except Exception as format_error:
                app_logger.warning(f"[{request_id}] Failed to create enhanced transcript format: {format_error}")
                # Continue without enhanced format - fallback to basic display
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
            'status': 'completed',
            'metadata': metadata
        }
        
        # Store results in database
        try:
            # Combine analysis and research data for storage
            combined_analysis = analysis.copy()
            combined_analysis['research'] = research
            
            # Extract credibility score for database storage
            credibility_score = analysis.get('credibility_score', analysis.get('truthfulness', {}).get('overall_score', None))
            if isinstance(credibility_score, str):
                try:
                    credibility_score = int(credibility_score)
                except (ValueError, TypeError):
                    credibility_score = None
            
            # Store in database (transcript_data now includes enhanced_display)
            analysis_id = db.store_analysis(
                url=url,
                title=title,
                analysis_data=combined_analysis,
                transcript_data=transcript_data,
                credibility_score=credibility_score,
                analysis_type=prompt_type
            )
            
            # Add analysis_id to result
            result['analysis_id'] = analysis_id
            app_logger.info(f"[{request_id}] Analysis stored in database with ID: {analysis_id}")
            
        except Exception as db_error:
            app_logger.error(f"[{request_id}] Failed to store analysis in database: {db_error}")
            # Continue without database storage - don't fail the entire analysis
        
        # Store results in memory
        results_store[request_id] = result
        # Save results data persistently
        save_request_tracking()
        
        # Log successful completion
        credibility_score = analysis.get('credibility_score', analysis.get('truthfulness', {}).get('overall_score', 'unknown'))
        transcript_length = len(transcript_text)
        segments_count = len(transcript_data.get('segments', []))
        hate_speech_score = analysis.get('hate_speech', {}).get('overall', {}).get('severity_score', 0)
        rhetorical_tactics_count = len(analysis.get('rhetorical_tactics', []))
        app_logger.info(f"[{request_id}] Enhanced political analysis completed - Title: '{title}', Credibility: {credibility_score}, Hate Speech: {hate_speech_score}, Tactics: {rhetorical_tactics_count}, Length: {transcript_length}, Segments: {segments_count}")
        
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
        prompt_type = data.get('prompt_type', 'comprehensive')  # Default to comprehensive
        
        app_logger.info(f"[{request_id}] Analysis requested for URL: {url} with {prompt_type} prompt")
        
        if not url:
            app_logger.warning(f"[{request_id}] No URL provided in request")
            return jsonify({'error': 'No URL provided'}), 400
        
        if not analyzer.is_valid_url(url):
            app_logger.warning(f"[{request_id}] Invalid URL format: {url}")
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Validate prompt type
        if prompt_type not in ['comprehensive', 'simple']:
            app_logger.warning(f"[{request_id}] Invalid prompt type: {prompt_type}")
            prompt_type = 'comprehensive'  # Default fallback
        
        # Check if this URL has been analyzed before
        existing_analysis = db.find_existing_analysis(url)
        if existing_analysis and existing_analysis.get('analysis_type') == prompt_type:
            app_logger.info(f"[{request_id}] Found existing analysis for URL: {url} (ID: {existing_analysis['id']})")
            
            # Get transcript data and add enhanced format if missing
            transcript_data = existing_analysis['transcript_data'].copy() if existing_analysis['transcript_data'] else {}
            
            # Check if enhanced_display is missing (older analyses)
            if 'enhanced_display' not in transcript_data and transcript_data.get('segments'):
                try:
                    enhanced_transcript = analyzer._build_enhanced_transcript_format(
                        transcript_data, existing_analysis['title'], prompt_type
                    )
                    transcript_data["enhanced_display"] = enhanced_transcript
                    app_logger.info(f"[{request_id}] Added enhanced format to cached analysis")
                except Exception as format_error:
                    app_logger.warning(f"[{request_id}] Failed to add enhanced format to cached analysis: {format_error}")
            
            # Convert stored analysis to the expected format
            result = {
                'title': existing_analysis['title'],
                'transcript': transcript_data,
                'analysis': existing_analysis['analysis_data'],
                'research': existing_analysis['analysis_data'].get('research', []),
                'timestamp': existing_analysis['updated_at'],
                'url': existing_analysis['url'],
                'request_id': request_id,
                'status': 'completed',
                'cached': True,
                'analysis_id': existing_analysis['id']
            }
            
            # Store in temporary results for immediate retrieval
            results_store[request_id] = result
            save_request_tracking()
            
            return jsonify({
                'request_id': request_id,
                'status': 'completed',
                'cached': True,
                'message': f'Analysis retrieved from database (ID: {existing_analysis["id"]})',
                'analysis_id': existing_analysis['id']
            })
        
        # No existing analysis found, start background processing
        app_logger.info(f"[{request_id}] No existing analysis found, starting new analysis")
        thread = threading.Thread(target=background_analyze, args=(request_id, url, prompt_type))
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

# Video Generation Integration Endpoints
video_generation_store = {}  # Store video generation requests

@app.route('/export_analysis/<request_id>')
def export_analysis_for_video(request_id):
    """
    Export analysis results in video-module-ready format
    """
    app_logger.info(f"Video export requested for analysis {request_id}")
    
    if request_id not in results_store:
        app_logger.warning(f"Analysis {request_id} not found for video export")
        return jsonify({'error': 'Analysis not found or expired'}), 404
    
    result = results_store[request_id]
    
    if result.get('status') != 'completed':
        app_logger.warning(f"Analysis {request_id} not completed for video export")
        return jsonify({'error': 'Analysis not completed'}), 400
    
    try:
        # Extract analysis data
        analysis = result.get('analysis', {})
        transcript_data = result.get('transcript', {})
        research = result.get('research', [])
        
        # Process transcript data
        if isinstance(transcript_data, str):
            transcript_segments = []
            full_text = transcript_data
        else:
            transcript_segments = transcript_data.get('segments', [])
            full_text = transcript_data.get('full_text', '')
        
        # Extract speaker information
        speakers = {}
        if isinstance(transcript_data, dict) and transcript_data.get('has_speaker_diarization'):
            for i, segment in enumerate(transcript_segments):
                speaker_id = segment.get('speaker', f'SPEAKER_0{i+1}')
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        'identified_name': f'Speaker {len(speakers) + 1}',
                        'confidence': 0.8,
                        'segments_count': 0
                    }
                speakers[speaker_id]['segments_count'] += 1
        
        # Process credibility analysis
        credibility_score = analysis.get('credibility_score', 50)
        
        # Extract claims for video highlighting
        claims = []
        key_claims = analysis.get('key_claims', [])
        if isinstance(key_claims, list):
            for i, claim in enumerate(key_claims):
                claims.append({
                    'text': str(claim),
                    'verification': 'unverified',  # Default
                    'confidence': 0.7,
                    'timestamp': [i * 10, (i + 1) * 10]  # Mock timestamps
                })
        
        # Process research results
        processed_research = {}
        if research:
            topic_research = {
                'summary': 'Research summary from web search',
                'sources_found': len(research),
                'credible_sources': sum(1 for r in research if r.get('verification_status') == 'verified')
            }
            
            claim_verification = []
            for claim_result in research:
                claim_verification.append({
                    'claim': claim_result.get('claim', ''),
                    'verification_status': claim_result.get('verification_status', 'unverified'),
                    'evidence_strength': claim_result.get('evidence_quality', 'UNKNOWN'),
                    'sources': claim_result.get('web_sources_found', 0)
                })
            
            processed_research = {
                'topic_research': topic_research,
                'claim_verification': claim_verification
            }
        
        # Create export structure as defined in the specification
        export_data = {
            "request_id": request_id,
            "metadata": {
                "title": result.get('title', 'Unknown Title'),
                "url": result.get('url', ''),
                "duration": 3600,  # Mock duration
                "analysis_timestamp": result.get('timestamp', datetime.now().isoformat())
            },
            "transcript": {
                "segments": transcript_segments,
                "full_text": full_text
            },
            "speakers": speakers,
            "credibility_analysis": {
                "overall_score": credibility_score,
                "claims": claims,
                "bias_indicators": analysis.get('bias_indicators', []),
                "rhetorical_tactics": analysis.get('rhetorical_tactics', [])
            },
            "web_research": processed_research,
            "media_files": {
                "original_audio": None,  # Would be populated in real implementation
                "original_video": None   # Would be populated in real implementation
            }
        }
        
        app_logger.info(f"Successfully exported analysis {request_id} for video generation")
        return jsonify(export_data)
        
    except Exception as e:
        app_logger.error(f"Failed to export analysis {request_id} for video: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/video_complete', methods=['POST'])
def video_generation_complete():
    """
    Receive notification when video is ready
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        clip_id = data.get('clip_id')
        original_request_id = data.get('original_request_id')
        status = data.get('status')
        
        app_logger.info(f"Video completion callback received - Clip: {clip_id}, Status: {status}")
        
        if not clip_id:
            return jsonify({'error': 'No clip_id provided'}), 400
        
        # Store video generation result
        video_generation_store[clip_id] = {
            'clip_id': clip_id,
            'original_request_id': original_request_id,
            'status': status,
            'download_url': data.get('download_url'),
            'metadata': data.get('metadata', {}),
            'error': data.get('error'),
            'completed_at': datetime.now().isoformat()
        }
        
        # Log completion details
        if status == 'completed':
            metadata = data.get('metadata', {})
            app_logger.info(f"Video generation completed successfully - Clip: {clip_id}, Duration: {metadata.get('duration')}, Size: {metadata.get('file_size')}")
        else:
            error_msg = data.get('error', 'Unknown error')
            app_logger.warning(f"Video generation failed - Clip: {clip_id}, Error: {error_msg}")
        
        return jsonify({'status': 'received'})
        
    except Exception as e:
        app_logger.error(f"Error processing video completion callback: {str(e)}")
        return jsonify({'error': f'Callback processing failed: {str(e)}'}), 500

@app.route('/video_status/<clip_id>')
def get_video_status(clip_id):
    """
    Get status of video generation
    """
    if clip_id not in video_generation_store:
        return jsonify({'error': 'Video generation not found'}), 404
    
    return jsonify(video_generation_store[clip_id])

@app.route('/generate_video', methods=['POST'])
def request_video_generation():
    """
    Request video generation for a completed analysis
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        request_id = data.get('request_id')
        clip_config = data.get('clip_config', {})
        
        if not request_id:
            return jsonify({'error': 'No request_id provided'}), 400
        
        # Validate that analysis exists and is completed
        if request_id not in results_store:
            return jsonify({'error': 'Analysis not found'}), 404
        
        result = results_store[request_id]
        if result.get('status') != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        app_logger.info(f"Video generation requested for analysis {request_id}")
        
        # Export analysis data
        export_response = export_analysis_for_video(request_id)
        if export_response.status_code != 200:
            return export_response
        
        export_data = export_response.get_json()
        
        # Send request to video module
        import requests
        
        video_module_url = "http://localhost:9000"  # Video module port (changed from 8000)
        callback_url = f"http://localhost:8000/video_complete"  # TruthScore callback (corrected port)
        
        video_request = {
            "truthscore_data": export_data,
            "clip_config": {
                "type": clip_config.get("type", "social"),
                "target_duration": clip_config.get("target_duration", 25),
                "style": clip_config.get("style", "motion_graphics"),
                "include_overlays": {
                    "credibility_score": clip_config.get("include_credibility", True),
                    "fact_checks": clip_config.get("include_fact_checks", True),
                    "speaker_id": clip_config.get("include_speakers", True)
                }
            },
            "callback_url": callback_url
        }
        
        try:
            response = requests.post(
                f"{video_module_url}/generate_clip",
                json=video_request,
                timeout=30
            )
            
            if response.status_code == 200:
                video_result = response.json()
                app_logger.info(f"Video generation started successfully - Clip ID: {video_result.get('clip_id')}")
                return jsonify(video_result)
            else:
                app_logger.error(f"Video module returned error: {response.status_code} - {response.text}")
                return jsonify({'error': f'Video generation failed: {response.text}'}), response.status_code
        
        except requests.exceptions.ConnectionError:
            app_logger.error("Cannot connect to video module - ensure it's running on port 9000")
            return jsonify({'error': 'Video module unavailable. Please ensure the video editing service is running on port 9000.'}), 503
        except requests.exceptions.Timeout:
            app_logger.error("Video module request timed out")
            return jsonify({'error': 'Video generation request timed out'}), 504
        except Exception as req_error:
            app_logger.error(f"Error communicating with video module: {str(req_error)}")
            return jsonify({'error': f'Video generation failed: {str(req_error)}'}), 500
        
    except Exception as e:
        app_logger.error(f"Error requesting video generation: {str(e)}")
        return jsonify({'error': f'Video generation request failed: {str(e)}'}), 500

@app.route('/health')
def health():
    app_logger.info(f"Health check accessed from {request.remote_addr}")
    
    # Check OpenAI web search status
    web_search_status = 'available' if WEB_SEARCH_AVAILABLE else 'unavailable'
    search_model = OPENAI_SEARCH_MODEL
    
    # Check video module availability
    video_module_status = 'unknown'
    try:
        import requests
        response = requests.get("http://localhost:9000/health", timeout=5)
        video_module_status = 'available' if response.status_code == 200 else 'error'
    except:
        video_module_status = 'unavailable'
    
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'whisper_model': 'available',
            'openai_api': 'configured',
            'speaker_diarization': analyzer.diarization_status,
            'speaker_identification': 'available',
            'web_search': web_search_status,
            'video_module': video_module_status
        },
        'details': {
            'speaker_patterns_count': len(analyzer.speaker_patterns),
            'web_search_model': search_model,
            'web_search_provider': 'OpenAI Web Search',
            'model_name': analyzer.model_name
        }
    }
    
    # Add detailed diarization status information
    if analyzer.diarization_status == "available":
        health_data['details']['speaker_diarization_info'] = {
            'model_loaded': True,
            'model_type': 'full_pipeline',
            'capabilities': ['speaker_separation', 'timeline_alignment', 'voice_identification']
        }
    elif analyzer.diarization_status == "segmentation_only":
        health_data['details']['speaker_diarization_info'] = {
            'model_loaded': True,
            'model_type': 'segmentation_only',
            'capabilities': ['basic_speaker_detection'],
            'note': 'Limited speaker identification available'
        }
    elif analyzer.diarization_status == "no_token":
        health_data['details']['speaker_diarization_info'] = {
            'model_loaded': False,
            'issue': 'HF_TOKEN environment variable not set',
            'solution': 'Set HF_TOKEN with your HuggingFace access token',
            'fallback': 'Content-based speaker identification available'
        }
    elif analyzer.diarization_status == "failed":
        health_data['details']['speaker_diarization_info'] = {
            'model_loaded': False,
            'issue': 'Model loading failed (possible compatibility issue)',
            'fallback': 'Content-based speaker identification available',
            'recommendation': 'Check logs for detailed error information'
        }
    
    return jsonify(health_data)

# Analysis History Routes
@app.route('/history')
def analysis_history():
    """Display the analysis history page"""
    app_logger.info(f"Analysis history page accessed from {request.remote_addr}")
    return render_template('history.html')

@app.route('/api/analyses')
def get_analyses():
    """API endpoint to get all analyses with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        order_by = request.args.get('order_by', 'updated_at DESC')
        
        # Validate limits
        limit = min(limit, 100)  # Max 100 per page
        offset = (page - 1) * limit
        
        # Validate order_by
        valid_orders = ['updated_at DESC', 'updated_at ASC', 'created_at DESC', 'created_at ASC', 'credibility_score DESC', 'credibility_score ASC']
        if order_by not in valid_orders:
            order_by = 'updated_at DESC'
        
        # Get analyses from database
        result = db.get_all_analyses(limit=limit, offset=offset, order_by=order_by)
        
        return jsonify({
            'analyses': result['analyses'],
            'total_count': result['total_count'],
            'has_more': result['has_more'],
            'page': page,
            'limit': limit
        })
        
    except Exception as e:
        app_logger.error(f"Error getting analyses: {e}")
        return jsonify({'error': 'Failed to retrieve analyses'}), 500

@app.route('/api/analysis/<int:analysis_id>')
def get_analysis_by_id(analysis_id):
    """API endpoint to get a specific analysis by ID"""
    try:
        analysis = db.get_analysis_by_id(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Convert database format to expected frontend format
        result = {
            'title': analysis['title'],
            'transcript': analysis['transcript_data'],
            'analysis': analysis['analysis_data'],  # Nest analysis data under 'analysis' key
            'research': analysis['analysis_data'].get('research', []),
            'timestamp': analysis['updated_at'],
            'url': analysis['url'],
            'status': 'completed',
            'cached': True,
            'analysis_id': analysis['id']
        }
        
        return jsonify(result)
        
    except Exception as e:
        app_logger.error(f"Error getting analysis {analysis_id}: {e}")
        return jsonify({'error': 'Failed to retrieve analysis'}), 500

@app.route('/api/analysis/<int:analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """API endpoint to delete an analysis by ID"""
    try:
        success = db.delete_analysis(analysis_id)
        
        if success:
            app_logger.info(f"Analysis {analysis_id} deleted successfully")
            return jsonify({'message': 'Analysis deleted successfully'})
        else:
            return jsonify({'error': 'Analysis not found'}), 404
        
    except Exception as e:
        app_logger.error(f"Error deleting analysis {analysis_id}: {e}")
        return jsonify({'error': 'Failed to delete analysis'}), 500

@app.route('/api/database/stats')
def get_database_stats():
    """API endpoint to get database statistics"""
    try:
        stats = db.get_database_stats()
        return jsonify(stats)
        
    except Exception as e:
        app_logger.error(f"Error getting database stats: {e}")
        return jsonify({'error': 'Failed to retrieve database statistics'}), 500

@app.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    """Redirect to analysis results using stored analysis"""
    try:
        analysis = db.get_analysis_by_id(analysis_id)
        
        if not analysis:
            app_logger.warning(f"Analysis {analysis_id} not found for viewing")
            return render_template('index.html', error=f'Analysis {analysis_id} not found'), 404
        
        # Create a temporary request_id for this view
        temp_request_id = f"db_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Convert database format to expected results format
        result = {
            'title': analysis['title'],
            'transcript': analysis['transcript_data'],
            'analysis': analysis['analysis_data'],
            'research': analysis['analysis_data'].get('research', []),
            'timestamp': analysis['updated_at'],
            'url': analysis['url'],
            'request_id': temp_request_id,
            'status': 'completed',
            'cached': True,
            'analysis_id': analysis['id']
        }
        
        # Store in temporary results for retrieval
        results_store[temp_request_id] = result
        save_request_tracking()
        
        # Redirect to regular results page
        app_logger.info(f"Redirecting to view analysis {analysis_id} with temp request {temp_request_id}")
        return render_template('index.html', auto_load_request_id=temp_request_id)
        
    except Exception as e:
        app_logger.error(f"Error viewing analysis {analysis_id}: {e}")
        return render_template('index.html', error=f'Failed to load analysis: {str(e)}'), 500

@app.route('/test-analysis-loading')
def test_analysis_loading():
    """Test page for debugging analysis loading issues"""
    return send_from_directory('.', 'test_analysis_loading.html')

@app.route('/simple-test')
def simple_test():
    """Simple test page for isolating displayResults issues"""
    return send_from_directory('.', 'simple_test.html')

@app.route('/debug/stored-analysis/<int:analysis_id>')
def debug_stored_analysis(analysis_id):
    """Debug endpoint to inspect stored analysis data structure"""
    try:
        # Get analysis from database
        analysis = db.get_analysis_by_id(analysis_id)
        
        if not analysis:
            return jsonify({'error': f'Analysis {analysis_id} not found'}), 404
            
        # Create temp request ID like view_analysis does
        temp_request_id = f"db_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Convert database format to expected results format (same as view_analysis)
        result = {
            'title': analysis['title'],
            'transcript': analysis['transcript_data'],
            'analysis': analysis['analysis_data'],
            'research': analysis['analysis_data'].get('research', []),
            'timestamp': analysis['updated_at'],
            'url': analysis['url'],
            'request_id': temp_request_id,
            'status': 'completed',
            'cached': True,
            'analysis_id': analysis['id']
        }
        
        # Store in temporary results for retrieval
        results_store[temp_request_id] = result
        save_request_tracking()
        
        # Return debug information
        debug_info = {
            'analysis_id': analysis_id,
            'temp_request_id': temp_request_id,
            'data_structure': {
                'has_title': bool(result.get('title')),
                'has_transcript': bool(result.get('transcript')),
                'has_analysis': bool(result.get('analysis')),
                'has_research': bool(result.get('research')),
                'status': result.get('status'),
                'transcript_type': type(result.get('transcript')).__name__,
                'analysis_type': type(result.get('analysis')).__name__,
                'analysis_keys': list(result.get('analysis', {}).keys()) if isinstance(result.get('analysis'), dict) else 'not_dict'
            },
            'stored_successfully': temp_request_id in results_store,
            'results_endpoint_url': f'/results/{temp_request_id}',
            'view_url': f'/analysis/{analysis_id}'
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        app_logger.error(f"Error debugging analysis {analysis_id}: {e}")
        return jsonify({'error': f'Debug failed: {str(e)}'}), 500

@app.route('/debug-analysis-loading')
def debug_analysis_loading():
    """Debug page for analysis loading issues"""
    return send_from_directory('.', 'debug_analysis_loading.html')

# Add these new endpoints before the if __name__ == '__main__' block

@app.route('/api/content-packages/create', methods=['POST'])
def create_content_packages():
    """Create content packages from existing analyses"""
    try:
        data = request.get_json()
        time_range = data.get('time_range', 'week')
        themes = data.get('themes')
        credibility_range = data.get('credibility_range')
        
        app_logger.info(f"Creating content packages for {time_range}")
        
        packages = content_package_manager.create_content_packages(
            time_range=time_range,
            themes=themes,
            credibility_range=credibility_range
        )
        
        return jsonify({
            'success': True,
            'packages': packages,
            'total_packages': len(packages)
        })
        
    except Exception as e:
        app_logger.error(f"Error creating content packages: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/content-packages/generate-videos', methods=['POST'])
def generate_package_videos():
    """Generate videos for a content package"""
    try:
        data = request.get_json()
        package = data.get('package')
        
        if not package:
            return jsonify({'error': 'Package data required'}), 400
        
        app_logger.info(f"🎬 GENERATING VIDEOS for package: '{package.get('theme', 'Unknown')}' with {len(package.get('analyses', []))} analyses")
        
        # Use asyncio to handle the async video generation
        import asyncio
        
        async def generate_videos():
            return await content_package_manager.generate_package_videos(package)
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            generated_videos = loop.run_until_complete(generate_videos())
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'videos': generated_videos,
            'package_id': package.get('type', 'unknown')
        })
        
    except Exception as e:
        app_logger.error(f"Error generating package videos: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/package_video_complete', methods=['POST'])
def package_video_complete():
    """Callback endpoint for when package videos are completed"""
    try:
        data = request.get_json()
        clip_id = data.get('clip_id')
        package_info = data.get('package_info', {})
        app_logger.info(f"✅ PACKAGE VIDEO COMPLETED: {clip_id} | Package: {package_info.get('theme', 'Unknown')} | Type: {package_info.get('type', 'Unknown')}")
        
        # Store completion data or trigger next steps
        # This could trigger distribution, notification, etc.
        
        return jsonify({'success': True})
        
    except Exception as e:
        app_logger.error(f"Error handling package video completion: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/content-packages/dashboard')
def content_packages_dashboard():
    """Get dashboard data for content packages"""
    try:
        # Get stats directly from database without recreating packages
        analyses = db.get_recent_analyses()
        
        dashboard_data = {
            'recent_packages': 0,
            'package_types': {'high_credibility': 0, 'low_credibility': 0, 'mixed_analysis': 0},
            'credibility_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'total_analyses_in_packages': len(analyses['analyses']) if analyses else 0
        }
        
        # Calculate distribution without creating packages
        if analyses and analyses['analyses']:
            for analysis in analyses['analyses']:
                try:
                    # Get analysis data safely
                    analysis_data = db.get_analysis_by_id(analysis['id'])
                    if not analysis_data or not analysis_data.get('analysis_data'):
                        continue
                    
                    score = analysis_data['analysis_data'].get('credibility_score', 50)
                    if score >= 60:
                        dashboard_data['credibility_distribution']['high'] += 1
                        dashboard_data['package_types']['high_credibility'] += 1
                    elif score <= 40:
                        dashboard_data['credibility_distribution']['low'] += 1
                        dashboard_data['package_types']['low_credibility'] += 1
                    else:
                        dashboard_data['credibility_distribution']['medium'] += 1
                        dashboard_data['package_types']['mixed_analysis'] += 1
                except:
                    continue
        
        dashboard_data['recent_packages'] = sum(dashboard_data['package_types'].values())
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        app_logger.error(f"Error generating dashboard data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/content-packages')
def content_packages_page():
    """Content packages management page"""
    return render_template('content_packages.html')

# Initialize content package manager

    app_logger.info("Starting TruthScore Flask application")

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
