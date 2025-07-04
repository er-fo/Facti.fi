from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import time
import threading
import asyncio
from datetime import datetime, timedelta
import json
import logging
import tempfile
import shutil
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    HOST = os.getenv('VIDEO_MODULE_HOST', '0.0.0.0')
    PORT = int(os.getenv('VIDEO_MODULE_PORT', 9000))  # Changed from 8000 to 9000 to avoid conflict
    
    # API keys - Load from file like main app
    try:
        with open('../eriks personliga api key', 'r') as f:
            OPENAI_API_KEY = f.read().strip()
    except FileNotFoundError:
        try:
            # Try absolute path fallback
            with open('/Users/erik/Desktop/truthscore/eriks personliga api key', 'r') as f:
                OPENAI_API_KEY = f.read().strip()
        except FileNotFoundError:
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Fallback to env var
    
    # TruthScore integration
    TRUTHSCORE_BASE_URL = os.getenv('TRUTHSCORE_URL', 'http://localhost:5000')
    
    # Processing settings
    MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', 2))
    TEMP_FILE_TTL_HOURS = int(os.getenv('TEMP_FILE_TTL', 24))
    
    # Video settings
    OUTPUT_RESOLUTION = (1080, 1920)  # Portrait format for social media
    DEFAULT_FRAME_RATE = 30
    DEFAULT_BITRATE = '5M'
    
    # File storage
    TEMP_DIR = os.getenv('TEMP_DIR', tempfile.gettempdir() + '/video_module')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', tempfile.gettempdir() + '/clips')

# Initialize directories
os.makedirs(Config.TEMP_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Global stores for tracking jobs
job_store = {}
progress_store = {}
current_jobs = 0

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_clip_id():
    """Generate unique clip ID"""
    return str(uuid.uuid4())

def get_estimated_completion():
    """Calculate estimated completion time"""
    return (datetime.now() + timedelta(minutes=2)).isoformat()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'active_jobs': current_jobs,
        'max_concurrent_jobs': Config.MAX_CONCURRENT_JOBS
    })

@app.route('/generate_clip', methods=['POST'])
def generate_video_clip():
    """
    Generate video clip from TruthScore analysis
    
    Request body:
    {
        "truthscore_data": {
            # Full export from TruthScore
        },
        "clip_config": {
            "type": "social|summary",
            "target_duration": 25,
            "style": "motion_graphics|minimal",
            "include_overlays": {
                "credibility_score": true,
                "fact_checks": true,
                "speaker_id": true
            }
        },
        "callback_url": "http://truthscore:5000/video_complete"
    }
    """
    global current_jobs
    
    try:
        # Check concurrent job limit
        if current_jobs >= Config.MAX_CONCURRENT_JOBS:
            return jsonify({
                'error': 'Maximum concurrent jobs reached. Please try again later.',
                'current_jobs': current_jobs,
                'max_jobs': Config.MAX_CONCURRENT_JOBS
            }), 429
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        truthscore_data = data.get('truthscore_data')
        clip_config = data.get('clip_config', {})
        callback_url = data.get('callback_url')
        
        if not truthscore_data:
            return jsonify({'error': 'No TruthScore data provided'}), 400
        
        # Validate clip config
        clip_type = clip_config.get('type', 'social')
        if clip_type not in ['social', 'summary']:
            return jsonify({'error': 'Invalid clip type. Must be "social" or "summary"'}), 400
        
        # Generate clip ID and set up job tracking
        clip_id = generate_clip_id()
        
        # Initialize job tracking
        job_store[clip_id] = {
            'clip_id': clip_id,
            'status': 'processing',
            'created_at': datetime.now().isoformat(),
            'truthscore_data': truthscore_data,
            'clip_config': clip_config,
            'callback_url': callback_url,
            'progress': 0,
            'current_step': 'Initializing...',
            'estimated_completion': get_estimated_completion()
        }
        
        progress_store[clip_id] = {
            'steps': [
                {'name': 'Content analysis', 'completed': False, 'progress': 0},
                {'name': 'TTS generation', 'completed': False, 'progress': 0},
                {'name': 'Video assembly', 'completed': False, 'progress': 0},
                {'name': 'Final render', 'completed': False, 'progress': 0}
            ]
        }
        
        # Start background processing
        current_jobs += 1
        thread = threading.Thread(target=process_video_generation, args=(clip_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started video generation job {clip_id} for clip type: {clip_type}")
        
        return jsonify({
            'clip_id': clip_id,
            'status': 'processing',
            'estimated_completion': get_estimated_completion(),
            'progress_url': f'/progress/{clip_id}'
        })
        
    except Exception as e:
        logger.error(f"Error starting video generation: {str(e)}")
        return jsonify({'error': f'Failed to start video generation: {str(e)}'}), 500

@app.route('/progress/<clip_id>')
def get_clip_progress(clip_id):
    """Get progress for a specific clip generation job"""
    if clip_id not in job_store:
        return jsonify({'error': 'Clip ID not found'}), 404
    
    job = job_store[clip_id]
    progress = progress_store.get(clip_id, {})
    
    # Calculate estimated remaining time
    estimated_remaining = "Unknown"
    if job['progress'] > 0:
        elapsed_minutes = (datetime.now() - datetime.fromisoformat(job['created_at'])).total_seconds() / 60
        if job['progress'] < 100:
            estimated_total = elapsed_minutes / (job['progress'] / 100)
            remaining_minutes = estimated_total - elapsed_minutes
            estimated_remaining = f"{int(remaining_minutes)}m {int((remaining_minutes % 1) * 60)}s"
    
    return jsonify({
        'clip_id': clip_id,
        'status': job['status'],
        'progress': job['progress'],
        'current_step': job['current_step'],
        'steps': progress.get('steps', []),
        'estimated_remaining': estimated_remaining,
        'created_at': job['created_at']
    })

@app.route('/download/<clip_id>')
def download_clip(clip_id):
    """Download generated video file"""
    if clip_id not in job_store:
        return jsonify({'error': 'Clip ID not found'}), 404
    
    job = job_store[clip_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Clip not ready for download'}), 400
    
    output_file = job.get('output_file')
    if not output_file or not os.path.exists(output_file):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=f"truthscore_clip_{clip_id}.mp4",
        mimetype='video/mp4'
    )

@app.route('/jobs')
def list_jobs():
    """List all jobs (for debugging/monitoring)"""
    jobs_summary = []
    for clip_id, job in job_store.items():
        jobs_summary.append({
            'clip_id': clip_id,
            'status': job['status'],
            'progress': job['progress'],
            'created_at': job['created_at'],
            'clip_type': job.get('clip_config', {}).get('type', 'unknown')
        })
    
    return jsonify({
        'total_jobs': len(jobs_summary),
        'active_jobs': current_jobs,
        'jobs': jobs_summary
    })

def process_video_generation(clip_id):
    """Background video generation processing - Phase 3 Implementation"""
    global current_jobs
    
    try:
        job = job_store[clip_id]
        truthscore_data = job['truthscore_data']
        clip_config = job['clip_config']
        callback_url = job.get('callback_url')
        
        logger.info(f"Processing video generation for clip {clip_id}")
        
        # Import Phase 3 components
        from content_classifier import ContentClassifier
        from tts_generator import TTSGenerator
        from visual_overlays import VisualOverlayRenderer
        from audio_mixer import AudioMixer
        from video_assembler import VideoAssembler
        
        # Step 1: Content Analysis and Classification
        update_job_progress(clip_id, 10, "Analyzing and classifying content...")
        classifier = ContentClassifier()
        video_content = classifier.classify_content(truthscore_data, clip_config.get('type', 'social'))
        complete_step(clip_id, 0)
        
        # Step 2: TTS Generation
        update_job_progress(clip_id, 35, "Generating AI voice-over...")
        openai_key = Config.OPENAI_API_KEY
        if not openai_key:
            raise ValueError("OpenAI API key required for TTS generation")
        
        tts_generator = TTSGenerator(openai_key)
        narration_script = asyncio.run(tts_generator.generate_narration(video_content, clip_config))
        complete_step(clip_id, 1)
        
        # Step 3: Video Assembly and Visual Overlays
        update_job_progress(clip_id, 70, "Assembling video with overlays...")
        
        # Generate visual overlays
        overlay_renderer = VisualOverlayRenderer()
        target_duration = narration_script.total_duration
        overlay_timeline = overlay_renderer.generate_overlay_timeline(video_content, target_duration)
        
        # Create video frames with overlays
        video_assembler = VideoAssembler()
        video_frames = video_assembler.create_video_frames(
            video_content, overlay_timeline, target_duration, Config.OUTPUT_RESOLUTION
        )
        complete_step(clip_id, 2)
        
        # Step 4: Audio Mixing and Final Render
        update_job_progress(clip_id, 95, "Mixing audio and rendering final video...")
        
        # Mix audio tracks
        audio_mixer = AudioMixer()
        mixed_audio = audio_mixer.create_mixed_audio(
            narration_script, 
            target_duration,
            background_music_file=None,  # Could be added in future
            sound_effects=None  # Could be generated based on content
        )
        
        # Final video assembly
        output_file = video_assembler.render_final_video(
            video_frames, 
            mixed_audio, 
            os.path.join(Config.OUTPUT_DIR, f"clip_{clip_id}.mp4")
        )
        complete_step(clip_id, 3)
        
        # Mark as completed
        job['status'] = 'completed'
        job['progress'] = 100
        job['current_step'] = 'Complete'
        job['output_file'] = output_file
        job['completed_at'] = datetime.now().isoformat()
        
        # Calculate file stats
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        # Cleanup temporary files
        try:
            tts_generator.cleanup_temp_files(narration_script.segments)
            audio_mixer.cleanup_temp_files()
            video_assembler.cleanup_temp_files()
        except Exception as cleanup_error:
            logger.warning(f"Cleanup failed: {str(cleanup_error)}")
        
        logger.info(f"Video generation completed for clip {clip_id}: {output_file}")
        
        # Send callback if provided
        if callback_url:
            send_completion_callback(callback_url, clip_id, truthscore_data.get('request_id'), file_size_mb)
        
    except Exception as e:
        logger.error(f"Error processing clip {clip_id}: {str(e)}")
        job = job_store[clip_id]
        job['status'] = 'failed'
        job['error'] = str(e)
        job['completed_at'] = datetime.now().isoformat()
        
        # Send failure callback if provided
        callback_url = job.get('callback_url')
        if callback_url:
            send_failure_callback(callback_url, clip_id, str(e))
    
    finally:
        current_jobs -= 1

def update_job_progress(clip_id, progress, step_message):
    """Update job progress"""
    if clip_id in job_store:
        job_store[clip_id]['progress'] = progress
        job_store[clip_id]['current_step'] = step_message

def complete_step(clip_id, step_index):
    """Mark a step as completed"""
    if clip_id in progress_store:
        steps = progress_store[clip_id]['steps']
        if step_index < len(steps):
            steps[step_index]['completed'] = True
            steps[step_index]['progress'] = 100

def send_completion_callback(callback_url, clip_id, original_request_id, file_size_mb):
    """Send completion callback to TruthScore"""
    try:
        import requests
        
        callback_data = {
            'clip_id': clip_id,
            'original_request_id': original_request_id,
            'status': 'completed',
            'download_url': f'http://localhost:{Config.PORT}/download/{clip_id}',
            'metadata': {
                'duration': 25.3,  # Mock duration
                'file_size': f"{file_size_mb:.1f}MB",
                'resolution': '1080x1920'
            }
        }
        
        response = requests.post(callback_url, json=callback_data, timeout=10)
        logger.info(f"Completion callback sent for clip {clip_id}: {response.status_code}")
        
    except Exception as e:
        logger.error(f"Failed to send completion callback for clip {clip_id}: {str(e)}")

def send_failure_callback(callback_url, clip_id, error_message):
    """Send failure callback to TruthScore"""
    try:
        import requests
        
        callback_data = {
            'clip_id': clip_id,
            'status': 'failed',
            'error': error_message
        }
        
        response = requests.post(callback_url, json=callback_data, timeout=10)
        logger.info(f"Failure callback sent for clip {clip_id}: {response.status_code}")
        
    except Exception as e:
        logger.error(f"Failed to send failure callback for clip {clip_id}: {str(e)}")

if __name__ == '__main__':
    logger.info(f"Starting Video Editing Module on {Config.HOST}:{Config.PORT}")
    logger.info(f"TruthScore integration URL: {Config.TRUTHSCORE_BASE_URL}")
    logger.info(f"Output directory: {Config.OUTPUT_DIR}")
    logger.info(f"Temp directory: {Config.TEMP_DIR}")
    
    app.run(host=Config.HOST, port=Config.PORT, debug=True) 