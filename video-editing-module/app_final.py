#!/usr/bin/env python3
"""
Final Working Video Module for TruthScore

This version uses the working video generator that gracefully handles
MoviePy issues and provides audio-only fallback.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import time
import threading
import asyncio
from datetime import datetime
import json
import logging
import tempfile
from pathlib import Path

from working_video_generator import WorkingVideoGenerator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    HOST = os.getenv('VIDEO_MODULE_HOST', '0.0.0.0')
    PORT = int(os.getenv('VIDEO_MODULE_PORT', 9002))  # Final working port
    
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
    
    # File storage
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', tempfile.gettempdir() + '/truthscore_clips')

# Initialize directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Global stores for tracking jobs
job_store = {}
current_jobs = 0

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_clip_id():
    """Generate unique clip ID"""
    return str(uuid.uuid4())

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-final',
        'active_jobs': current_jobs,
        'max_concurrent_jobs': Config.MAX_CONCURRENT_JOBS,
        'api_key_configured': bool(Config.OPENAI_API_KEY)
    })

@app.route('/generate_clip', methods=['POST'])
def generate_video_clip():
    """Generate video/audio clip from TruthScore data"""
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
        
        # Validate OpenAI API key
        if not Config.OPENAI_API_KEY:
            return jsonify({'error': 'OpenAI API key not configured'}), 500
        
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
            'current_step': 'Initializing...'
        }
        
        # Start background processing
        current_jobs += 1
        thread = threading.Thread(target=process_video_generation, args=(clip_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started video generation job {clip_id}")
        
        return jsonify({
            'clip_id': clip_id,
            'status': 'processing',
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
    
    return jsonify({
        'clip_id': clip_id,
        'status': job['status'],
        'progress': job['progress'],
        'current_step': job['current_step'],
        'created_at': job['created_at'],
        'error': job.get('error')
    })

@app.route('/download/<clip_id>')
def download_clip(clip_id):
    """Download generated video/audio file"""
    if clip_id not in job_store:
        return jsonify({'error': 'Clip ID not found'}), 404
    
    job = job_store[clip_id]
    output_file = job.get('output_file')
    
    if not output_file or not os.path.exists(output_file):
        return jsonify({'error': 'Output file not found'}), 404
    
    # Determine file type and appropriate download name
    file_extension = Path(output_file).suffix
    download_name = f"truthscore_analysis_{clip_id}{file_extension}"
    mimetype = 'video/mp4' if file_extension == '.mp4' else 'audio/mpeg'
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=download_name,
        mimetype=mimetype
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
            'output_type': 'video' if job.get('output_file', '').endswith('.mp4') else 'audio'
        })
    
    return jsonify({
        'total_jobs': len(jobs_summary),
        'active_jobs': current_jobs,
        'jobs': jobs_summary
    })

def process_video_generation(clip_id):
    """Background video generation processing"""
    global current_jobs
    
    try:
        job = job_store[clip_id]
        truthscore_data = job['truthscore_data']
        callback_url = job.get('callback_url')
        
        logger.info(f"Processing video generation for clip {clip_id}")
        
        # Update progress
        job['progress'] = 10
        job['current_step'] = "Initializing video generator..."
        
        # Initialize video generator
        generator = WorkingVideoGenerator(Config.OPENAI_API_KEY)
        
        # Update progress
        job['progress'] = 25
        job['current_step'] = "Generating script and TTS audio..."
        
        # Generate video/audio
        output_file = os.path.join(Config.OUTPUT_DIR, f"truthscore_clip_{clip_id}.mp4")
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            job['progress'] = 50
            job['current_step'] = "Creating content..."
            
            # Generate the video/audio
            final_output_path = loop.run_until_complete(
                generator.generate_video(truthscore_data, output_file)
            )
            
            job['progress'] = 90
            job['current_step'] = "Finalizing..."
            
            # Mark as completed
            job['status'] = 'completed'
            job['progress'] = 100
            job['current_step'] = 'Complete'
            job['output_file'] = final_output_path
            job['completed_at'] = datetime.now().isoformat()
            
            # Calculate file stats
            file_size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
            file_type = 'video' if final_output_path.endswith('.mp4') else 'audio'
            
            logger.info(f"Content generation completed for clip {clip_id}: {final_output_path}")
            logger.info(f"File type: {file_type}, Size: {file_size_mb:.1f}MB")
            
            # Send callback if provided
            if callback_url:
                send_completion_callback(
                    callback_url, clip_id, 
                    truthscore_data.get('request_id'), 
                    file_size_mb, file_type
                )
            
        finally:
            loop.close()
            generator.cleanup()
        
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

def send_completion_callback(callback_url, clip_id, original_request_id, file_size_mb, file_type):
    """Send completion callback to TruthScore"""
    try:
        import requests
        
        callback_data = {
            'clip_id': clip_id,
            'original_request_id': original_request_id,
            'status': 'completed',
            'download_url': f'http://localhost:{Config.PORT}/download/{clip_id}',
            'metadata': {
                'file_type': file_type,
                'file_size': f"{file_size_mb:.1f}MB",
                'format': 'mp4' if file_type == 'video' else 'mp3'
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
    logger.info(f"Starting Final Video Module on {Config.HOST}:{Config.PORT}")
    logger.info(f"TruthScore integration URL: {Config.TRUTHSCORE_BASE_URL}")
    logger.info(f"Output directory: {Config.OUTPUT_DIR}")
    logger.info(f"OpenAI API key configured: {bool(Config.OPENAI_API_KEY)}")
    
    app.run(host=Config.HOST, port=Config.PORT, debug=False) 