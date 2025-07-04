#!/usr/bin/env python3
"""
Simplified Video Module App for Testing

This version uses the simplified video generator to test the complete pipeline
without the complex visual overlay system.
"""

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
from pathlib import Path

from simple_video_generator import SimpleVideoGenerator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    HOST = os.getenv('VIDEO_MODULE_HOST', '0.0.0.0')
    PORT = int(os.getenv('VIDEO_MODULE_PORT', 9001))  # Different port for testing
    
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
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', tempfile.gettempdir() + '/simple_clips')

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
        'version': '1.0.0-simple',
        'active_jobs': current_jobs,
        'max_concurrent_jobs': Config.MAX_CONCURRENT_JOBS
    })

@app.route('/generate_clip', methods=['POST'])
def generate_video_clip():
    """Generate video clip using simplified pipeline"""
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
        thread = threading.Thread(target=process_simple_video_generation, args=(clip_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started simple video generation job {clip_id}")
        
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
        'created_at': job['created_at']
    })

@app.route('/download/<clip_id>')
def download_clip(clip_id):
    """Download generated video file"""
    if clip_id not in job_store:
        return jsonify({'error': 'Clip ID not found'}), 404
    
    job = job_store[clip_id]
    output_file = job.get('output_file')
    
    if not output_file or not os.path.exists(output_file):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=f"truthscore_simple_clip_{clip_id}.mp4",
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
            'created_at': job['created_at']
        })
    
    return jsonify({
        'total_jobs': len(jobs_summary),
        'active_jobs': current_jobs,
        'jobs': jobs_summary
    })

def process_simple_video_generation(clip_id):
    """Background simple video generation processing"""
    global current_jobs
    
    try:
        job = job_store[clip_id]
        truthscore_data = job['truthscore_data']
        callback_url = job.get('callback_url')
        
        logger.info(f"Processing simple video generation for clip {clip_id}")
        
        # Update progress
        job['progress'] = 10
        job['current_step'] = "Initializing video generator..."
        
        # Initialize simple video generator
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        
        generator = SimpleVideoGenerator(Config.OPENAI_API_KEY)
        
        # Update progress
        job['progress'] = 25
        job['current_step'] = "Generating script and TTS..."
        
        # Generate video
        output_file = os.path.join(Config.OUTPUT_DIR, f"simple_clip_{clip_id}.mp4")
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            job['progress'] = 50
            job['current_step'] = "Creating visual frames..."
            
            # Generate the video
            final_video_path = loop.run_until_complete(
                generator.generate_simple_video(truthscore_data, output_file)
            )
            
            job['progress'] = 90
            job['current_step'] = "Finalizing video..."
            
            # Mark as completed
            job['status'] = 'completed'
            job['progress'] = 100
            job['current_step'] = 'Complete'
            job['output_file'] = final_video_path
            job['completed_at'] = datetime.now().isoformat()
            
            # Calculate file stats
            file_size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
            
            logger.info(f"Simple video generation completed for clip {clip_id}: {final_video_path}")
            
            # Send callback if provided
            if callback_url:
                send_completion_callback(callback_url, clip_id, truthscore_data.get('request_id'), file_size_mb)
            
        finally:
            loop.close()
            generator.cleanup()
        
    except Exception as e:
        logger.error(f"Error processing simple video clip {clip_id}: {str(e)}")
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
                'duration': 25.0,  # Estimated duration
                'file_size': f"{file_size_mb:.1f}MB",
                'resolution': '1920x1080'
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
    logger.info(f"Starting Simple Video Module on {Config.HOST}:{Config.PORT}")
    logger.info(f"TruthScore integration URL: {Config.TRUTHSCORE_BASE_URL}")
    logger.info(f"Output directory: {Config.OUTPUT_DIR}")
    logger.info(f"OpenAI API key configured: {bool(Config.OPENAI_API_KEY)}")
    
    app.run(host=Config.HOST, port=Config.PORT, debug=True) 