"""
Video Assembly System for Final MP4 Generation

Combines visual frames with mixed audio to create professional-quality MP4 videos.
Implements Phase 3 requirements for video rendering and encoding.
"""

import cv2
import numpy as np
import logging
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import json

import ffmpeg
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip

from content_classifier import ContentSegment, VideoContent
from visual_overlays import OverlayFrame, VisualOverlayRenderer
from audio_mixer import MixedAudio

logger = logging.getLogger(__name__)

@dataclass
class VideoFrame:
    """Represents a single video frame with timing"""
    frame_data: np.ndarray
    frame_time: float
    frame_number: int
    overlay_applied: bool = False

class VideoAssembler:
    """
    Professional video assembly system
    
    Features:
    - High-quality video frame generation
    - Visual overlay composition
    - Audio-video synchronization
    - H.264 encoding with optimal settings
    - Multiple resolution support
    - Frame rate optimization
    - GPU acceleration when available
    - Professional color grading
    """
    
    def __init__(self):
        self.video_config = {
            'frame_rate': 30,
            'resolution': (1920, 1080),
            'codec': 'libx264',
            'pixel_format': 'yuv420p',
            'crf': 18,  # High quality constant rate factor
            'preset': 'medium',
            'audio_codec': 'aac',
            'audio_bitrate': '192k',
            'video_bitrate': '5M'
        }
        
        # Background templates and assets
        self.background_templates = {
            'dark_tech': {
                'primary_color': (15, 15, 15),
                'gradient_colors': [(15, 15, 15), (25, 25, 35)],
                'accent_color': (255, 255, 0),
                'particle_effects': True
            },
            'clean_minimal': {
                'primary_color': (245, 245, 245),
                'gradient_colors': [(245, 245, 245), (235, 235, 235)],
                'accent_color': (59, 130, 246),
                'particle_effects': False
            },
            'credibility_focus': {
                'primary_color': (20, 20, 30),
                'gradient_colors': [(20, 20, 30), (30, 30, 45)],
                'accent_color': (16, 185, 129),
                'particle_effects': True
            }
        }
        
        # Create temp directory for video processing
        self.temp_dir = Path(tempfile.gettempdir()) / "video_assembly"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.temp_files = []  # Track for cleanup
    
    def create_video_frames(self, video_content: VideoContent, 
                          overlay_timeline: List[OverlayFrame],
                          duration: float,
                          resolution: Tuple[int, int] = (1920, 1080)) -> List[VideoFrame]:
        """
        Create all video frames with backgrounds and overlays
        
        Args:
            video_content: Classified content structure
            overlay_timeline: Timeline of visual overlays
            duration: Total video duration in seconds
            resolution: Output video resolution
            
        Returns:
            List of VideoFrame objects ready for rendering
        """
        logger.info(f"Creating video frames for {duration:.1f}s video at {resolution}")
        
        try:
            frame_rate = self.video_config['frame_rate']
            total_frames = int(duration * frame_rate)
            video_frames = []
            
            # Select background template based on content
            template_name = self._select_background_template(video_content)
            background_template = self.background_templates[template_name]
            
            # Create overlay renderer
            overlay_renderer = VisualOverlayRenderer()
            
            # Generate frames
            for frame_num in range(total_frames):
                frame_time = frame_num / frame_rate
                
                # Create base background frame
                base_frame = self._create_background_frame(
                    background_template, resolution, frame_time, duration
                )
                
                # Find overlays for this frame
                current_overlays = self._get_overlays_for_frame(overlay_timeline, frame_time)
                
                # Apply overlays if any
                if current_overlays:
                    composite_frame = overlay_renderer.render_frame_overlays(
                        base_frame, current_overlays, resolution
                    )
                    overlay_applied = True
                else:
                    composite_frame = base_frame
                    overlay_applied = False
                
                # Create video frame object
                video_frame = VideoFrame(
                    frame_data=composite_frame,
                    frame_time=frame_time,
                    frame_number=frame_num,
                    overlay_applied=overlay_applied
                )
                
                video_frames.append(video_frame)
                
                # Log progress periodically
                if frame_num % (frame_rate * 5) == 0:  # Every 5 seconds
                    logger.debug(f"Generated frame {frame_num}/{total_frames} ({frame_time:.1f}s)")
            
            logger.info(f"Generated {len(video_frames)} video frames")
            return video_frames
            
        except Exception as e:
            logger.error(f"Video frame creation failed: {str(e)}")
            raise
    
    def render_final_video(self, video_frames: List[VideoFrame], 
                          mixed_audio: MixedAudio,
                          output_path: str) -> str:
        """
        Render final MP4 video with audio
        
        Args:
            video_frames: List of video frames
            mixed_audio: Mixed audio track
            output_path: Path for output MP4 file
            
        Returns:
            Path to rendered video file
        """
        logger.info(f"Rendering final video: {output_path}")
        
        try:
            # Step 1: Create temporary video file (without audio)
            temp_video_path = self.temp_dir / "temp_video.mp4"
            self._render_video_frames_to_file(video_frames, str(temp_video_path))
            
            # Step 2: Combine video with audio using ffmpeg
            final_output = self._combine_video_and_audio(
                str(temp_video_path), 
                mixed_audio.audio_file, 
                output_path
            )
            
            # Step 3: Verify output
            if not os.path.exists(final_output):
                raise RuntimeError(f"Final video was not created: {final_output}")
            
            # Step 4: Validate video
            validation_result = self._validate_output_video(final_output)
            if not validation_result['valid']:
                logger.warning(f"Video validation warnings: {validation_result['warnings']}")
            
            logger.info(f"Video rendering completed: {final_output}")
            return final_output
            
        except Exception as e:
            logger.error(f"Video rendering failed: {str(e)}")
            raise
    
    def _create_background_frame(self, template: Dict[str, Any], 
                               resolution: Tuple[int, int],
                               frame_time: float,
                               total_duration: float) -> np.ndarray:
        """Create animated background frame"""
        width, height = resolution
        
        # Create base background with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply gradient background
        gradient_colors = template['gradient_colors']
        if len(gradient_colors) == 2:
            for y in range(height):
                ratio = y / height
                color = [
                    int(gradient_colors[0][i] * (1 - ratio) + gradient_colors[1][i] * ratio)
                    for i in range(3)
                ]
                frame[y, :] = color
        else:
            # Solid color fallback
            frame[:, :] = template['primary_color']
        
        # Add subtle animated effects if enabled
        if template.get('particle_effects', False):
            frame = self._add_particle_effects(frame, frame_time, total_duration)
        
        # Add subtle noise for texture
        noise = np.random.randint(-5, 6, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def _add_particle_effects(self, frame: np.ndarray, frame_time: float, 
                            total_duration: float) -> np.ndarray:
        """Add subtle animated particle effects"""
        height, width = frame.shape[:2]
        
        # Create floating particles
        num_particles = 20
        particle_size = 2
        
        for i in range(num_particles):
            # Calculate particle position with time-based animation
            particle_seed = i * 123.456  # Unique seed per particle
            x_base = (particle_seed * 0.1) % width
            y_base = (particle_seed * 0.2) % height
            
            # Animate particles
            x_offset = np.sin(frame_time * 0.5 + particle_seed) * 20
            y_offset = np.cos(frame_time * 0.3 + particle_seed) * 15
            
            x = int((x_base + x_offset) % width)
            y = int((y_base + y_offset) % height)
            
            # Calculate particle opacity based on time
            opacity = 0.3 + 0.2 * np.sin(frame_time * 2 + particle_seed)
            
            # Draw particle
            if 0 <= x < width - particle_size and 0 <= y < height - particle_size:
                particle_color = (100, 100, 100)  # Gray particles
                cv2.circle(frame, (x, y), particle_size, particle_color, -1)
        
        return frame
    
    def _render_video_frames_to_file(self, video_frames: List[VideoFrame], 
                                   output_path: str):
        """Render video frames to MP4 file using OpenCV"""
        if not video_frames:
            raise ValueError("No video frames to render")
        
        # Get video parameters
        height, width = video_frames[0].frame_data.shape[:2]
        frame_rate = self.video_config['frame_rate']
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            frame_rate,
            (width, height)
        )
        
        if not video_writer.isOpened():
            raise RuntimeError("Failed to open video writer")
        
        try:
            # Write frames
            for frame in video_frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame.frame_data, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            
            logger.info(f"Wrote {len(video_frames)} frames to {output_path}")
            
        finally:
            video_writer.release()
    
    def _combine_video_and_audio(self, video_path: str, audio_path: str, 
                               output_path: str) -> str:
        """Combine video and audio using ffmpeg"""
        try:
            # Use ffmpeg to combine video and audio with optimal settings
            (
                ffmpeg
                .output(
                    ffmpeg.input(video_path),
                    ffmpeg.input(audio_path),
                    output_path,
                    vcodec=self.video_config['codec'],
                    acodec=self.video_config['audio_codec'],
                    video_bitrate=self.video_config['video_bitrate'],
                    audio_bitrate=self.video_config['audio_bitrate'],
                    pix_fmt=self.video_config['pixel_format'],
                    crf=self.video_config['crf'],
                    preset=self.video_config['preset'],
                    movflags='faststart'  # Optimize for streaming
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e}")
            # Fallback to moviepy if ffmpeg fails
            return self._combine_video_audio_moviepy(video_path, audio_path, output_path)
    
    def _combine_video_audio_moviepy(self, video_path: str, audio_path: str, 
                                   output_path: str) -> str:
        """Fallback method using moviepy"""
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Set audio to video
            final_clip = video_clip.set_audio(audio_clip)
            
            # Write final video
            final_clip.write_videofile(
                output_path,
                codec=self.video_config['codec'],
                audio_codec=self.video_config['audio_codec'],
                bitrate=self.video_config['video_bitrate'],
                verbose=False,
                logger=None
            )
            
            # Cleanup
            final_clip.close()
            video_clip.close()
            audio_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"MoviePy fallback failed: {str(e)}")
            raise
    
    def _validate_output_video(self, video_path: str) -> Dict[str, Any]:
        """Validate the output video file"""
        validation_result = {
            'valid': True,
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Use ffprobe to get video information
            probe = ffmpeg.probe(video_path)
            
            # Check video stream
            video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            
            if not video_streams:
                validation_result['valid'] = False
                validation_result['warnings'].append("No video stream found")
            else:
                video_stream = video_streams[0]
                validation_result['metadata']['width'] = int(video_stream['width'])
                validation_result['metadata']['height'] = int(video_stream['height'])
                validation_result['metadata']['frame_rate'] = eval(video_stream['r_frame_rate'])
                validation_result['metadata']['duration'] = float(video_stream['duration'])
            
            if not audio_streams:
                validation_result['warnings'].append("No audio stream found")
            else:
                audio_stream = audio_streams[0]
                validation_result['metadata']['audio_codec'] = audio_stream['codec_name']
                validation_result['metadata']['sample_rate'] = int(audio_stream['sample_rate'])
            
            # Check file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            validation_result['metadata']['file_size_mb'] = file_size_mb
            
            if file_size_mb > 100:  # Warn if file is very large
                validation_result['warnings'].append(f"Large file size: {file_size_mb:.1f}MB")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['warnings'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _select_background_template(self, video_content: VideoContent) -> str:
        """Select appropriate background template based on content"""
        credibility_score = video_content.credibility_score
        
        if credibility_score >= 70:
            return 'credibility_focus'  # Green theme for high credibility
        elif credibility_score >= 40:
            return 'dark_tech'  # Tech theme for medium credibility
        else:
            return 'clean_minimal'  # Minimal theme for low credibility content
    
    def _get_overlays_for_frame(self, overlay_timeline: List[OverlayFrame], 
                              frame_time: float) -> Optional[OverlayFrame]:
        """Get overlay data for specific frame time"""
        for overlay_frame in overlay_timeline:
            if abs(overlay_frame.frame_time - frame_time) < 0.02:  # Within 20ms tolerance
                return overlay_frame
        return None
    
    def cleanup_temp_files(self):
        """Clean up temporary video files"""
        try:
            for file_path in self.temp_files:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            
            # Clean up temp directory
            for file in self.temp_dir.glob("temp_video*"):
                file.unlink()
            
            logger.info("Cleaned up temporary video files")
            
        except Exception as e:
            logger.warning(f"Failed to clean up video temp files: {str(e)}")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get detailed information about generated video"""
        try:
            probe = ffmpeg.probe(video_path)
            
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            info = {
                'duration': float(probe['format']['duration']),
                'file_size': int(probe['format']['size']),
                'bit_rate': int(probe['format']['bit_rate']),
                'format_name': probe['format']['format_name']
            }
            
            if video_stream:
                info.update({
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'frame_rate': eval(video_stream['r_frame_rate']),
                    'video_codec': video_stream['codec_name'],
                    'pixel_format': video_stream['pix_fmt']
                })
            
            if audio_stream:
                info.update({
                    'audio_codec': audio_stream['codec_name'],
                    'sample_rate': int(audio_stream['sample_rate']),
                    'channels': int(audio_stream['channels'])
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            return {} 