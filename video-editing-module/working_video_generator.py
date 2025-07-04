#!/usr/bin/env python3
"""
Working Video Generator for TruthScore

This version fixes import issues and provides robust video generation
with proper error handling and fallbacks.
"""

import os
import logging
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any
import asyncio

import openai
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Try to import MoviePy with proper error handling
try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, ColorClip,
        concatenate_videoclips, concatenate_audioclips
    )
    MOVIEPY_AVAILABLE = True
    print("MoviePy imported successfully")
except ImportError as e:
    print(f"MoviePy import failed: {e}")
    MOVIEPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class WorkingVideoGenerator:
    """Working video generator with robust error handling"""
    
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.temp_dir = Path(tempfile.gettempdir()) / "working_video_gen"
        self.temp_dir.mkdir(exist_ok=True)
        
        if not MOVIEPY_AVAILABLE:
            logger.warning("MoviePy not available - will generate audio-only content")
    
    async def generate_video(self, truthscore_data: Dict[str, Any], 
                           output_path: str) -> str:
        """Generate video from TruthScore data"""
        
        logger.info("Starting video generation")
        
        try:
            # Step 1: Extract key information
            title = truthscore_data.get('metadata', {}).get('title', 'Analysis')
            credibility_score = truthscore_data.get('credibility_analysis', {}).get('overall_score', 50)
            claims = truthscore_data.get('credibility_analysis', {}).get('claims', [])
            
            # Step 2: Generate script
            script = self._generate_script(title, credibility_score, claims)
            logger.info(f"Generated script: {len(script)} characters")
            
            # Step 3: Generate TTS
            tts_file = await self._generate_tts(script)
            logger.info(f"Generated TTS audio: {tts_file}")
            
            if MOVIEPY_AVAILABLE:
                # Step 4: Create visual frames
                frames_dir = self._create_visual_frames(title, credibility_score, claims)
                logger.info(f"Created visual frames in: {frames_dir}")
                
                # Step 5: Assemble video
                final_video = await self._assemble_video(tts_file, frames_dir, output_path)
                logger.info(f"Video generated successfully: {final_video}")
                
                return final_video
            else:
                # Fallback: Just return audio file
                logger.info("Returning audio-only file (MoviePy unavailable)")
                audio_output = output_path.replace('.mp4', '.mp3')
                
                # Copy TTS file to output location
                import shutil
                shutil.copy2(tts_file, audio_output)
                return audio_output
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
    
    def _generate_script(self, title: str, credibility_score: int, 
                        claims: List[Any]) -> str:
        """Generate script for the video"""
        
        if credibility_score >= 70:
            credibility_level = "high"
            credibility_desc = "well-supported and reliable"
        elif credibility_score >= 40:
            credibility_level = "moderate"
            credibility_desc = "partially supported but requires verification"
        else:
            credibility_level = "low"
            credibility_desc = "questionable and needs careful fact-checking"
        
        script_parts = [
            f"This is a TruthScore credibility analysis.",
            f"We analyzed: {title}.",
            f"Our analysis gives this content a credibility score of {credibility_score} out of 100.",
            f"This indicates {credibility_level} credibility - the content appears {credibility_desc}."
        ]
        
        # Add key claims
        if claims and len(claims) > 0:
            script_parts.append("Key claims we identified include:")
            for i, claim in enumerate(claims[:2]):  # Limit to 2 claims for brevity
                claim_text = str(claim) if isinstance(claim, str) else str(claim.get('text', claim))
                # Truncate very long claims
                if len(claim_text) > 150:
                    claim_text = claim_text[:150] + "..."
                script_parts.append(f"Claim {i+1}: {claim_text}")
        
        if credibility_score < 60:
            script_parts.append("We recommend verifying this information through multiple reliable sources before sharing.")
        
        script_parts.append("This analysis was generated by TruthScore AI.")
        
        return " ".join(script_parts)
    
    async def _generate_tts(self, script: str) -> str:
        """Generate TTS audio for the script"""
        
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=script
            )
            
            tts_file = str(self.temp_dir / "narration.mp3")
            
            with open(tts_file, 'wb') as f:
                f.write(response.content)
            
            return tts_file
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    def _create_visual_frames(self, title: str, credibility_score: int, 
                            claims: List[Any]) -> str:
        """Create visual frames for the video"""
        
        frames_dir = self.temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Video dimensions - Portrait format for social media
        width, height = 1080, 1920
        background_color = (20, 25, 40)  # Dark blue
        
        # Title frame
        self._create_text_frame(
            str(frames_dir / "title.png"),
            f"TruthScore Analysis\n\n{title[:80]}{'...' if len(title) > 80 else ''}",
            width, height, background_color,
            font_size=64
        )
        
        # Credibility score frame
        score_color = self._get_credibility_color(credibility_score)
        self._create_score_frame(
            str(frames_dir / "score.png"),
            credibility_score,
            width, height, background_color, score_color
        )
        
        # Summary frame
        summary_text = f"Credibility Analysis Complete\n\nScore: {credibility_score}/100\n\nVerify information independently"
        self._create_text_frame(
            str(frames_dir / "summary.png"),
            summary_text,
            width, height, background_color,
            font_size=48
        )
        
        return str(frames_dir)
    
    def _create_text_frame(self, output_path: str, text: str, width: int, height: int,
                         bg_color: tuple, font_size: int = 48):
        """Create a frame with text"""
        
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to use a system font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Handle multi-line text
        lines = text.split('\n')
        line_height = font_size + 10
        total_height = len(lines) * line_height
        
        start_y = (height - total_height) // 2
        
        for i, line in enumerate(lines):
            # Calculate text position (centered)
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            
            x = (width - text_width) // 2
            y = start_y + (i * line_height)
            
            # Draw text with slight shadow for better readability
            draw.text((x+2, y+2), line, fill=(0, 0, 0), font=font)  # Shadow
            draw.text((x, y), line, fill=(255, 255, 255), font=font)  # Main text
        
        img.save(output_path)
    
    def _create_score_frame(self, output_path: str, score: int, width: int, height: int,
                          bg_color: tuple, score_color: tuple):
        """Create a frame showing the credibility score"""
        
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to get fonts
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 120)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw score
        score_text = f"{score}/100"
        bbox = draw.textbbox((0, 0), score_text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2 - 50
        
        # Shadow
        draw.text((x+3, y+3), score_text, fill=(0, 0, 0), font=font_large)
        # Main text
        draw.text((x, y), score_text, fill=score_color, font=font_large)
        
        # Draw label
        label_text = "Credibility Score"
        bbox = draw.textbbox((0, 0), label_text, font=font_small)
        label_width = bbox[2] - bbox[0]
        
        x_label = (width - label_width) // 2
        y_label = y + text_height + 30
        
        # Shadow
        draw.text((x_label+2, y_label+2), label_text, fill=(0, 0, 0), font=font_small)
        # Main text
        draw.text((x_label, y_label), label_text, fill=(255, 255, 255), font=font_small)
        
        img.save(output_path)
    
    def _get_credibility_color(self, score: int) -> tuple:
        """Get color based on credibility score"""
        if score >= 70:
            return (76, 175, 80)  # Green
        elif score >= 40:
            return (255, 193, 7)  # Yellow/Orange
        else:
            return (244, 67, 54)  # Red
    
    async def _assemble_video(self, tts_file: str, frames_dir: str, output_path: str) -> str:
        """Assemble the final video using MoviePy"""
        
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError("MoviePy not available for video assembly")
        
        try:
            # Load audio
            audio = AudioFileClip(tts_file)
            duration = audio.duration
            
            # Create video clips from frames
            clips = []
            frames_path = Path(frames_dir)
            
            # Duration per frame (distribute time across frames)
            frame_duration = duration / 3
            
            frame_files = ['title.png', 'score.png', 'summary.png']
            
            for i, frame_file in enumerate(frame_files):
                frame_path = frames_path / frame_file
                if frame_path.exists():
                    clip = ImageClip(str(frame_path), duration=frame_duration)
                    clips.append(clip)
                    logger.info(f"Added frame {frame_file} with duration {frame_duration:.1f}s")
            
            if not clips:
                # Fallback: create a simple colored video - Portrait format
                clips = [ColorClip(size=(1080, 1920), color=(20, 25, 40), duration=duration)]
                logger.warning("No frame files found, using color clip")
            
            # Concatenate video clips
            video = concatenate_videoclips(clips)
            
            # Set audio
            final_video = video.set_audio(audio)
            
            # Write final video
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Cleanup
            audio.close()
            video.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video assembly failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            try:
                shutil.rmtree(str(self.temp_dir))
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}") 