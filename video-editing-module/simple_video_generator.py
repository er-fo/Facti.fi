#!/usr/bin/env python3
"""
Fixed Simplified Video Generator for Social Media

This version fixes all major issues:
1. Portrait aspect ratio (1080x1920) for social media
2. Robust frame generation with comprehensive error handling
3. Better script generation with proper content extraction
4. Multiple font fallbacks and error recovery
5. Improved visual layout for portrait format
6. Detailed logging for debugging
"""

import os
import logging
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any
import asyncio

import openai
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)

class SimpleVideoGenerator:
    """Fixed simplified video generator for social media content"""
    
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.temp_dir = Path(tempfile.gettempdir()) / "simple_video_gen"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Portrait format for social media (TikTok, Instagram Stories, etc.)
        self.video_width = 1080
        self.video_height = 1920
        
        # Color scheme
        self.bg_color = (15, 23, 42)  # Dark slate
        self.text_color = (255, 255, 255)  # White
        self.accent_color = (59, 130, 246)  # Blue
        
        logger.info(f"Initialized SimpleVideoGenerator with resolution {self.video_width}x{self.video_height}")
        
    async def generate_simple_video(self, truthscore_data: Dict[str, Any], 
                                  output_path: str) -> str:
        """Generate a simple video from TruthScore data"""
        
        logger.info("Starting simple video generation with fixed parameters")
        
        try:
            # Step 1: Extract and validate key information
            metadata = truthscore_data.get('metadata', {})
            analysis = truthscore_data.get('credibility_analysis', {})
            
            title = metadata.get('title', 'Content Analysis')
            credibility_score = analysis.get('overall_score', 50)
            
            # Better claim extraction
            claims = self._extract_claims_safely(analysis)
            summary = analysis.get('summary', '')
            
            logger.info(f"Extracted: title='{title[:50]}...', score={credibility_score}, claims={len(claims)}")
            
            # Step 2: Generate improved script
            script = self._generate_improved_script(title, credibility_score, claims, summary)
            logger.info(f"Generated script: {len(script)} characters - '{script[:100]}...'")
            
            # Step 3: Generate TTS with error handling
            tts_file = await self._generate_tts_safe(script)
            logger.info(f"Generated TTS audio successfully: {tts_file}")
            
            # Step 4: Create visual frames with comprehensive error handling
            frames_dir = self._create_visual_frames_robust(title, credibility_score, claims, summary)
            logger.info(f"Created visual frames successfully in: {frames_dir}")
            
            # Step 5: Assemble video with portrait format
            final_video = self._assemble_video_robust(tts_file, frames_dir, output_path)
            logger.info(f"Video generated successfully: {final_video}")
            
            return final_video
            
        except Exception as e:
            logger.error(f"Simple video generation failed: {e}", exc_info=True)
            raise
    
    def _extract_claims_safely(self, analysis: Dict[str, Any]) -> List[str]:
        """Safely extract claims from analysis data"""
        claims = []
        
        # Try multiple possible claim locations
        claim_sources = [
            analysis.get('claims', []),
            analysis.get('key_claims', []),
            analysis.get('main_claims', []),
            analysis.get('factual_claims', [])
        ]
        
        for source in claim_sources:
            if source and isinstance(source, list):
                for claim in source[:3]:  # Limit to 3 claims
                    if isinstance(claim, str):
                        claims.append(claim)
                    elif isinstance(claim, dict):
                        claim_text = claim.get('claim') or claim.get('text') or claim.get('statement') or str(claim)
                        claims.append(claim_text)
                break
        
        # Fallback: create placeholder claims if none found
        if not claims:
            claims = ["Analysis of content credibility", "Multiple factors considered", "Results based on available evidence"]
        
        logger.info(f"Extracted {len(claims)} claims successfully")
        return claims
    
    def _generate_improved_script(self, title: str, credibility_score: int, 
                                claims: List[str], summary: str) -> str:
        """Generate an improved script with better content structure"""
        
        # Clean and truncate title
        clean_title = title.replace("YouTube", "").replace("Video:", "").strip()
        if len(clean_title) > 80:
            clean_title = clean_title[:77] + "..."
        
        # Determine credibility assessment
        if credibility_score >= 75:
            credibility_assessment = "high credibility"
            recommendation = "This content appears to be reliable."
        elif credibility_score >= 50:
            credibility_assessment = "moderate credibility"
            recommendation = "This content has mixed reliability. Verify key claims."
        else:
            credibility_assessment = "low credibility"
            recommendation = "This content has significant concerns. Cross-check information."
        
        # Build script with clear structure
        script_parts = [
            f"TruthScore Analysis: {clean_title}",
            f"Credibility Score: {credibility_score} out of 100, indicating {credibility_assessment}."
        ]
        
        # Add summary if available
        if summary and len(summary) > 20:
            summary_brief = summary[:150] + "..." if len(summary) > 150 else summary
            script_parts.append(f"Summary: {summary_brief}")
        
        # Add key claims
        if claims:
            script_parts.append("Key findings include:")
            for i, claim in enumerate(claims[:2]):  # Limit to 2 claims for timing
                claim_brief = claim[:120] + "..." if len(claim) > 120 else claim
                script_parts.append(f"{claim_brief}")
        
        script_parts.append(recommendation)
        script_parts.append("Always verify information from multiple sources.")
        
        full_script = " ".join(script_parts)
        logger.info(f"Generated script with {len(script_parts)} parts")
        return full_script
    
    async def _generate_tts_safe(self, script: str) -> str:
        """Generate TTS audio with comprehensive error handling"""
        
        try:
            # Ensure script isn't too long (TTS limit is ~4096 chars)
            if len(script) > 4000:
                script = script[:3997] + "..."
                logger.warning(f"Script truncated to {len(script)} characters")
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",  # Clear, professional voice
                input=script,
                speed=0.9  # Slightly slower for better comprehension
            )
            
            tts_file = str(self.temp_dir / "narration.mp3")
            
            with open(tts_file, 'wb') as f:
                f.write(response.content)
            
            # Verify file was created
            if not os.path.exists(tts_file) or os.path.getsize(tts_file) == 0:
                raise RuntimeError("TTS file was not created or is empty")
            
            logger.info(f"TTS generated successfully: {os.path.getsize(tts_file)} bytes")
            return tts_file
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    def _create_visual_frames_robust(self, title: str, credibility_score: int, 
                                   claims: List[str], summary: str) -> str:
        """Create visual frames with comprehensive error handling and portrait layout"""
        
        frames_dir = self.temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            # Frame 1: Title and branding
            self._create_title_frame_portrait(
                str(frames_dir / "title.png"),
                title,
                credibility_score
            )
            
            # Frame 2: Credibility score with visual indicator
            self._create_score_frame_portrait(
                str(frames_dir / "score.png"),
                credibility_score
            )
            
            # Frame 3: Key claims or summary
            self._create_content_frame_portrait(
                str(frames_dir / "content.png"),
                claims,
                summary
            )
            
            # Verify all frames were created
            required_frames = ["title.png", "score.png", "content.png"]
            for frame_file in required_frames:
                frame_path = frames_dir / frame_file
                if not frame_path.exists() or frame_path.stat().st_size == 0:
                    logger.error(f"Frame {frame_file} was not created or is empty")
                    raise RuntimeError(f"Failed to create frame: {frame_file}")
            
            logger.info(f"All {len(required_frames)} frames created successfully")
            return str(frames_dir)
            
        except Exception as e:
            logger.error(f"Frame creation failed: {e}", exc_info=True)
            raise
    
    def _get_font_robust(self, size: int) -> ImageFont.ImageFont:
        """Get font with multiple fallbacks for cross-platform compatibility"""
        
        font_paths = [
            # macOS fonts
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SF-Pro-Display-Regular.otf",
            # Linux fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            # Windows fonts (if running on Windows or WSL)
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf"
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, size)
                    logger.debug(f"Using font: {font_path} size {size}")
                    return font
            except Exception as e:
                logger.debug(f"Failed to load font {font_path}: {e}")
                continue
        
        # Final fallback to default font
        logger.warning(f"All system fonts failed, using default font for size {size}")
        return ImageFont.load_default()
    
    def _create_title_frame_portrait(self, output_path: str, title: str, score: int):
        """Create title frame optimized for portrait format"""
        
        img = Image.new('RGB', (self.video_width, self.video_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        try:
            # Clean title
            clean_title = title.replace("YouTube", "").replace("Video:", "").strip()
            if len(clean_title) > 60:
                clean_title = clean_title[:57] + "..."
            
            # Fonts
            title_font = self._get_font_robust(64)
            brand_font = self._get_font_robust(48)
            score_font = self._get_font_robust(36)
            
            # Layout for portrait format
            y_pos = 200
            
            # TruthScore branding at top
            brand_text = "TruthScore Analysis"
            brand_bbox = draw.textbbox((0, 0), brand_text, font=brand_font)
            brand_width = brand_bbox[2] - brand_bbox[0]
            brand_x = (self.video_width - brand_width) // 2
            draw.text((brand_x, y_pos), brand_text, fill=self.accent_color, font=brand_font)
            
            y_pos += 120
            
            # Title (wrapped if necessary)
            title_lines = self._wrap_text(clean_title, title_font, self.video_width - 80)
            for line in title_lines:
                line_bbox = draw.textbbox((0, 0), line, font=title_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (self.video_width - line_width) // 2
                draw.text((line_x, y_pos), line, fill=self.text_color, font=title_font)
                y_pos += 80
            
            y_pos += 60
            
            # Score preview
            score_text = f"Score: {score}/100"
            score_bbox = draw.textbbox((0, 0), score_text, font=score_font)
            score_width = score_bbox[2] - score_bbox[0]
            score_x = (self.video_width - score_width) // 2
            
            score_color = self._get_credibility_color(score)
            draw.text((score_x, y_pos), score_text, fill=score_color, font=score_font)
            
            img.save(output_path)
            logger.info(f"Title frame created successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Title frame creation failed: {e}")
            raise
    
    def _create_score_frame_portrait(self, output_path: str, score: int):
        """Create credibility score frame with visual elements for portrait format"""
        
        img = Image.new('RGB', (self.video_width, self.video_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        try:
            # Fonts
            score_font = self._get_font_robust(120)
            label_font = self._get_font_robust(48)
            desc_font = self._get_font_robust(36)
            
            # Score color
            score_color = self._get_credibility_color(score)
            
            # Center the score
            score_text = f"{score}"
            score_bbox = draw.textbbox((0, 0), score_text, font=score_font)
            score_width = score_bbox[2] - score_bbox[0]
            score_height = score_bbox[3] - score_bbox[1]
            
            score_x = (self.video_width - score_width) // 2
            score_y = (self.video_height // 2) - 100
            
            # Draw large score
            draw.text((score_x, score_y), score_text, fill=score_color, font=score_font)
            
            # Draw "/100" next to score
            hundred_text = "/100"
            hundred_font = self._get_font_robust(60)
            hundred_x = score_x + score_width + 20
            hundred_y = score_y + 40
            draw.text((hundred_x, hundred_y), hundred_text, fill=self.text_color, font=hundred_font)
            
            # Label below
            label_text = "Credibility Score"
            label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
            label_x = (self.video_width - label_width) // 2
            label_y = score_y + score_height + 40
            draw.text((label_x, label_y), label_text, fill=self.text_color, font=label_font)
            
            # Description
            if score >= 75:
                desc_text = "High Credibility"
                desc_color = (76, 175, 80)  # Green
            elif score >= 50:
                desc_text = "Moderate Credibility"  
                desc_color = (255, 193, 7)  # Yellow
            else:
                desc_text = "Low Credibility"
                desc_color = (244, 67, 54)  # Red
            
            desc_bbox = draw.textbbox((0, 0), desc_text, font=desc_font)
            desc_width = desc_bbox[2] - desc_bbox[0]
            desc_x = (self.video_width - desc_width) // 2
            desc_y = label_y + 80
            draw.text((desc_x, desc_y), desc_text, fill=desc_color, font=desc_font)
            
            # Add visual progress bar
            bar_width = 400
            bar_height = 20
            bar_x = (self.video_width - bar_width) // 2
            bar_y = desc_y + 80
            
            # Background bar
            draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                         fill=(50, 50, 50), outline=(100, 100, 100))
            
            # Progress fill
            fill_width = int((score / 100) * bar_width)
            if fill_width > 0:
                draw.rectangle([bar_x, bar_y, bar_x + fill_width, bar_y + bar_height], 
                             fill=score_color)
            
            img.save(output_path)
            logger.info(f"Score frame created successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Score frame creation failed: {e}")
            raise
    
    def _create_content_frame_portrait(self, output_path: str, claims: List[str], summary: str):
        """Create content frame with claims or summary for portrait format"""
        
        img = Image.new('RGB', (self.video_width, self.video_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        try:
            # Fonts
            header_font = self._get_font_robust(48)
            content_font = self._get_font_robust(32)
            
            y_pos = 200
            
            # Header
            header_text = "Key Findings"
            header_bbox = draw.textbbox((0, 0), header_text, font=header_font)
            header_width = header_bbox[2] - header_bbox[0]
            header_x = (self.video_width - header_width) // 2
            draw.text((header_x, y_pos), header_text, fill=self.accent_color, font=header_font)
            
            y_pos += 100
            
            # Content
            content_lines = []
            
            if claims and len(claims) > 0:
                for i, claim in enumerate(claims[:3]):  # Max 3 claims
                    if len(claim) > 80:
                        claim = claim[:77] + "..."
                    content_lines.append(f"• {claim}")
            elif summary:
                # Use summary if no claims
                summary_brief = summary[:200] + "..." if len(summary) > 200 else summary
                wrapped_summary = self._wrap_text(summary_brief, content_font, self.video_width - 80)
                content_lines.extend(wrapped_summary)
            else:
                # Fallback content
                content_lines = [
                    "• Content analysis completed",
                    "• Multiple factors evaluated", 
                    "• Results based on available data"
                ]
            
            # Draw content with proper spacing
            for line in content_lines:
                if y_pos > self.video_height - 200:  # Don't overflow
                    break
                    
                line_bbox = draw.textbbox((0, 0), line, font=content_font)
                line_height = line_bbox[3] - line_bbox[1]
                
                # Center if not bullet point, otherwise left align
                if line.startswith("•"):
                    line_x = 80
                else:
                    line_width = line_bbox[2] - line_bbox[0]
                    line_x = (self.video_width - line_width) // 2
                
                draw.text((line_x, y_pos), line, fill=self.text_color, font=content_font)
                y_pos += line_height + 20
            
            img.save(output_path)
            logger.info(f"Content frame created successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Content frame creation failed: {e}")
            raise
    
    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        """Wrap text to fit within specified width"""
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _get_credibility_color(self, score: int) -> tuple:
        """Get color based on credibility score"""
        if score >= 75:
            return (76, 175, 80)  # Green
        elif score >= 50:
            return (255, 193, 7)  # Yellow
        else:
            return (244, 67, 54)  # Red
    
    def _assemble_video_robust(self, tts_file: str, frames_dir: str, output_path: str) -> str:
        """Assemble the final video with comprehensive error handling"""
        
        try:
            logger.info("Starting video assembly")
            
            # Load and validate audio
            if not os.path.exists(tts_file):
                raise FileNotFoundError(f"TTS file not found: {tts_file}")
            
            audio = AudioFileClip(tts_file)
            duration = audio.duration
            logger.info(f"Audio duration: {duration:.1f} seconds")
            
            if duration <= 0:
                raise ValueError("Audio duration is invalid")
            
            # Validate frames
            frames_path = Path(frames_dir)
            required_frames = ["title.png", "score.png", "content.png"]
            
            clips = []
            frame_duration = duration / len(required_frames)  # Distribute time evenly
            
            for frame_file in required_frames:
                frame_path = frames_path / frame_file
                if not frame_path.exists():
                    logger.error(f"Missing frame: {frame_path}")
                    raise FileNotFoundError(f"Frame not found: {frame_path}")
                
                # Create clip with proper duration
                clip = ImageClip(str(frame_path), duration=frame_duration)
                clips.append(clip)
                logger.info(f"Added frame {frame_file} with duration {frame_duration:.1f}s")
            
            if not clips:
                raise RuntimeError("No video clips created")
            
            # Concatenate video clips
            video = concatenate_videoclips(clips)
            
            # Set audio
            final_video = video.set_audio(audio)
            
            # Write final video with optimized settings
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                bitrate='3000k',  # Good quality for social media
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Cleanup clips
            audio.close()
            video.close()
            final_video.close()
            
            # Verify output
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file was not created: {output_path}")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"Video assembly completed: {output_path} ({file_size:.1f}MB)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video assembly failed: {e}", exc_info=True)
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