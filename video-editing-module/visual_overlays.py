"""
Visual Overlay System for Video Generation

Advanced overlay rendering system that adds credibility scores, fact-check indicators,
speaker identification, and other visual elements to video frames.
Implements Phase 3 requirements for comprehensive visual feedback.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import math

from content_classifier import ContentSegment, VideoContent

logger = logging.getLogger(__name__)

@dataclass
class OverlayElement:
    """Represents a single visual overlay element"""
    element_type: str  # 'credibility_meter', 'speaker_badge', 'fact_check', 'subtitle'
    position: Tuple[int, int]
    size: Tuple[int, int]
    start_time: float
    end_time: float
    content: Dict[str, Any]
    animation: Optional[Dict[str, Any]] = None

@dataclass
class OverlayFrame:
    """Represents all overlays for a single video frame"""
    frame_time: float
    elements: List[OverlayElement]
    frame_number: int

class VisualOverlayRenderer:
    """
    Advanced visual overlay rendering system
    
    Features:
    - Credibility score meters with animated progress
    - Speaker identification badges with smooth transitions
    - Fact-check verification symbols with color coding
    - Dynamic subtitles with highlighting
    - Glass morphism design aesthetic
    - Smooth animations and transitions
    - Multi-layer composition
    """
    
    def __init__(self):
        self.color_scheme = {
            'primary_bg': (15, 15, 15),      # Dark background
            'glass_bg': (255, 255, 255, 13), # Semi-transparent white (RGBA)
            'accent_cyan': (255, 255, 0),     # Cyan accent
            'accent_purple': (246, 92, 139),  # Purple accent
            'success_green': (16, 185, 129),  # Success green
            'warning_yellow': (245, 158, 11), # Warning yellow
            'error_red': (239, 68, 68),       # Error red
            'text_primary': (255, 255, 255),  # White text
            'text_secondary': (160, 160, 160) # Gray text
        }
        
        self.overlay_config = {
            'credibility_meter': {
                'position_type': 'bottom_right',
                'offset': (50, 50),
                'size': (200, 80),
                'animation_duration': 0.8,
                'show_duration': 3.0
            },
            'speaker_badge': {
                'position_type': 'top_left',
                'offset': (50, 50),
                'size': (250, 60),
                'fade_duration': 0.3,
                'show_duration': 2.0
            },
            'fact_check_indicator': {
                'position_type': 'top_right',
                'offset': (50, 50),
                'size': (60, 60),
                'pulse_duration': 1.0,
                'show_duration': 2.5
            },
            'subtitle': {
                'position_type': 'bottom_center',
                'offset': (0, 100),
                'font_size': 24,
                'font_family': cv2.FONT_HERSHEY_SIMPLEX,
                'max_width': 800,
                'show_duration': 3.0
            },
            'progress_bar': {
                'position_type': 'bottom',
                'offset': (0, 20),
                'size': (0, 4),  # Full width, 4px height
                'color': (255, 255, 0),
                'bg_color': (255, 255, 255, 20)
            }
        }
        
        self.animation_cache = {}
    
    def generate_overlay_timeline(self, video_content: VideoContent, 
                                video_duration: float, fps: int = 30) -> List[OverlayFrame]:
        """
        Generate complete overlay timeline for the video
        
        Args:
            video_content: Classified content from ContentClassifier
            video_duration: Total video duration in seconds
            fps: Frames per second
            
        Returns:
            List of OverlayFrame objects for each video frame
        """
        logger.info(f"Generating overlay timeline for {video_duration:.1f}s video at {fps} FPS")
        
        try:
            total_frames = int(video_duration * fps)
            overlay_frames = []
            
            # Generate overlay elements based on content
            overlay_elements = self._generate_overlay_elements(video_content, video_duration)
            
            # Create frame-by-frame overlay data
            for frame_num in range(total_frames):
                frame_time = frame_num / fps
                
                # Find active overlays for this frame
                active_elements = self._get_active_elements(overlay_elements, frame_time)
                
                overlay_frame = OverlayFrame(
                    frame_time=frame_time,
                    elements=active_elements,
                    frame_number=frame_num
                )
                
                overlay_frames.append(overlay_frame)
            
            logger.info(f"Generated {len(overlay_frames)} overlay frames with {len(overlay_elements)} total elements")
            return overlay_frames
            
        except Exception as e:
            logger.error(f"Overlay timeline generation failed: {str(e)}")
            raise
    
    def render_frame_overlays(self, frame: np.ndarray, overlay_frame: OverlayFrame, 
                            video_resolution: Tuple[int, int]) -> np.ndarray:
        """
        Render all overlays for a single video frame
        
        Args:
            frame: Input video frame (BGR format)
            overlay_frame: Overlay data for this frame
            video_resolution: (width, height) of video
            
        Returns:
            Frame with overlays rendered
        """
        if not overlay_frame.elements:
            return frame
        
        # Create overlay composition layers
        overlay_layer = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)  # RGBA overlay
        
        for element in overlay_frame.elements:
            try:
                if element.element_type == 'credibility_meter':
                    self._render_credibility_meter(overlay_layer, element, overlay_frame.frame_time, video_resolution)
                elif element.element_type == 'speaker_badge':
                    self._render_speaker_badge(overlay_layer, element, overlay_frame.frame_time, video_resolution)
                elif element.element_type == 'fact_check_indicator':
                    self._render_fact_check_indicator(overlay_layer, element, overlay_frame.frame_time, video_resolution)
                elif element.element_type == 'subtitle':
                    self._render_subtitle(overlay_layer, element, video_resolution)
                elif element.element_type == 'progress_bar':
                    self._render_progress_bar(overlay_layer, element, overlay_frame.frame_time, video_resolution)
                
            except Exception as e:
                logger.warning(f"Failed to render overlay element {element.element_type}: {str(e)}")
        
        # Composite overlays onto original frame
        result_frame = self._composite_overlays(frame, overlay_layer)
        
        return result_frame
    
    def _generate_overlay_elements(self, video_content: VideoContent, 
                                 video_duration: float) -> List[OverlayElement]:
        """Generate all overlay elements for the video"""
        elements = []
        
        # 1. Credibility meter (appears early and stays)
        credibility_element = OverlayElement(
            element_type='credibility_meter',
            position=self._calculate_position('bottom_right', self.overlay_config['credibility_meter']['offset'], (1920, 1080)),
            size=self.overlay_config['credibility_meter']['size'],
            start_time=2.0,  # Appear after 2 seconds
            end_time=min(video_duration - 1.0, 8.0),  # Show for up to 6 seconds
            content={
                'score': video_content.credibility_score,
                'label': 'Credibility Score',
                'animation_progress': 0.0
            },
            animation={
                'type': 'fill_progress',
                'duration': self.overlay_config['credibility_meter']['animation_duration']
            }
        )
        elements.append(credibility_element)
        
        # 2. Speaker badges for segments with identified speakers
        speaker_time = 0.0
        for segment in video_content.segments:
            if segment.speaker and segment.speaker != 'UNKNOWN':
                speaker_element = OverlayElement(
                    element_type='speaker_badge',
                    position=self._calculate_position('top_left', self.overlay_config['speaker_badge']['offset'], (1920, 1080)),
                    size=self.overlay_config['speaker_badge']['size'],
                    start_time=speaker_time,
                    end_time=min(speaker_time + self.overlay_config['speaker_badge']['show_duration'], video_duration),
                    content={
                        'speaker_name': segment.speaker,
                        'confidence': segment.credibility_indicators.get('speaker_confidence', 0.5)
                    },
                    animation={
                        'type': 'fade_in',
                        'duration': self.overlay_config['speaker_badge']['fade_duration']
                    }
                )
                elements.append(speaker_element)
                speaker_time += 4.0  # Space out speaker badges
        
        # 3. Fact-check indicators for key claims
        fact_check_time = 3.0
        key_claims = [s for s in video_content.segments if s.segment_type == 'key_claim']
        
        for i, claim_segment in enumerate(key_claims[:3]):  # Limit to 3 indicators
            verification_score = claim_segment.credibility_indicators.get('claim_verification', 0.5)
            
            if verification_score > 0.7:
                status = 'verified'
            elif verification_score > 0.4:
                status = 'partial'
            else:
                status = 'disputed'
            
            fact_check_element = OverlayElement(
                element_type='fact_check_indicator',
                position=self._calculate_position('top_right', 
                                                (self.overlay_config['fact_check_indicator']['offset'][0], 
                                                 self.overlay_config['fact_check_indicator']['offset'][1] + i * 70), 
                                                (1920, 1080)),
                size=self.overlay_config['fact_check_indicator']['size'],
                start_time=fact_check_time + i * 1.0,
                end_time=fact_check_time + i * 1.0 + self.overlay_config['fact_check_indicator']['show_duration'],
                content={
                    'status': status,
                    'confidence': verification_score,
                    'claim_preview': claim_segment.text[:50] + "..." if len(claim_segment.text) > 50 else claim_segment.text
                },
                animation={
                    'type': 'pulse',
                    'duration': self.overlay_config['fact_check_indicator']['pulse_duration']
                }
            )
            elements.append(fact_check_element)
        
        # 4. Progress bar (throughout video)
        progress_element = OverlayElement(
            element_type='progress_bar',
            position=self._calculate_position('bottom', self.overlay_config['progress_bar']['offset'], (1920, 1080)),
            size=(1920, self.overlay_config['progress_bar']['size'][1]),  # Full width
            start_time=0.0,
            end_time=video_duration,
            content={
                'total_duration': video_duration
            }
        )
        elements.append(progress_element)
        
        # 5. Key subtitle for important claims
        subtitle_time = 5.0
        for claim_segment in key_claims[:2]:  # Show subtitles for top 2 claims
            subtitle_text = self._generate_subtitle_text(claim_segment)
            
            subtitle_element = OverlayElement(
                element_type='subtitle',
                position=self._calculate_position('bottom_center', self.overlay_config['subtitle']['offset'], (1920, 1080)),
                size=(self.overlay_config['subtitle']['max_width'], 100),
                start_time=subtitle_time,
                end_time=subtitle_time + self.overlay_config['subtitle']['show_duration'],
                content={
                    'text': subtitle_text,
                    'font_size': self.overlay_config['subtitle']['font_size']
                }
            )
            elements.append(subtitle_element)
            subtitle_time += 6.0
        
        return elements
    
    def _render_credibility_meter(self, overlay_layer: np.ndarray, element: OverlayElement, 
                                frame_time: float, video_resolution: Tuple[int, int]):
        """Render animated credibility score meter"""
        x, y = element.position
        w, h = element.size
        score = element.content['score']
        
        # Calculate animation progress
        time_in_element = frame_time - element.start_time
        animation_progress = min(1.0, time_in_element / element.animation['duration']) if element.animation else 1.0
        
        # Glass background
        self._draw_glass_rectangle(overlay_layer, (x, y, w, h), self.color_scheme['glass_bg'])
        
        # Score bar background
        bar_y = y + h // 2 - 10
        bar_h = 20
        bar_margin = 20
        bar_w = w - 2 * bar_margin
        
        self._draw_rectangle(overlay_layer, (x + bar_margin, bar_y, bar_w, bar_h), 
                           (*self.color_scheme['text_secondary'], 30))
        
        # Animated score bar fill
        fill_width = int(bar_w * (score / 100) * animation_progress)
        bar_color = self._get_score_color(score)
        
        if fill_width > 0:
            self._draw_rectangle(overlay_layer, (x + bar_margin, bar_y, fill_width, bar_h), bar_color)
        
        # Score text
        score_text = f"Credibility: {int(score * animation_progress)}%"
        self._draw_text(overlay_layer, score_text, (x + 10, y + 25), 
                       self.color_scheme['text_primary'], font_scale=0.6)
    
    def _render_speaker_badge(self, overlay_layer: np.ndarray, element: OverlayElement, 
                            frame_time: float, video_resolution: Tuple[int, int]):
        """Render speaker identification badge"""
        x, y = element.position
        w, h = element.size
        
        # Calculate fade animation
        time_in_element = frame_time - element.start_time
        fade_progress = min(1.0, time_in_element / element.animation['duration']) if element.animation else 1.0
        alpha = int(255 * fade_progress * 0.9)  # Max 90% opacity
        
        # Glass background with fade
        glass_color = (*self.color_scheme['glass_bg'][:3], alpha // 5)
        self._draw_glass_rectangle(overlay_layer, (x, y, w, h), glass_color)
        
        # Speaker name
        speaker_name = element.content['speaker_name']
        if speaker_name.startswith('SPEAKER_'):
            display_name = f"Speaker {speaker_name.split('_')[1]}"
        else:
            display_name = speaker_name
        
        text_color = (*self.color_scheme['accent_cyan'], alpha)
        self._draw_text(overlay_layer, display_name, (x + 15, y + 30), text_color, font_scale=0.7)
        
        # Confidence indicator
        confidence = element.content['confidence'] * 100
        confidence_text = f"({confidence:.0f}% confidence)"
        confidence_color = (*self.color_scheme['text_secondary'], alpha)
        self._draw_text(overlay_layer, confidence_text, (x + 15, y + 50), confidence_color, font_scale=0.4)
    
    def _render_fact_check_indicator(self, overlay_layer: np.ndarray, element: OverlayElement, 
                                   frame_time: float, video_resolution: Tuple[int, int]):
        """Render fact-check verification indicator with pulse animation"""
        x, y = element.position
        size = element.size[0]  # Circular indicator
        status = element.content['status']
        
        # Calculate pulse animation
        time_in_element = frame_time - element.start_time
        pulse_cycle = (time_in_element / element.animation['duration']) % 1.0 if element.animation else 0
        pulse_scale = 1.0 + 0.1 * math.sin(pulse_cycle * 2 * math.pi)
        
        # Status colors and symbols
        status_config = {
            'verified': {'color': self.color_scheme['success_green'], 'symbol': '✓'},
            'partial': {'color': self.color_scheme['warning_yellow'], 'symbol': '?'},
            'disputed': {'color': self.color_scheme['error_red'], 'symbol': '✗'}
        }
        
        config = status_config.get(status, status_config['partial'])
        
        # Draw pulsing circle
        center = (x + size // 2, y + size // 2)
        radius = int((size // 2) * pulse_scale)
        
        self._draw_circle(overlay_layer, center, radius, config['color'])
        
        # Draw symbol
        symbol_color = self.color_scheme['text_primary']
        self._draw_text(overlay_layer, config['symbol'], (x + size//2 - 10, y + size//2 + 5), 
                       symbol_color, font_scale=1.0)
    
    def _render_subtitle(self, overlay_layer: np.ndarray, element: OverlayElement, 
                       video_resolution: Tuple[int, int]):
        """Render subtitle text with background"""
        x, y = element.position
        max_width = element.size[0]
        text = element.content['text']
        font_size = element.content['font_size']
        
        # Text wrapping for long subtitles
        wrapped_lines = self._wrap_text(text, max_width, font_size)
        
        # Calculate background size
        line_height = int(font_size * 1.5)
        bg_height = len(wrapped_lines) * line_height + 20
        bg_width = min(max_width, max(len(line) * font_size // 2 for line in wrapped_lines) + 40)
        
        # Center background
        bg_x = x - bg_width // 2
        bg_y = y - bg_height // 2
        
        # Draw background
        self._draw_glass_rectangle(overlay_layer, (bg_x, bg_y, bg_width, bg_height), 
                                 (*self.color_scheme['primary_bg'], 180))
        
        # Draw text lines
        for i, line in enumerate(wrapped_lines):
            line_y = bg_y + 15 + i * line_height
            line_x = x - len(line) * font_size // 4  # Center text
            self._draw_text(overlay_layer, line, (line_x, line_y), 
                           self.color_scheme['text_primary'], font_scale=font_size/30)
    
    def _render_progress_bar(self, overlay_layer: np.ndarray, element: OverlayElement, 
                           frame_time: float, video_resolution: Tuple[int, int]):
        """Render video progress bar"""
        x, y = element.position
        w, h = element.size
        total_duration = element.content['total_duration']
        
        # Calculate progress
        progress = min(1.0, frame_time / total_duration) if total_duration > 0 else 0.0
        
        # Background bar
        bg_color = self.color_scheme['glass_bg']
        self._draw_rectangle(overlay_layer, (x, y, w, h), bg_color)
        
        # Progress fill
        fill_width = int(w * progress)
        if fill_width > 0:
            self._draw_rectangle(overlay_layer, (x, y, fill_width, h), 
                               self.overlay_config['progress_bar']['color'])
    
    # Helper methods for drawing
    def _draw_glass_rectangle(self, overlay_layer: np.ndarray, rect: Tuple[int, int, int, int], 
                            color: Tuple[int, int, int, int]):
        """Draw glass morphism rectangle with blur effect"""
        x, y, w, h = rect
        
        # Ensure coordinates are within bounds
        h_max, w_max = overlay_layer.shape[:2]
        x = max(0, min(x, w_max - 1))
        y = max(0, min(y, h_max - 1))
        w = max(1, min(w, w_max - x))
        h = max(1, min(h, h_max - y))
        
        # Draw rectangle with alpha blending
        if len(color) == 4:  # RGBA
            overlay_layer[y:y+h, x:x+w, :3] = color[:3]
            overlay_layer[y:y+h, x:x+w, 3] = color[3]
    
    def _draw_rectangle(self, overlay_layer: np.ndarray, rect: Tuple[int, int, int, int], 
                       color: Tuple[int, int, int, int]):
        """Draw solid rectangle"""
        x, y, w, h = rect
        
        # Ensure coordinates are within bounds
        h_max, w_max = overlay_layer.shape[:2]
        x = max(0, min(x, w_max - 1))
        y = max(0, min(y, h_max - 1))
        w = max(1, min(w, w_max - x))
        h = max(1, min(h, h_max - y))
        
        if len(color) == 3:  # RGB
            overlay_layer[y:y+h, x:x+w, :3] = color
            overlay_layer[y:y+h, x:x+w, 3] = 255
        else:  # RGBA
            overlay_layer[y:y+h, x:x+w] = color
    
    def _draw_circle(self, overlay_layer: np.ndarray, center: Tuple[int, int], 
                    radius: int, color: Tuple[int, int, int]):
        """Draw filled circle"""
        # Convert overlay layer to 3-channel for OpenCV operations
        temp_img = overlay_layer[:, :, :3].copy()
        cv2.circle(temp_img, center, radius, color, -1)
        
        # Create alpha mask for the circle
        mask = np.zeros(overlay_layer.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply the circle to overlay layer
        overlay_layer[:, :, :3][mask > 0] = temp_img[mask > 0]
        overlay_layer[:, :, 3][mask > 0] = 255
    
    def _draw_text(self, overlay_layer: np.ndarray, text: str, position: Tuple[int, int], 
                  color: Tuple[int, int, int], font_scale: float = 0.6):
        """Draw text with outline for better visibility"""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(font_scale * 2))
        
        # Convert overlay layer to 3-channel for text drawing
        temp_img = overlay_layer[:, :, :3].copy()
        
        # Draw text outline (black)
        cv2.putText(temp_img, text, (x+1, y+1), font, font_scale, (0, 0, 0), thickness+1)
        # Draw main text
        cv2.putText(temp_img, text, (x, y), font, font_scale, color, thickness)
        
        # Get text bounding box for alpha mask
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Apply text to overlay layer
        overlay_layer[y-text_h:y+baseline, x:x+text_w, :3] = temp_img[y-text_h:y+baseline, x:x+text_w]
        overlay_layer[y-text_h:y+baseline, x:x+text_w, 3] = 255
    
    def _calculate_position(self, position_type: str, offset: Tuple[int, int], 
                          video_resolution: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate absolute position from position type and offset"""
        width, height = video_resolution
        offset_x, offset_y = offset
        
        positions = {
            'top_left': (offset_x, offset_y),
            'top_right': (width - offset_x, offset_y),
            'top_center': (width // 2 - offset_x, offset_y),
            'bottom_left': (offset_x, height - offset_y),
            'bottom_right': (width - offset_x, height - offset_y),
            'bottom_center': (width // 2 - offset_x, height - offset_y),
            'center': (width // 2 - offset_x, height // 2 - offset_y),
            'bottom': (offset_x, height - offset_y)
        }
        
        return positions.get(position_type, (offset_x, offset_y))
    
    def _get_active_elements(self, all_elements: List[OverlayElement], 
                           frame_time: float) -> List[OverlayElement]:
        """Get overlay elements active at given frame time"""
        active_elements = []
        
        for element in all_elements:
            if element.start_time <= frame_time <= element.end_time:
                active_elements.append(element)
        
        return active_elements
    
    def _get_score_color(self, score: float) -> Tuple[int, int, int, int]:
        """Get color based on credibility score"""
        if score >= 70:
            return (*self.color_scheme['success_green'], 255)
        elif score >= 40:
            return (*self.color_scheme['warning_yellow'], 255)
        else:
            return (*self.color_scheme['error_red'], 255)
    
    def _generate_subtitle_text(self, claim_segment: ContentSegment) -> str:
        """Generate subtitle text from claim segment"""
        text = claim_segment.text
        
        # Truncate long text for subtitle display
        if len(text) > 100:
            text = text[:97] + "..."
        
        return f'"{text}"'
    
    def _wrap_text(self, text: str, max_width: int, font_size: int) -> List[str]:
        """Wrap text to fit within max width"""
        words = text.split()
        lines = []
        current_line = []
        
        chars_per_line = max_width // (font_size // 2)  # Rough estimate
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= chars_per_line:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines[:3]  # Limit to 3 lines
    
    def _composite_overlays(self, base_frame: np.ndarray, overlay_layer: np.ndarray) -> np.ndarray:
        """Composite overlay layer onto base frame with alpha blending"""
        # Convert base frame to RGBA
        base_rgba = cv2.cvtColor(base_frame, cv2.COLOR_BGR2BGRA)
        
        # Extract alpha channel from overlay
        alpha = overlay_layer[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        
        # Alpha blend each channel
        for c in range(3):
            base_rgba[:, :, c] = (alpha * overlay_layer[:, :, c] + 
                                alpha_inv * base_rgba[:, :, c])
        
        # Convert back to BGR
        result = cv2.cvtColor(base_rgba, cv2.COLOR_BGRA2BGR)
        return result 