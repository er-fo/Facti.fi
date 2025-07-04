"""
Text-to-Speech (TTS) Generation System

Advanced TTS pipeline for creating AI host voice-overs from TruthScore analysis content.
Implements Phase 3 requirements for high-quality narration generation.
"""

import os
import logging
import tempfile
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import asyncio
from pathlib import Path

import openai
import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf

from content_classifier import ContentSegment, VideoContent

logger = logging.getLogger(__name__)

@dataclass
class TTSSegment:
    """Represents a generated TTS audio segment"""
    text: str
    audio_file: str
    duration: float
    speaker_voice: str
    start_time: float
    end_time: float
    processing_metadata: Dict[str, Any]

@dataclass
class NarrationScript:
    """Complete narration script with timing and voice assignments"""
    segments: List[TTSSegment]
    total_duration: float
    voice_profile: Dict[str, Any]
    script_metadata: Dict[str, Any]

class TTSGenerator:
    """
    Advanced TTS generation system with voice matching and audio optimization
    
    Features:
    - OpenAI TTS integration with voice selection
    - Dynamic script generation from analysis
    - Audio quality optimization and normalization
    - Speaker voice matching where possible
    - Emotional tone adaptation
    - Audio caching for efficiency
    """
    
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for TTS generation")
        
        # Initialize OpenAI client with the API key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        self.tts_config = {
            'default_voice': 'alloy',  # OpenAI TTS voice
            'voice_options': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
            'voice_characteristics': {
                'alloy': {'tone': 'balanced', 'gender': 'neutral', 'energy': 'medium'},
                'echo': {'tone': 'authoritative', 'gender': 'male', 'energy': 'high'},
                'fable': {'tone': 'warm', 'gender': 'male', 'energy': 'low'},
                'onyx': {'tone': 'deep', 'gender': 'male', 'energy': 'medium'},
                'nova': {'tone': 'professional', 'gender': 'female', 'energy': 'high'},
                'shimmer': {'tone': 'friendly', 'gender': 'female', 'energy': 'medium'}
            },
            'audio_settings': {
                'sample_rate': 44100,
                'bit_depth': 16,
                'format': 'mp3',
                'speed': 1.0,
                'volume_normalization': True
            },
            'script_templates': {
                'intro': "In this analysis, we examine {topic} with a credibility score of {score} out of 100.",
                'key_claim': "A key claim states: {claim_text}",
                'evidence': "Research shows: {evidence_text}",
                'verification': "Fact-checking reveals this claim is {status}.",
                'conclusion': "In summary, this content has {credibility_level} credibility based on our analysis.",
                'transition': "Let's examine the next point."
            }
        }
        
        # Audio processing cache
        self.audio_cache = {}
        self.cache_dir = Path(tempfile.gettempdir()) / "tts_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    async def generate_narration(self, video_content: VideoContent, 
                               clip_config: Dict[str, Any]) -> NarrationScript:
        """
        Generate complete narration script from classified video content
        
        Args:
            video_content: Classified content from ContentClassifier
            clip_config: Video generation configuration
            
        Returns:
            NarrationScript with all TTS segments and metadata
        """
        logger.info("Starting TTS narration generation")
        
        try:
            # Step 1: Generate script text from content
            script_segments = self._generate_script_segments(video_content, clip_config)
            
            # Step 2: Select optimal voice profile
            voice_profile = self._select_voice_profile(video_content, clip_config)
            
            # Step 3: Generate TTS audio for each segment
            tts_segments = await self._generate_tts_segments(script_segments, voice_profile)
            
            # Step 4: Optimize audio timing and quality
            optimized_segments = self._optimize_audio_segments(tts_segments)
            
            # Step 5: Calculate total duration and metadata
            total_duration = sum(segment.duration for segment in optimized_segments)
            
            script_metadata = {
                'generation_timestamp': 'timestamp',
                'total_segments': len(optimized_segments),
                'voice_profile': voice_profile,
                'content_type': clip_config.get('type', 'social'),
                'credibility_score': video_content.credibility_score
            }
            
            narration_script = NarrationScript(
                segments=optimized_segments,
                total_duration=total_duration,
                voice_profile=voice_profile,
                script_metadata=script_metadata
            )
            
            logger.info(f"TTS generation completed: {len(optimized_segments)} segments, "
                       f"{total_duration:.1f}s total duration")
            
            return narration_script
            
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise
    
    def _generate_script_segments(self, video_content: VideoContent, 
                                clip_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate script text segments from video content"""
        script_segments = []
        clip_type = clip_config.get('type', 'social')
        
        # Introduction segment
        intro_text = self._generate_intro_text(video_content)
        script_segments.append({
            'type': 'intro',
            'text': intro_text,
            'timing': 'start',
            'priority': 'high'
        })
        
        # Key claims segments
        key_claims = [s for s in video_content.segments if s.segment_type == 'key_claim']
        for i, claim_segment in enumerate(key_claims[:3]):  # Limit to top 3 claims
            claim_text = self._generate_claim_text(claim_segment, video_content)
            script_segments.append({
                'type': 'key_claim',
                'text': claim_text,
                'timing': 'content',
                'priority': 'high',
                'original_segment': claim_segment
            })
        
        # Evidence segments (for longer videos)
        if clip_type == 'summary':
            evidence_segments = [s for s in video_content.segments if s.segment_type == 'evidence']
            for evidence_segment in evidence_segments[:2]:
                evidence_text = self._generate_evidence_text(evidence_segment, video_content)
                script_segments.append({
                    'type': 'evidence',
                    'text': evidence_text,
                    'timing': 'content',
                    'priority': 'medium',
                    'original_segment': evidence_segment
                })
        
        # Credibility assessment
        credibility_text = self._generate_credibility_assessment(video_content)
        script_segments.append({
            'type': 'credibility',
            'text': credibility_text,
            'timing': 'content',
            'priority': 'high'
        })
        
        # Conclusion segment
        conclusion_text = self._generate_conclusion_text(video_content)
        script_segments.append({
            'type': 'conclusion',
            'text': conclusion_text,
            'timing': 'end',
            'priority': 'high'
        })
        
        return script_segments
    
    def _generate_intro_text(self, video_content: VideoContent) -> str:
        """Generate introduction narration text"""
        themes_text = ", ".join(video_content.key_themes[:2]) if video_content.key_themes else "this content"
        
        # Handle rhetorical tactics and bias indicators from enhanced analysis
        analysis_summary = video_content.analysis_summary or {}
        has_tactics = len(analysis_summary.get('rhetorical_tactics', [])) > 0
        has_bias = len(analysis_summary.get('bias_indicators', [])) > 0
        
        if has_tactics or has_bias:
            intro_options = [
                f"Let's analyze {themes_text} for rhetorical tactics and credibility issues.",
                f"In this fact-check, we'll examine {themes_text} for bias and manipulation.",
                f"Here's our analysis of {themes_text} and its persuasion techniques."
            ]
        else:
            intro_options = [
                f"Let's analyze {themes_text} and examine its credibility.",
                f"In this fact-check, we'll evaluate {themes_text}.",
                f"Here's our analysis of {themes_text} and its accuracy."
            ]
        
        # Select based on credibility score
        if video_content.credibility_score >= 70:
            return intro_options[0]
        elif video_content.credibility_score >= 40:
            return intro_options[1]
        else:
            return intro_options[2]
    
    def _generate_claim_text(self, claim_segment: ContentSegment, video_content: VideoContent) -> str:
        """Generate narration for key claims"""
        # Simplify and summarize the claim
        claim_text = claim_segment.text
        
        # Truncate if too long for narration
        if len(claim_text) > 150:
            claim_text = claim_text[:147] + "..."
        
        # Add verification context
        verification_score = claim_segment.credibility_indicators.get('claim_verification', 0)
        
        if verification_score > 0.7:
            status_text = "This claim appears to be well-supported."
        elif verification_score > 0.4:
            status_text = "This claim has mixed evidence."
        else:
            status_text = "This claim lacks strong evidence."
        
        return f"The content states: '{claim_text}' {status_text}"
    
    def _generate_evidence_text(self, evidence_segment: ContentSegment, video_content: VideoContent) -> str:
        """Generate narration for evidence segments"""
        research_backing = evidence_segment.credibility_indicators.get('research_backing', 0)
        
        if research_backing > 0.6:
            return f"Research supports this point: {evidence_segment.text[:100]}..."
        else:
            return f"The evidence for this claim is limited: {evidence_segment.text[:100]}..."
    
    def _generate_credibility_assessment(self, video_content: VideoContent) -> str:
        """Generate credibility assessment narration"""
        score = int(video_content.credibility_score)  # Ensure integer display
        analysis_summary = video_content.analysis_summary or {}
        
        # Include specific findings in the assessment
        key_concerns = analysis_summary.get('key_concerns', [])
        main_findings = analysis_summary.get('main_findings', [])
        
        base_assessment = ""
        if score >= 80:
            base_assessment = f"This content has high credibility with a score of {score} out of 100."
        elif score >= 60:
            base_assessment = f"This content has good credibility with a score of {score} out of 100."
        elif score >= 40:
            base_assessment = f"This content has moderate credibility with a score of {score} out of 100."
        else:
            base_assessment = f"This content has low credibility with a score of {score} out of 100."
        
        # Add specific concerns if present
        if key_concerns:
            concern_text = key_concerns[0] if len(key_concerns) == 1 else f"{len(key_concerns)} credibility issues"
            base_assessment += f" Key concerns include {concern_text}."
        
        return base_assessment
    
    def _generate_conclusion_text(self, video_content: VideoContent) -> str:
        """Generate conclusion narration text"""
        if video_content.content_warnings:
            return "In conclusion, verify this information independently before sharing."
        elif video_content.credibility_score >= 70:
            return "In conclusion, this content appears to be largely accurate and well-supported."
        else:
            return "In conclusion, this content should be approached with caution."
    
    def _select_voice_profile(self, video_content: VideoContent, clip_config: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal voice profile based on content characteristics"""
        # Default voice selection based on content credibility
        if video_content.credibility_score >= 70:
            # High credibility - authoritative voice
            selected_voice = 'echo'
        elif video_content.credibility_score >= 40:
            # Medium credibility - balanced voice
            selected_voice = 'alloy'
        else:
            # Low credibility - careful/cautious voice
            selected_voice = 'nova'
        
        # Adjust for clip type
        if clip_config.get('type') == 'social':
            # Social media clips - more engaging voice
            selected_voice = 'shimmer' if selected_voice == 'nova' else selected_voice
        
        voice_profile = {
            'voice_name': selected_voice,
            'characteristics': self.tts_config['voice_characteristics'][selected_voice],
            'speed': 1.0,
            'volume': 1.0
        }
        
        return voice_profile
    
    async def _generate_tts_segments(self, script_segments: List[Dict[str, Any]], 
                                   voice_profile: Dict[str, Any]) -> List[TTSSegment]:
        """Generate TTS audio for all script segments"""
        tts_segments = []
        current_time = 0.0
        
        for segment_data in script_segments:
            text = segment_data['text']
            
            # Check cache first
            cache_key = self._get_cache_key(text, voice_profile['voice_name'])
            cached_audio = self._get_cached_audio(cache_key)
            
            if cached_audio:
                audio_file = cached_audio
                logger.debug(f"Using cached TTS for: {text[:50]}...")
            else:
                # Generate new TTS
                audio_file = await self._generate_single_tts(text, voice_profile)
                self._cache_audio(cache_key, audio_file)
            
            # Get audio duration
            audio_duration = self._get_audio_duration(audio_file)
            
            tts_segment = TTSSegment(
                text=text,
                audio_file=audio_file,
                duration=audio_duration,
                speaker_voice=voice_profile['voice_name'],
                start_time=current_time,
                end_time=current_time + audio_duration,
                processing_metadata={
                    'segment_type': segment_data['type'],
                    'priority': segment_data['priority'],
                    'cache_used': cached_audio is not None
                }
            )
            
            tts_segments.append(tts_segment)
            current_time += audio_duration + 0.5  # Add pause between segments
        
        return tts_segments
    
    async def _generate_single_tts(self, text: str, voice_profile: Dict[str, Any]) -> str:
        """Generate single TTS audio file using OpenAI API"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1-hd",  # High quality model
                voice=voice_profile['voice_name'],
                input=text,
                speed=voice_profile.get('speed', 1.0)
            )
            
            # Save to temporary file
            audio_file = tempfile.NamedTemporaryFile(
                suffix='.mp3', 
                delete=False,
                dir=self.cache_dir
            ).name
            
            # Write response content to file
            with open(audio_file, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"Generated TTS audio: {audio_file}")
            return audio_file
            
        except Exception as e:
            logger.error(f"TTS generation failed for text: {text[:50]}... Error: {str(e)}")
            raise
    
    def _optimize_audio_segments(self, tts_segments: List[TTSSegment]) -> List[TTSSegment]:
        """Optimize audio quality and timing for all segments"""
        optimized_segments = []
        
        for segment in tts_segments:
            try:
                # Load audio
                audio = AudioSegment.from_mp3(segment.audio_file)
                
                # Normalize volume
                if self.tts_config['audio_settings']['volume_normalization']:
                    audio = audio.normalize()
                
                # Apply gentle compression for consistency
                audio = audio.compress_dynamic_range(threshold=-20.0, ratio=2.0)
                
                # Save optimized audio
                optimized_file = tempfile.NamedTemporaryFile(
                    suffix='_optimized.mp3',
                    delete=False,
                    dir=self.cache_dir
                ).name
                
                audio.export(optimized_file, format='mp3', bitrate='192k')
                
                # Update segment with optimized file
                optimized_segment = TTSSegment(
                    text=segment.text,
                    audio_file=optimized_file,
                    duration=len(audio) / 1000.0,  # Convert ms to seconds
                    speaker_voice=segment.speaker_voice,
                    start_time=segment.start_time,
                    end_time=segment.start_time + (len(audio) / 1000.0),
                    processing_metadata={
                        **segment.processing_metadata,
                        'optimized': True,
                        'original_file': segment.audio_file
                    }
                )
                
                optimized_segments.append(optimized_segment)
                
            except Exception as e:
                logger.warning(f"Audio optimization failed for segment: {str(e)}")
                # Keep original segment if optimization fails
                optimized_segments.append(segment)
        
        return optimized_segments
    
    def _get_cache_key(self, text: str, voice: str) -> str:
        """Generate cache key for TTS audio"""
        content = f"{text}_{voice}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str) -> Optional[str]:
        """Retrieve cached audio file if available"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        return str(cache_file) if cache_file.exists() else None
    
    def _cache_audio(self, cache_key: str, audio_file: str):
        """Cache generated audio file"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        try:
            # Copy audio file to cache
            audio = AudioSegment.from_mp3(audio_file)
            audio.export(str(cache_file), format='mp3')
            logger.debug(f"Cached TTS audio: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache audio: {str(e)}")
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            audio = AudioSegment.from_mp3(audio_file)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {str(e)}")
            return 5.0  # Default fallback duration
    
    def cleanup_temp_files(self, tts_segments: List[TTSSegment]):
        """Clean up temporary TTS files"""
        for segment in tts_segments:
            try:
                if os.path.exists(segment.audio_file):
                    os.unlink(segment.audio_file)
                
                # Also clean up optimized files
                optimized_file = segment.processing_metadata.get('original_file')
                if optimized_file and os.path.exists(optimized_file):
                    os.unlink(optimized_file)
                    
            except Exception as e:
                logger.warning(f"Failed to cleanup TTS file {segment.audio_file}: {str(e)}")
        
        logger.info(f"Cleaned up {len(tts_segments)} TTS files") 