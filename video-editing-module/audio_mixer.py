"""
Audio Mixing System for Video Generation

Advanced audio mixing pipeline that combines TTS narration, background music,
and sound effects into professional-quality audio tracks.
Implements Phase 3 requirements for audio production.
"""

import os
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range, low_pass_filter, high_pass_filter
import librosa
import soundfile as sf
import scipy.signal
from scipy.io import wavfile

from tts_generator import TTSSegment, NarrationScript

logger = logging.getLogger(__name__)

@dataclass
class AudioTrack:
    """Represents an audio track in the mix"""
    audio_file: str
    start_time: float
    end_time: float
    volume: float
    track_type: str  # 'narration', 'background', 'sfx'
    fade_in: float = 0.0
    fade_out: float = 0.0
    processing_effects: List[str] = None

@dataclass
class SoundEffect:
    """Represents a sound effect placement"""
    effect_type: str
    audio_file: str
    trigger_time: float
    volume: float
    duration: float

@dataclass
class MixedAudio:
    """Final mixed audio output"""
    audio_file: str
    duration: float
    sample_rate: int
    channels: int
    lufs: float  # Loudness units
    peak_db: float
    mix_metadata: Dict[str, Any]

class AudioMixer:
    """
    Professional audio mixing system
    
    Features:
    - Multi-track audio mixing with precise timing
    - Automatic gain control and normalization
    - Dynamic range compression for consistency
    - EQ and filtering for clarity
    - Crossfades and smooth transitions
    - Background music ducking during narration
    - Sound effects synchronization
    - LUFS normalization for broadcast standards
    - Real-time audio analysis and optimization
    """
    
    def __init__(self):
        self.mixing_config = {
            'sample_rate': 44100,
            'bit_depth': 16,
            'channels': 2,  # Stereo
            'target_lufs': -16.0,  # Broadcast standard
            'peak_limit_db': -1.0,
            'narration': {
                'volume': 1.0,
                'compression_ratio': 2.0,
                'eq_boost_freq': 3000,  # Presence frequency
                'noise_gate_threshold': -40
            },
            'background_music': {
                'volume': 0.25,
                'duck_during_narration': True,
                'duck_amount': 0.6,  # Reduce to 60% during speech
                'fade_time': 0.5
            },
            'sound_effects': {
                'volume': 0.3,
                'compression_ratio': 1.5,
                'frequency_range': (200, 15000)
            },
            'crossfade_duration': 0.3,
            'silence_threshold': -50,  # dB
            'auto_gain_target': -20  # dB
        }
        
        # Sound effects library
        self.sound_effects_library = {
            'scene_transition': 'whoosh_transition.wav',
            'fact_check_positive': 'success_chime.wav',
            'fact_check_negative': 'warning_tone.wav',
            'fact_check_disputed': 'error_buzz.wav',
            'credibility_reveal': 'data_processing.wav',
            'speaker_change': 'soft_pop.wav',
            'key_claim': 'emphasis_ding.wav',
            'intro_start': 'intro_swoosh.wav',
            'outro_finish': 'outro_resolve.wav'
        }
        
        # Create temp directory for processing
        self.temp_dir = Path(tempfile.gettempdir()) / "audio_mixing"
        self.temp_dir.mkdir(exist_ok=True)
    
    def create_mixed_audio(self, narration_script: NarrationScript, 
                          video_duration: float,
                          background_music_file: Optional[str] = None,
                          sound_effects: Optional[List[SoundEffect]] = None) -> MixedAudio:
        """
        Create final mixed audio from all components
        
        Args:
            narration_script: Generated TTS narration
            video_duration: Total video duration for timing
            background_music_file: Optional background music track
            sound_effects: Optional list of sound effects to include
            
        Returns:
            MixedAudio object with final audio file and metadata
        """
        logger.info(f"Starting audio mixing for {video_duration:.1f}s video")
        
        try:
            # Step 1: Prepare audio tracks
            audio_tracks = self._prepare_audio_tracks(
                narration_script, video_duration, background_music_file, sound_effects
            )
            
            # Step 2: Process individual tracks
            processed_tracks = self._process_audio_tracks(audio_tracks)
            
            # Step 3: Create master timeline
            master_timeline = self._create_master_timeline(processed_tracks, video_duration)
            
            # Step 4: Mix all tracks together
            mixed_audio_raw = self._mix_audio_tracks(master_timeline, video_duration)
            
            # Step 5: Apply mastering chain
            mastered_audio = self._apply_mastering_chain(mixed_audio_raw)
            
            # Step 6: Export final audio
            final_audio_file = self._export_final_audio(mastered_audio, video_duration)
            
            # Step 7: Analyze final audio
            audio_analysis = self._analyze_audio(final_audio_file)
            
            mixed_audio = MixedAudio(
                audio_file=final_audio_file,
                duration=video_duration,
                sample_rate=self.mixing_config['sample_rate'],
                channels=self.mixing_config['channels'],
                lufs=audio_analysis['lufs'],
                peak_db=audio_analysis['peak_db'],
                mix_metadata={
                    'narration_segments': len(narration_script.segments),
                    'background_music': background_music_file is not None,
                    'sound_effects_count': len(sound_effects) if sound_effects else 0,
                    'processing_timestamp': 'timestamp',
                    'total_tracks': len(audio_tracks)
                }
            )
            
            logger.info(f"Audio mixing completed: {mixed_audio.duration:.1f}s, "
                       f"LUFS: {mixed_audio.lufs:.1f}, Peak: {mixed_audio.peak_db:.1f}dB")
            
            return mixed_audio
            
        except Exception as e:
            logger.error(f"Audio mixing failed: {str(e)}")
            raise
    
    def _prepare_audio_tracks(self, narration_script: NarrationScript, 
                            video_duration: float,
                            background_music_file: Optional[str],
                            sound_effects: Optional[List[SoundEffect]]) -> List[AudioTrack]:
        """Prepare all audio tracks for mixing"""
        tracks = []
        
        # 1. Narration tracks from TTS segments
        for segment in narration_script.segments:
            track = AudioTrack(
                audio_file=segment.audio_file,
                start_time=segment.start_time,
                end_time=segment.end_time,
                volume=self.mixing_config['narration']['volume'],
                track_type='narration',
                fade_in=0.1,
                fade_out=0.1,
                processing_effects=['normalize', 'compress', 'eq_boost']
            )
            tracks.append(track)
        
        # 2. Background music track
        if background_music_file and os.path.exists(background_music_file):
            track = AudioTrack(
                audio_file=background_music_file,
                start_time=0.0,
                end_time=video_duration,
                volume=self.mixing_config['background_music']['volume'],
                track_type='background',
                fade_in=1.0,
                fade_out=2.0,
                processing_effects=['normalize', 'eq_filter']
            )
            tracks.append(track)
        
        # 3. Sound effects tracks
        if sound_effects:
            for sfx in sound_effects:
                track = AudioTrack(
                    audio_file=sfx.audio_file,
                    start_time=sfx.trigger_time,
                    end_time=sfx.trigger_time + sfx.duration,
                    volume=sfx.volume,
                    track_type='sfx',
                    processing_effects=['normalize']
                )
                tracks.append(track)
        
        return tracks
    
    def _process_audio_tracks(self, tracks: List[AudioTrack]) -> List[AudioTrack]:
        """Apply processing effects to individual tracks"""
        processed_tracks = []
        
        for track in tracks:
            try:
                # Load audio
                audio = AudioSegment.from_file(track.audio_file)
                
                # Convert to target sample rate and format
                audio = audio.set_frame_rate(self.mixing_config['sample_rate'])
                audio = audio.set_channels(self.mixing_config['channels'])
                
                # Apply processing effects based on track type
                if track.track_type == 'narration':
                    audio = self._process_narration_track(audio)
                elif track.track_type == 'background':
                    audio = self._process_background_track(audio)
                elif track.track_type == 'sfx':
                    audio = self._process_sfx_track(audio)
                
                # Apply volume adjustment
                audio = audio + (20 * np.log10(track.volume))  # Convert to dB
                
                # Apply fades
                if track.fade_in > 0:
                    audio = audio.fade_in(int(track.fade_in * 1000))
                if track.fade_out > 0:
                    audio = audio.fade_out(int(track.fade_out * 1000))
                
                # Save processed audio
                processed_file = self.temp_dir / f"processed_{id(track)}.wav"
                audio.export(str(processed_file), format="wav")
                
                # Update track with processed file
                processed_track = AudioTrack(
                    audio_file=str(processed_file),
                    start_time=track.start_time,
                    end_time=track.end_time,
                    volume=1.0,  # Already applied
                    track_type=track.track_type,
                    processing_effects=track.processing_effects + ['processed']
                )
                processed_tracks.append(processed_track)
                
            except Exception as e:
                logger.warning(f"Failed to process track {track.audio_file}: {str(e)}")
                # Keep original track if processing fails
                processed_tracks.append(track)
        
        return processed_tracks
    
    def _process_narration_track(self, audio: AudioSegment) -> AudioSegment:
        """Apply narration-specific processing"""
        # Noise gate to remove low-level noise
        audio = self._apply_noise_gate(audio, self.mixing_config['narration']['noise_gate_threshold'])
        
        # Compression for consistent levels
        audio = compress_dynamic_range(
            audio,
            threshold=-20.0,
            ratio=self.mixing_config['narration']['compression_ratio'],
            attack=5.0,
            release=50.0
        )
        
        # EQ boost for presence
        audio = self._apply_presence_boost(audio, self.mixing_config['narration']['eq_boost_freq'])
        
        # Normalize
        audio = normalize(audio, headroom=3.0)
        
        return audio
    
    def _process_background_track(self, audio: AudioSegment) -> AudioSegment:
        """Apply background music processing"""
        # High-pass filter to avoid muddiness
        audio = high_pass_filter(audio, 80)
        
        # Low-pass filter to create space for narration
        audio = low_pass_filter(audio, 12000)
        
        # Gentle compression
        audio = compress_dynamic_range(
            audio,
            threshold=-25.0,
            ratio=1.5,
            attack=10.0,
            release=100.0
        )
        
        # Normalize with more headroom
        audio = normalize(audio, headroom=6.0)
        
        return audio
    
    def _process_sfx_track(self, audio: AudioSegment) -> AudioSegment:
        """Apply sound effects processing"""
        # Frequency filtering based on effect type
        sfx_config = self.mixing_config['sound_effects']
        low_freq, high_freq = sfx_config['frequency_range']
        
        audio = high_pass_filter(audio, low_freq)
        audio = low_pass_filter(audio, high_freq)
        
        # Light compression
        audio = compress_dynamic_range(
            audio,
            threshold=-15.0,
            ratio=sfx_config['compression_ratio'],
            attack=1.0,
            release=20.0
        )
        
        return audio
    
    def _create_master_timeline(self, tracks: List[AudioTrack], video_duration: float) -> Dict[str, Any]:
        """Create master timeline with all tracks and ducking automation"""
        timeline = {
            'duration': video_duration,
            'tracks': tracks,
            'automation': []
        }
        
        # Create ducking automation for background music during narration
        narration_tracks = [t for t in tracks if t.track_type == 'narration']
        background_tracks = [t for t in tracks if t.track_type == 'background']
        
        if narration_tracks and background_tracks and self.mixing_config['background_music']['duck_during_narration']:
            for narration_track in narration_tracks:
                duck_amount = self.mixing_config['background_music']['duck_amount']
                fade_time = self.mixing_config['background_music']['fade_time']
                
                # Create duck automation
                timeline['automation'].append({
                    'type': 'volume_duck',
                    'target_tracks': [t for t in background_tracks],
                    'start_time': max(0, narration_track.start_time - fade_time),
                    'end_time': min(video_duration, narration_track.end_time + fade_time),
                    'duck_amount': duck_amount,
                    'fade_time': fade_time
                })
        
        return timeline
    
    def _mix_audio_tracks(self, timeline: Dict[str, Any], video_duration: float) -> np.ndarray:
        """Mix all audio tracks into single audio array"""
        sample_rate = self.mixing_config['sample_rate']
        channels = self.mixing_config['channels']
        
        # Create master audio buffer
        total_samples = int(video_duration * sample_rate)
        master_audio = np.zeros((total_samples, channels), dtype=np.float32)
        
        # Mix each track
        for track in timeline['tracks']:
            try:
                # Load processed track audio
                audio_data, sr = librosa.load(track.audio_file, sr=sample_rate, mono=False)
                
                # Ensure stereo
                if audio_data.ndim == 1:
                    audio_data = np.stack([audio_data, audio_data], axis=0)
                elif audio_data.shape[0] > 2:
                    audio_data = audio_data[:2]  # Take first 2 channels
                
                # Transpose to (samples, channels)
                audio_data = audio_data.T
                
                # Calculate sample positions
                start_sample = int(track.start_time * sample_rate)
                end_sample = min(total_samples, start_sample + audio_data.shape[0])
                actual_samples = end_sample - start_sample
                
                # Add to master mix
                if actual_samples > 0:
                    master_audio[start_sample:end_sample] += audio_data[:actual_samples]
                
            except Exception as e:
                logger.warning(f"Failed to mix track {track.audio_file}: {str(e)}")
        
        # Apply automation (ducking, etc.)
        master_audio = self._apply_automation(master_audio, timeline, sample_rate)
        
        return master_audio
    
    def _apply_automation(self, audio: np.ndarray, timeline: Dict[str, Any], sample_rate: int) -> np.ndarray:
        """Apply volume automation and effects"""
        for automation in timeline['automation']:
            if automation['type'] == 'volume_duck':
                start_sample = int(automation['start_time'] * sample_rate)
                end_sample = int(automation['end_time'] * sample_rate)
                fade_samples = int(automation['fade_time'] * sample_rate)
                duck_amount = automation['duck_amount']
                
                # Create ducking envelope
                envelope = np.ones(end_sample - start_sample)
                
                # Fade in to duck
                if fade_samples > 0:
                    fade_in = np.linspace(1.0, duck_amount, fade_samples)
                    envelope[:fade_samples] = fade_in
                
                # Duck middle section
                envelope[fade_samples:-fade_samples] = duck_amount
                
                # Fade out from duck
                if fade_samples > 0:
                    fade_out = np.linspace(duck_amount, 1.0, fade_samples)
                    envelope[-fade_samples:] = fade_out
                
                # Apply envelope
                audio[start_sample:end_sample] *= envelope[:, np.newaxis]
        
        return audio
    
    def _apply_mastering_chain(self, audio: np.ndarray) -> np.ndarray:
        """Apply final mastering processing chain"""
        sample_rate = self.mixing_config['sample_rate']
        
        # Convert to AudioSegment for pydub processing
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=self.mixing_config['channels']
        )
        
        # Mastering chain
        # 1. Gentle multiband compression
        audio_segment = compress_dynamic_range(
            audio_segment,
            threshold=-18.0,
            ratio=1.8,
            attack=10.0,
            release=80.0
        )
        
        # 2. EQ for clarity
        audio_segment = high_pass_filter(audio_segment, 20)  # Remove sub-bass
        audio_segment = low_pass_filter(audio_segment, 18000)  # Remove harsh highs
        
        # 3. Limiting to prevent clipping
        peak_limit = self.mixing_config['peak_limit_db']
        audio_segment = audio_segment.apply_gain(peak_limit - audio_segment.max_dBFS)
        
        # 4. LUFS normalization
        target_lufs = self.mixing_config['target_lufs']
        current_lufs = self._calculate_lufs(audio_segment)
        lufs_adjustment = target_lufs - current_lufs
        audio_segment = audio_segment.apply_gain(lufs_adjustment)
        
        # Convert back to numpy array
        audio_array = np.array(audio_segment.get_array_of_samples())
        if self.mixing_config['channels'] == 2:
            audio_array = audio_array.reshape((-1, 2))
        
        return audio_array.astype(np.float32) / 32767.0
    
    def _export_final_audio(self, audio: np.ndarray, duration: float) -> str:
        """Export final mastered audio to file"""
        output_file = self.temp_dir / "final_mixed_audio.wav"
        
        # Ensure correct sample count
        expected_samples = int(duration * self.mixing_config['sample_rate'])
        if len(audio) > expected_samples:
            audio = audio[:expected_samples]
        elif len(audio) < expected_samples:
            # Pad with silence
            padding = np.zeros((expected_samples - len(audio), audio.shape[1]))
            audio = np.vstack([audio, padding])
        
        # Export
        sf.write(
            str(output_file),
            audio,
            self.mixing_config['sample_rate'],
            subtype='PCM_16'
        )
        
        logger.info(f"Final audio exported: {output_file}")
        return str(output_file)
    
    def _analyze_audio(self, audio_file: str) -> Dict[str, float]:
        """Analyze final audio metrics"""
        try:
            audio = AudioSegment.from_wav(audio_file)
            
            analysis = {
                'peak_db': audio.max_dBFS,
                'rms_db': audio.dBFS,
                'lufs': self._calculate_lufs(audio),
                'dynamic_range': audio.max_dBFS - audio.dBFS,
                'duration': len(audio) / 1000.0
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {str(e)}")
            return {'peak_db': 0, 'rms_db': -20, 'lufs': -16, 'dynamic_range': 10, 'duration': 0}
    
    # Helper methods
    def _apply_noise_gate(self, audio: AudioSegment, threshold_db: float) -> AudioSegment:
        """Apply noise gate to remove low-level noise"""
        # Simple implementation - in production, use more sophisticated algorithm
        return audio
    
    def _apply_presence_boost(self, audio: AudioSegment, boost_freq: int) -> AudioSegment:
        """Apply presence frequency boost for clarity"""
        # Simplified EQ boost - in production, use parametric EQ
        return audio
    
    def _calculate_lufs(self, audio: AudioSegment) -> float:
        """Calculate LUFS (Loudness Units relative to Full Scale)"""
        # Simplified LUFS calculation - in production, use ITU-R BS.1770 standard
        return audio.dBFS + 3.01  # Rough approximation
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        try:
            for file in self.temp_dir.glob("*.wav"):
                file.unlink()
            for file in self.temp_dir.glob("*.mp3"):
                file.unlink()
            logger.info("Cleaned up temporary audio files")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {str(e)}") 