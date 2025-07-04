"""
Content Classification System for Video Generation

This module analyzes TruthScore data to classify and prioritize content segments
for optimal video clip generation, implementing Phase 3 requirements.
"""

import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class ContentSegment:
    """Represents a classified content segment for video generation"""
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str]
    importance_score: float
    segment_type: str  # 'intro', 'key_claim', 'evidence', 'conclusion', 'filler'
    credibility_indicators: Dict[str, Any]
    claims: List[Dict[str, Any]]
    research_backing: List[Dict[str, Any]]

@dataclass
class VideoContent:
    """Complete video content structure with prioritized segments"""
    segments: List[ContentSegment]
    overall_narrative: str
    key_themes: List[str]
    credibility_score: float
    recommended_duration: float
    content_warnings: List[str]
    chain_of_thought: Optional[str] = None  # Add chain of thought analysis
    analysis_summary: Optional[Dict[str, Any]] = None  # Add detailed analysis summary

class ContentClassifier:
    """
    Advanced content classification system for intelligent video generation
    
    Features:
    - Semantic importance scoring
    - Claim verification prioritization  
    - Speaker credibility weighting
    - Narrative flow optimization
    - Content type classification
    - Chain of thought analysis extraction
    """
    
    def __init__(self):
        self.classification_config = {
            'importance_weights': {
                'credibility_score': 0.3,
                'claim_verification': 0.25,
                'speaker_confidence': 0.15,
                'semantic_density': 0.15,
                'research_backing': 0.15
            },
            'segment_types': {
                'intro': {'min_duration': 2.0, 'max_duration': 8.0, 'importance_threshold': 0.4},
                'key_claim': {'min_duration': 3.0, 'max_duration': 15.0, 'importance_threshold': 0.7},
                'evidence': {'min_duration': 2.0, 'max_duration': 10.0, 'importance_threshold': 0.6},
                'conclusion': {'min_duration': 2.0, 'max_duration': 8.0, 'importance_threshold': 0.5},
                'filler': {'min_duration': 0.5, 'max_duration': 5.0, 'importance_threshold': 0.2}
            },
            'duration_targets': {
                'social': {'min': 20, 'max': 30, 'optimal': 25},
                'summary': {'min': 90, 'max': 180, 'optimal': 120}
            }
        }
        
        # Initialize TF-IDF vectorizer for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=0.01
        )
        
    def classify_content(self, truthscore_data: Dict[str, Any], target_type: str = 'social') -> VideoContent:
        """
        Main content classification pipeline
        
        Args:
            truthscore_data: Complete analysis data from TruthScore
            target_type: 'social' for short clips, 'summary' for longer content
        
        Returns:
            VideoContent with classified and prioritized segments
        """
        logger.info(f"Starting content classification for {target_type} video")
        
        try:
            # Extract and prepare transcript data
            transcript_data = self._extract_transcript_segments(truthscore_data)
            
            # Extract credibility analysis with proper structure handling
            analysis_data = self._extract_credibility_analysis(truthscore_data)
            research_data = truthscore_data.get('web_research', {})
            
            # Extract chain of thought if available
            chain_of_thought = self._extract_chain_of_thought(truthscore_data)
            
            # Step 1: Semantic analysis and density scoring
            semantic_scores = self._calculate_semantic_density(transcript_data)
            
            # Step 2: Claim verification scoring  
            claim_scores = self._score_claim_verification(transcript_data, analysis_data)
            
            # Step 3: Speaker credibility weighting
            speaker_scores = self._calculate_speaker_credibility(transcript_data, truthscore_data.get('speakers', {}))
            
            # Step 4: Research backing analysis
            research_scores = self._analyze_research_backing(transcript_data, research_data)
            
            # Step 5: Calculate composite importance scores
            segments = self._calculate_importance_scores(
                transcript_data, semantic_scores, claim_scores, speaker_scores, research_scores
            )
            
            # Step 6: Classify segment types based on actual content analysis
            classified_segments = self._classify_segment_types_enhanced(segments, analysis_data)
            
            # Step 7: Optimize for target duration and narrative flow
            optimized_segments = self._optimize_for_target_duration(
                classified_segments, target_type
            )
            
            # Step 8: Generate overall narrative structure
            narrative = self._generate_narrative_structure(optimized_segments, analysis_data)
            
            # Step 9: Extract key themes and warnings
            themes = self._extract_key_themes_enhanced(optimized_segments, analysis_data)
            warnings = self._generate_content_warnings(analysis_data)
            
            # Extract proper credibility score (ensure it's 0-100, not inflated)
            credibility_score = self._extract_credibility_score(analysis_data)
            
            video_content = VideoContent(
                segments=optimized_segments,
                overall_narrative=narrative,
                key_themes=themes,
                credibility_score=credibility_score,
                recommended_duration=self._calculate_recommended_duration(optimized_segments),
                content_warnings=warnings,
                chain_of_thought=chain_of_thought,
                analysis_summary=self._create_analysis_summary(analysis_data)
            )
            
            logger.info(f"Content classification completed: {len(optimized_segments)} segments, "
                       f"{video_content.credibility_score}/100 credibility, "
                       f"{video_content.recommended_duration:.1f}s duration")
            
            return video_content
            
        except Exception as e:
            logger.error(f"Content classification failed: {str(e)}")
            # Create fallback content instead of raising exception
            return self._create_fallback_content(truthscore_data, target_type)
    
    def _extract_credibility_analysis(self, truthscore_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract credibility analysis from TruthScore data with proper structure handling"""
        # TruthScore exports analysis in credibility_analysis key
        analysis = truthscore_data.get('credibility_analysis', {})
        
        # Handle case where analysis might be nested in different ways
        if not analysis:
            # Fallback: check if analysis is at root level
            analysis = {
                'overall_score': truthscore_data.get('credibility_score', 50),
                'claims': truthscore_data.get('key_claims', []),
                'bias_indicators': truthscore_data.get('bias_indicators', []),
                'rhetorical_tactics': truthscore_data.get('rhetorical_tactics', [])
            }
        
        return analysis
    
    def _extract_credibility_score(self, analysis_data: Dict[str, Any]) -> float:
        """Extract credibility score ensuring it's properly bounded 0-100"""
        # Multiple possible locations for credibility score
        score_candidates = [
            analysis_data.get('overall_score'),
            analysis_data.get('credibility_score'),
            analysis_data.get('truthfulness', {}).get('overall_score') if isinstance(analysis_data.get('truthfulness'), dict) else None
        ]
        
        for score in score_candidates:
            if score is not None:
                # Handle different score formats
                if isinstance(score, (int, float)):
                    # If score is between 0-1, convert to 0-100
                    if 0 <= score <= 1:
                        return score * 100
                    # If score is already 0-100, use as-is (but cap at 100)
                    elif 0 <= score <= 100:
                        return min(score, 100)
                    # If score is > 100, it might be a calculation error, normalize it
                    elif score > 100:
                        logger.warning(f"Credibility score {score} > 100, normalizing to 0-100 range")
                        return min(score / 10, 100)  # Try dividing by 10
                
                # Handle string representations
                elif isinstance(score, str):
                    try:
                        numeric_score = float(score)
                        if 0 <= numeric_score <= 1:
                            return numeric_score * 100
                        elif 0 <= numeric_score <= 100:
                            return min(numeric_score, 100)
                        else:
                            return min(numeric_score / 10, 100)
                    except ValueError:
                        continue
        
        # Fallback to 50 if no valid score found
        logger.warning("No valid credibility score found, using fallback score of 50")
        return 50.0
    
    def _extract_chain_of_thought(self, truthscore_data: Dict[str, Any]) -> Optional[str]:
        """Extract chain of thought analysis if available"""
        # Chain of thought might be in various locations
        cot_candidates = [
            truthscore_data.get('chain_of_thought'),
            truthscore_data.get('analysis_reasoning'),
            truthscore_data.get('credibility_analysis', {}).get('reasoning'),
            truthscore_data.get('meta_analysis', {}).get('analysis_reasoning')
        ]
        
        for cot in cot_candidates:
            if cot and isinstance(cot, str) and len(cot.strip()) > 10:
                return cot.strip()
        
        # Try to reconstruct from analysis data
        analysis = truthscore_data.get('credibility_analysis', {})
        if analysis:
            reasoning_parts = []
            
            # Add credibility reasoning
            if 'credibility_reasoning' in analysis:
                reasoning_parts.append(f"**Credibility Assessment:** {analysis['credibility_reasoning']}")
            
            # Add key factors
            if 'key_factors' in analysis:
                factors = analysis['key_factors']
                if isinstance(factors, list) and factors:
                    reasoning_parts.append(f"**Key Factors:** {'; '.join(factors)}")
            
            # Add content reasoning
            if 'content_reasoning' in analysis:
                reasoning_parts.append(f"**Content Analysis:** {analysis['content_reasoning']}")
            
            if reasoning_parts:
                return "\n\n".join(reasoning_parts)
        
        return None
    
    def _create_analysis_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive analysis summary for video context"""
        summary = {
            'main_findings': [],
            'rhetorical_tactics': [],
            'bias_indicators': [],
            'evidence_quality': 'Unknown',
            'key_concerns': []
        }
        
        # Extract main findings from various analysis structures
        claims = analysis_data.get('claims', [])
        if not claims:
            claims = analysis_data.get('key_claims', [])  # Fallback
        if isinstance(claims, list):
            for claim in claims[:3]:
                if isinstance(claim, dict):
                    summary['main_findings'].append(str(claim.get('text', claim)))
                else:
                    summary['main_findings'].append(str(claim))
        
        # Extract rhetorical tactics
        if 'rhetorical_tactics' in analysis_data:
            tactics = analysis_data['rhetorical_tactics']
            if isinstance(tactics, list):
                for tactic in tactics:
                    if isinstance(tactic, dict):
                        tactic_name = tactic.get('tactic', str(tactic))
                        intensity = tactic.get('intensity_score', 0)
                        summary['rhetorical_tactics'].append(f"{tactic_name} (intensity: {intensity:.2f})")
                    else:
                        summary['rhetorical_tactics'].append(str(tactic))
        
        # Extract bias indicators
        if 'bias_indicators' in analysis_data:
            indicators = analysis_data['bias_indicators']
            if isinstance(indicators, list):
                summary['bias_indicators'] = [str(indicator) for indicator in indicators[:3]]
        
        # Extract evidence quality assessment
        truthfulness = analysis_data.get('truthfulness', {})
        if isinstance(truthfulness, dict):
            summary['evidence_quality'] = truthfulness.get('evidence_quality', 'Unknown')
        
        # Extract key concerns
        red_flags = analysis_data.get('red_flags', [])
        if isinstance(red_flags, list):
            summary['key_concerns'] = [str(flag) for flag in red_flags[:3]]
        
        return summary
    
    def _classify_segment_types_enhanced(self, segments: List[ContentSegment], analysis_data: Dict[str, Any]) -> List[ContentSegment]:
        """Enhanced segment classification using actual analysis data"""
        
        # Extract key claims and rhetorical tactics for better classification
        key_claims = analysis_data.get('claims', [])
        if not key_claims:
            key_claims = analysis_data.get('key_claims', [])  # Fallback
        rhetorical_tactics = analysis_data.get('rhetorical_tactics', [])
        
        # Convert claims to text for matching
        key_claim_texts = []
        if isinstance(key_claims, list):
            for claim in key_claims:
                if isinstance(claim, dict):
                    key_claim_texts.append(claim.get('text', str(claim)))
                else:
                    key_claim_texts.append(str(claim))
        
        for i, segment in enumerate(segments):
            segment_text = segment.text.lower()
            duration = segment.end_time - segment.start_time
            
            # Check if segment contains key claims
            contains_key_claim = False
            for claim_text in key_claim_texts:
                if self._text_overlap(segment_text, claim_text.lower()) > 0.3:
                    contains_key_claim = True
                    break
            
            # Classification logic based on content, position, and analysis
            if i == 0:  # First segment
                if any(keyword in segment_text for keyword in ['hello', 'welcome', 'today', 'going to', 'let me', 'we will']):
                    segment.segment_type = 'intro'
                elif contains_key_claim:
                    segment.segment_type = 'key_claim'
                else:
                    segment.segment_type = 'intro'
            
            elif i >= len(segments) - 1:  # Last segment
                if any(keyword in segment_text for keyword in ['conclusion', 'summary', 'final', 'end', 'thank you']):
                    segment.segment_type = 'conclusion'
                elif contains_key_claim:
                    segment.segment_type = 'key_claim'
                else:
                    segment.segment_type = 'conclusion'
            
            else:  # Middle segments
                if contains_key_claim:
                    segment.segment_type = 'key_claim'
                elif segment.importance_score > 0.6:
                    segment.segment_type = 'evidence'
                elif any(keyword in segment_text for keyword in ['because', 'therefore', 'research shows', 'according to']):
                    segment.segment_type = 'evidence'
                else:
                    segment.segment_type = 'filler'
        
        return segments
    
    def _extract_key_themes_enhanced(self, segments: List[ContentSegment], analysis_data: Dict[str, Any]) -> List[str]:
        """Enhanced theme extraction using analysis data"""
        themes = []
        
        # Extract themes from rhetorical tactics
        rhetorical_tactics = analysis_data.get('rhetorical_tactics', [])
        if isinstance(rhetorical_tactics, list):
            for tactic in rhetorical_tactics[:3]:
                if isinstance(tactic, dict):
                    tactic_name = tactic.get('tactic', str(tactic))
                    themes.append(f"Rhetorical tactic: {tactic_name}")
                else:
                    themes.append(f"Rhetorical tactic: {str(tactic)}")
        
        # Extract themes from bias indicators
        bias_indicators = analysis_data.get('bias_indicators', [])
        if isinstance(bias_indicators, list) and bias_indicators:
            themes.append(f"Bias detected: {len(bias_indicators)} indicators")
        
        # Extract themes from segment content
        all_text = " ".join([segment.text for segment in segments])
        if all_text:
            try:
                blob = TextBlob(all_text)
                noun_phrases = blob.noun_phrases
                
                # Get most common themes
                theme_counts = {}
                for phrase in noun_phrases:
                    if len(phrase.split()) >= 2 and len(phrase) > 5:  # Multi-word phrases only
                        theme_counts[phrase] = theme_counts.get(phrase, 0) + 1
                
                # Add top content themes
                sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
                content_themes = [theme[0] for theme in sorted_themes[:2]]
                themes.extend(content_themes)
            except Exception as e:
                logger.warning(f"Theme extraction failed: {e}")
        
        return themes[:5]  # Limit to 5 themes
    
    def _create_fallback_content(self, truthscore_data: Dict[str, Any], target_type: str) -> VideoContent:
        """Create fallback content when classification fails"""
        logger.warning("Creating fallback content due to classification error")
        
        # Extract basic transcript
        transcript_data = self._extract_transcript_segments(truthscore_data)
        
        # Create basic segments
        fallback_segments = []
        for i, segment in enumerate(transcript_data[:3]):  # Limit to first 3 segments
            content_segment = ContentSegment(
                start_time=segment.get('start', i * 10),
                end_time=segment.get('end', (i + 1) * 10),
                text=segment.get('text', 'Content unavailable'),
                speaker=segment.get('speaker', 'SPEAKER_01'),
                importance_score=0.5,
                segment_type='key_claim' if i == 1 else 'intro' if i == 0 else 'conclusion',
                credibility_indicators={'fallback': True},
                claims=[],
                research_backing=[]
            )
            fallback_segments.append(content_segment)
        
        return VideoContent(
            segments=fallback_segments,
            overall_narrative="Fallback analysis due to processing error",
            key_themes=["Content analysis", "Fallback mode"],
            credibility_score=50.0,
            recommended_duration=25.0,
            content_warnings=["Analysis incomplete due to processing error"],
            chain_of_thought="Fallback content created due to classification error",
            analysis_summary={'fallback': True}
        )

    def _extract_transcript_segments(self, truthscore_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and normalize transcript segments from TruthScore data"""
        transcript = truthscore_data.get('transcript', {})
        
        if isinstance(transcript, dict) and 'segments' in transcript:
            segments = transcript['segments']
        elif isinstance(transcript, str):
            # Create single segment from string transcript
            segments = [{
                'start': 0.0,
                'end': 60.0,  # Default duration
                'text': transcript,
                'speaker': 'SPEAKER_01'
            }]
        else:
            segments = []
        
        # Normalize segment format
        normalized_segments = []
        for segment in segments:
            normalized_segments.append({
                'start': float(segment.get('start', 0)),
                'end': float(segment.get('end', 0)),
                'text': str(segment.get('text', '')),
                'speaker': segment.get('speaker', 'UNKNOWN'),
                'original_segment': segment
            })
        
        return normalized_segments
    
    def _calculate_semantic_density(self, segments: List[Dict[str, Any]]) -> Dict[int, float]:
        """Calculate semantic density scores using TF-IDF analysis"""
        if not segments:
            return {}
        
        # Extract text content
        texts = [segment['text'] for segment in segments]
        
        try:
            # Calculate TF-IDF matrix
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate semantic density scores
            semantic_scores = {}
            for i, segment in enumerate(segments):
                # Average TF-IDF score for the segment
                segment_vector = tfidf_matrix[i]
                density_score = np.mean(segment_vector.data) if segment_vector.data.size > 0 else 0.0
                
                # Normalize by segment length (longer segments get slightly lower scores)
                segment_length = len(segment['text'].split())
                length_factor = min(1.0, 50 / max(segment_length, 1))  # Optimal around 50 words
                
                semantic_scores[i] = density_score * length_factor
            
            return semantic_scores
            
        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {str(e)}, using fallback scoring")
            # Fallback to simple word density
            return {i: len(segment['text'].split()) / 100 for i, segment in enumerate(segments)}
    
    def _score_claim_verification(self, segments: List[Dict[str, Any]], analysis_data: Dict[str, Any]) -> Dict[int, float]:
        """Score segments based on claim verification status"""
        claims = analysis_data.get('claims', [])  # Updated key name to match export structure
        if not claims:
            claims = analysis_data.get('key_claims', [])  # Fallback
        red_flags = analysis_data.get('red_flags', [])
        
        claim_scores = {}
        
        for i, segment in enumerate(segments):
            segment_text = segment['text'].lower()
            score = 0.0
            
            # Check for verified claims
            for claim in claims:
                if isinstance(claim, dict):
                    claim_text = str(claim.get('text', '')).lower()
                else:
                    claim_text = str(claim).lower()
                
                if claim_text and self._text_overlap(segment_text, claim_text) > 0.3:
                    # High score for segments containing verified claims
                    score += 0.8
            
            # Penalize for red flags
            for flag in red_flags:
                if isinstance(flag, dict):
                    flag_text = str(flag.get('text', '')).lower()
                else:
                    flag_text = str(flag).lower()
                
                if flag_text and self._text_overlap(segment_text, flag_text) > 0.3:
                    score -= 0.5
            
            # Ensure score is between 0 and 1
            claim_scores[i] = max(0.0, min(1.0, score))
        
        return claim_scores
    
    def _calculate_speaker_credibility(self, segments: List[Dict[str, Any]], speakers: Dict[str, Any]) -> Dict[int, float]:
        """Calculate speaker credibility scores"""
        speaker_scores = {}
        
        for i, segment in enumerate(segments):
            speaker_id = segment.get('speaker', 'UNKNOWN')
            speaker_info = speakers.get(speaker_id, {})
            
            # Base credibility from speaker identification confidence
            confidence = speaker_info.get('confidence', 0.5)
            
            # Boost for identified speakers vs unknown
            identification_bonus = 0.2 if speaker_info.get('identified_name') else 0.0
            
            speaker_scores[i] = min(1.0, confidence + identification_bonus)
        
        return speaker_scores
    
    def _analyze_research_backing(self, segments: List[Dict[str, Any]], research_data: Dict[str, Any]) -> Dict[int, float]:
        """Analyze research backing for segment content"""
        research_scores = {}
        
        # Extract research results
        claim_verification = research_data.get('claim_verification', [])
        topic_research = research_data.get('topic_research', [])
        
        # Handle case where topic_research is a dict (like in our test data)
        if isinstance(topic_research, dict):
            topic_research = [topic_research]
        
        for i, segment in enumerate(segments):
            segment_text = segment['text'].lower()
            score = 0.0
            
            # Check against claim verification research
            for research_item in claim_verification:
                if self._segment_matches_research(segment_text, research_item):
                    verification_status = research_item.get('verification_status', 'UNVERIFIABLE')
                    if verification_status == 'VERIFIED':
                        score += 0.8
                    elif verification_status == 'PARTIALLY_VERIFIED':
                        score += 0.5
                    elif verification_status == 'DISPUTED':
                        score -= 0.3
            
            # Check against topic research
            for topic_item in topic_research:
                if self._segment_matches_research(segment_text, topic_item):
                    score += 0.3
            
            research_scores[i] = max(0.0, min(1.0, score))
        
        return research_scores
    
    def _calculate_importance_scores(self, segments: List[Dict[str, Any]], 
                                   semantic_scores: Dict[int, float],
                                   claim_scores: Dict[int, float],
                                   speaker_scores: Dict[int, float],
                                   research_scores: Dict[int, float]) -> List[ContentSegment]:
        """Calculate composite importance scores using weighted combination"""
        content_segments = []
        weights = self.classification_config['importance_weights']
        
        for i, segment in enumerate(segments):
            # Get individual scores
            semantic = semantic_scores.get(i, 0.0)
            claim = claim_scores.get(i, 0.0)
            speaker = speaker_scores.get(i, 0.0)
            research = research_scores.get(i, 0.0)
            
            # Calculate composite score
            importance_score = (
                semantic * weights['semantic_density'] +
                claim * weights['claim_verification'] +
                speaker * weights['speaker_confidence'] +
                research * weights['research_backing']
            )
            
            # Create ContentSegment
            content_segment = ContentSegment(
                start_time=segment['start'],
                end_time=segment['end'],
                text=segment['text'],
                speaker=segment.get('speaker'),
                importance_score=importance_score,
                segment_type='unclassified',  # Will be classified in next step
                credibility_indicators={
                    'semantic_density': semantic,
                    'claim_verification': claim,
                    'speaker_confidence': speaker,
                    'research_backing': research
                },
                claims=[],  # Will be populated later
                research_backing=[]  # Will be populated later
            )
            
            content_segments.append(content_segment)
        
        return content_segments
    
    def _optimize_for_target_duration(self, segments: List[ContentSegment], target_type: str) -> List[ContentSegment]:
        """Optimize segment selection for target video duration"""
        duration_config = self.classification_config['duration_targets'][target_type]
        target_duration = duration_config['optimal']
        
        # Sort segments by importance score
        sorted_segments = sorted(segments, key=lambda x: x.importance_score, reverse=True)
        
        # Select segments for optimal narrative flow
        selected_segments = []
        current_duration = 0.0
        
        # Always include highest importance segments first
        for segment in sorted_segments:
            segment_duration = segment.end_time - segment.start_time
            
            # Check if adding this segment would exceed target duration
            if current_duration + segment_duration <= duration_config['max']:
                selected_segments.append(segment)
                current_duration += segment_duration
                
                # Stop if we've reached optimal duration with high-importance content
                if (current_duration >= duration_config['min'] and 
                    segment.importance_score < 0.6):
                    break
            
            # Must have minimum duration
            elif current_duration < duration_config['min']:
                selected_segments.append(segment)
                current_duration += segment_duration
        
        # Sort selected segments back to chronological order
        selected_segments.sort(key=lambda x: x.start_time)
        
        return selected_segments
    
    def _generate_narrative_structure(self, segments: List[ContentSegment], analysis_data: Dict[str, Any]) -> str:
        """Generate overall narrative description for the video"""
        key_claims = [s for s in segments if s.segment_type == 'key_claim']
        credibility_score = analysis_data.get('credibility_score', 50)
        
        if credibility_score >= 70:
            credibility_desc = "high credibility"
        elif credibility_score >= 40:
            credibility_desc = "mixed credibility" 
        else:
            credibility_desc = "low credibility"
        
        narrative = f"Video analysis of content with {credibility_desc} (score: {credibility_score}/100). "
        
        if key_claims:
            narrative += f"Features {len(key_claims)} key claims with verification analysis. "
        
        total_duration = sum(s.end_time - s.start_time for s in segments)
        narrative += f"Total duration: {total_duration:.1f} seconds."
        
        return narrative
    
    def _generate_content_warnings(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate content warnings based on analysis"""
        warnings = []
        
        # Check for bias indicators
        bias_indicators = analysis_data.get('bias_indicators', [])
        if bias_indicators:
            warnings.append("Contains potential bias indicators")
        
        # Check for red flags
        red_flags = analysis_data.get('red_flags', [])
        if red_flags:
            warnings.append("Contains factual accuracy concerns")
        
        # Check credibility score
        credibility_score = analysis_data.get('credibility_score', 50)
        if credibility_score < 40:
            warnings.append("Low credibility score - verify claims independently")
        
        return warnings
    
    def _calculate_recommended_duration(self, segments: List[ContentSegment]) -> float:
        """Calculate recommended video duration based on selected segments"""
        return sum(segment.end_time - segment.start_time for segment in segments)
    
    # Helper methods
    def _text_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap ratio using simple word matching"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _segment_matches_research(self, segment_text: str, research_item: Any) -> bool:
        """Check if segment content matches research item"""
        if isinstance(research_item, dict):
            research_text = str(research_item.get('claim', '')) + " " + str(research_item.get('research_summary', ''))
        else:
            research_text = str(research_item)
        
        if not research_text.strip():
            return False
            
        return self._text_overlap(segment_text, research_text.lower()) > 0.2
    
 