"""
Deck summarization system for generating 4-bullet point summaries of pitch decks.
Uses TextBlob and key point extraction algorithms based on section importance.
"""
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from textblob import TextBlob, Sentence
from dataclasses import dataclass


@dataclass
class SummaryPoint:
    """Represents a single summary point with metadata."""
    text: str
    section: str
    importance_score: float
    keywords: List[str]


@dataclass
class DeckSummary:
    """Complete deck summary with quality metrics."""
    summary_points: List[str]
    quality_score: float
    coverage_sections: List[str]
    key_themes: List[str]
    confidence_level: float


class DeckSummarizer:
    """
    Generates 4-bullet point summaries of pitch decks using NLP techniques.
    Implements key point extraction based on section importance and content analysis.
    """
    
    def __init__(self):
        """Initialize the deck summarizer with section weights and extraction patterns."""
        # Section importance weights for summary generation
        self.section_weights = {
            'problem': 0.20,
            'solution': 0.25,
            'market': 0.15,
            'traction': 0.20,
            'business_model': 0.10,
            'team': 0.05,
            'funding': 0.05
        }
        
        # Key phrases that indicate important summary-worthy content
        self.importance_indicators = [
            # Problem indicators
            'problem', 'challenge', 'pain point', 'issue', 'difficulty', 'struggle',
            'inefficient', 'broken', 'frustrating', 'costly', 'time-consuming',
            
            # Solution indicators
            'solution', 'solve', 'address', 'fix', 'improve', 'optimize', 'streamline',
            'platform', 'technology', 'system', 'approach', 'method', 'innovation',
            
            # Market indicators
            'market', 'industry', 'customers', 'users', 'target', 'segment',
            'opportunity', 'demand', 'need', 'tam', 'sam', 'som', 'billion', 'million',
            
            # Traction indicators
            'growth', 'traction', 'users', 'customers', 'revenue', 'sales',
            'adoption', 'engagement', 'retention', 'partnerships', 'clients',
            'mom', 'yoy', 'increase', 'scale', 'expansion',
            
            # Business model indicators
            'monetization', 'revenue model', 'pricing', 'subscription', 'saas',
            'marketplace', 'commission', 'freemium', 'enterprise', 'b2b', 'b2c',
            
            # Team indicators
            'team', 'founder', 'experience', 'background', 'expertise', 'track record',
            'led', 'built', 'scaled', 'previous', 'former',
            
            # Funding indicators
            'funding', 'investment', 'capital', 'raise', 'series', 'seed',
            'investors', 'valuation', 'use of funds'
        ]
        
        # Sentence quality patterns
        self.quality_patterns = {
            'metrics': r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|million|billion|k|thousand|users|customers|revenue)',
            'growth': r'\d+(?:\.\d+)?\s*%\s*(?:growth|increase|improvement)',
            'specificity': r'(?:specifically|exactly|precisely|particularly|notably)',
            'achievements': r'(?:achieved|accomplished|delivered|secured|reached|built|launched)',
            'quantified': r'(?:\$\d+|\d+\s*(?:million|billion|k|thousand|%|x|times))',
        }
        
        # Fallback templates for different scenarios
        self.fallback_templates = {
            'generic': [
                "The company addresses a significant market opportunity with an innovative solution.",
                "The team has relevant experience and domain expertise in their target market.",
                "The business model demonstrates potential for scalable revenue generation.",
                "The startup shows early signs of market validation and customer interest."
            ],
            'tech_focused': [
                "The platform leverages advanced technology to solve industry challenges.",
                "The solution provides measurable improvements over existing alternatives.",
                "The company targets a large addressable market with strong demand signals.",
                "The technical approach offers competitive advantages and defensibility."
            ],
            'market_focused': [
                "The startup addresses a well-defined problem in a growing market segment.",
                "The solution demonstrates clear value proposition for target customers.",
                "The business model aligns with market dynamics and customer needs.",
                "The company shows understanding of competitive landscape and positioning."
            ]
        }
    
    def generate_summary(self, sections: Dict[str, str], deck_name: str = "") -> DeckSummary:
        """
        Generate a 4-bullet point summary of a pitch deck.
        
        Args:
            sections: Dictionary of pitch deck sections and their content
            deck_name: Optional name of the deck for context
            
        Returns:
            DeckSummary with 4 bullet points and quality metrics
        """
        if not sections or not any(content.strip() for content in sections.values()):
            return self._generate_fallback_summary("generic")
        
        # Extract candidate summary points from all sections
        candidate_points = self._extract_candidate_points(sections)
        
        # Score and rank candidate points
        scored_points = self._score_summary_points(candidate_points, sections)
        
        # Select top 4 points ensuring diversity
        selected_points = self._select_diverse_points(scored_points, target_count=4)
        
        # Generate final summary text
        summary_texts = self._format_summary_points(selected_points)
        
        # Calculate quality metrics
        quality_score = self._calculate_summary_quality(selected_points, sections)
        coverage_sections = list(set(point.section for point in selected_points))
        key_themes = self._extract_key_themes(selected_points)
        confidence_level = self._calculate_confidence_level(selected_points, sections)
        
        # Apply fallback if quality is too low
        if quality_score < 0.3 or len(summary_texts) < 4:
            return self._generate_fallback_summary(self._determine_fallback_type(sections))
        
        return DeckSummary(
            summary_points=summary_texts,
            quality_score=quality_score,
            coverage_sections=coverage_sections,
            key_themes=key_themes,
            confidence_level=confidence_level
        )
    
    def _extract_candidate_points(self, sections: Dict[str, str]) -> List[SummaryPoint]:
        """
        Extract candidate summary points from all sections.
        
        Args:
            sections: Dictionary of section content
            
        Returns:
            List of candidate summary points
        """
        candidate_points = []
        
        for section_name, content in sections.items():
            if not content or not content.strip():
                continue
            
            # Clean section name for processing
            clean_section = section_name.lower().replace(' ', '_')
            
            # Extract sentences from section
            try:
                blob = TextBlob(content)
                sentences = blob.sentences
            except:
                # Fallback to simple sentence splitting
                sentences = [Sentence(s.strip()) for s in re.split(r'[.!?]+', content) if s.strip()]
            
            # Process each sentence as a potential summary point
            for sentence in sentences:
                sentence_text = str(sentence).strip()
                
                # Skip very short or very long sentences
                if len(sentence_text) < 20 or len(sentence_text) > 200:
                    continue
                
                # Calculate importance score for this sentence
                importance_score = self._calculate_sentence_importance(
                    sentence_text, clean_section, content
                )
                
                # Extract keywords from sentence
                keywords = self._extract_sentence_keywords(sentence_text)
                
                candidate_points.append(SummaryPoint(
                    text=sentence_text,
                    section=clean_section,
                    importance_score=importance_score,
                    keywords=keywords
                ))
        
        return candidate_points
    
    def _calculate_sentence_importance(self, sentence: str, section: str, full_content: str) -> float:
        """
        Calculate importance score for a sentence based on multiple factors.
        
        Args:
            sentence: The sentence text
            section: Section name where sentence appears
            full_content: Full content of the section
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Base score from section weight
        section_weight = self.section_weights.get(section, 0.1)
        score += section_weight
        
        # Boost for importance indicators
        indicator_matches = sum(1 for indicator in self.importance_indicators 
                              if indicator in sentence_lower)
        score += min(indicator_matches * 0.1, 0.3)
        
        # Boost for quality patterns (metrics, specificity, etc.)
        for pattern_name, pattern in self.quality_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 0.15
        
        # Boost for sentence position (first and last sentences often important)
        sentences_in_section = len(re.split(r'[.!?]+', full_content))
        if sentences_in_section > 1:
            sentence_position = full_content.lower().find(sentence_lower)
            relative_position = sentence_position / len(full_content)
            
            # Boost for beginning and end of sections
            if relative_position < 0.2 or relative_position > 0.8:
                score += 0.1
        
        # Penalty for very generic language
        generic_phrases = ['we are', 'our company', 'this is', 'we believe', 'we think']
        generic_count = sum(1 for phrase in generic_phrases if phrase in sentence_lower)
        score -= generic_count * 0.05
        
        # Boost for concrete, specific language
        if re.search(r'\b(?:specifically|exactly|precisely|particularly)\b', sentence_lower):
            score += 0.1
        
        # Boost for action-oriented language
        action_words = ['built', 'created', 'developed', 'launched', 'achieved', 'delivered']
        action_count = sum(1 for word in action_words if word in sentence_lower)
        score += min(action_count * 0.05, 0.15)
        
        return min(score, 1.0)
    
    def _extract_sentence_keywords(self, sentence: str) -> List[str]:
        """
        Extract key terms from a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set', 'run'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Return most frequent keywords
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(5)]
    
    def _score_summary_points(self, candidate_points: List[SummaryPoint], 
                            sections: Dict[str, str]) -> List[SummaryPoint]:
        """
        Score and rank candidate summary points.
        
        Args:
            candidate_points: List of candidate points
            sections: Original section content for context
            
        Returns:
            Sorted list of scored summary points
        """
        # Additional scoring based on global context
        for point in candidate_points:
            # Boost for unique information (not repeated across sections)
            uniqueness_score = self._calculate_uniqueness(point, candidate_points)
            point.importance_score += uniqueness_score * 0.1
            
            # Boost for complementary information
            complementary_score = self._calculate_complementary_value(point, candidate_points)
            point.importance_score += complementary_score * 0.05
        
        # Sort by importance score
        return sorted(candidate_points, key=lambda x: x.importance_score, reverse=True)
    
    def _calculate_uniqueness(self, point: SummaryPoint, all_points: List[SummaryPoint]) -> float:
        """
        Calculate how unique a point is compared to others.
        
        Args:
            point: The point to evaluate
            all_points: All candidate points
            
        Returns:
            Uniqueness score between 0.0 and 1.0
        """
        point_keywords = set(point.keywords)
        
        if not point_keywords:
            return 0.5  # Neutral score for points without keywords
        
        # Calculate overlap with other points
        total_overlap = 0
        comparison_count = 0
        
        for other_point in all_points:
            if other_point == point:
                continue
            
            other_keywords = set(other_point.keywords)
            if other_keywords:
                overlap = len(point_keywords.intersection(other_keywords))
                total_overlap += overlap / len(point_keywords.union(other_keywords))
                comparison_count += 1
        
        if comparison_count == 0:
            return 1.0
        
        # Higher uniqueness = lower average overlap
        average_overlap = total_overlap / comparison_count
        return 1.0 - average_overlap
    
    def _calculate_complementary_value(self, point: SummaryPoint, 
                                     all_points: List[SummaryPoint]) -> float:
        """
        Calculate how well a point complements others in covering different aspects.
        
        Args:
            point: The point to evaluate
            all_points: All candidate points
            
        Returns:
            Complementary value score between 0.0 and 1.0
        """
        # Check section diversity
        sections_covered = set(p.section for p in all_points[:4])  # Top 4 points
        
        if point.section not in sections_covered:
            return 1.0  # High value for covering new section
        
        # Check keyword diversity
        keywords_covered = set()
        for p in all_points[:4]:
            keywords_covered.update(p.keywords)
        
        new_keywords = set(point.keywords) - keywords_covered
        keyword_diversity = len(new_keywords) / max(len(point.keywords), 1)
        
        return keyword_diversity
    
    def _select_diverse_points(self, scored_points: List[SummaryPoint], 
                             target_count: int = 4) -> List[SummaryPoint]:
        """
        Select diverse summary points ensuring good coverage.
        
        Args:
            scored_points: Sorted list of scored points
            target_count: Number of points to select
            
        Returns:
            List of selected diverse points
        """
        if len(scored_points) <= target_count:
            return scored_points
        
        selected = []
        used_sections = set()
        used_keywords = set()
        
        # First pass: select highest scoring points from different sections
        for point in scored_points:
            if len(selected) >= target_count:
                break
            
            # Prefer points from new sections
            if point.section not in used_sections:
                selected.append(point)
                used_sections.add(point.section)
                used_keywords.update(point.keywords)
        
        # Second pass: fill remaining slots with best remaining points
        for point in scored_points:
            if len(selected) >= target_count:
                break
            
            if point not in selected:
                # Check for keyword diversity
                new_keywords = set(point.keywords) - used_keywords
                if len(new_keywords) > 0 or len(selected) < target_count:
                    selected.append(point)
                    used_keywords.update(point.keywords)
        
        # Ensure we have exactly target_count points
        while len(selected) < target_count and len(selected) < len(scored_points):
            for point in scored_points:
                if point not in selected:
                    selected.append(point)
                    break
        
        return selected[:target_count]
    
    def _format_summary_points(self, selected_points: List[SummaryPoint]) -> List[str]:
        """
        Format selected points into clean summary text.
        
        Args:
            selected_points: List of selected summary points
            
        Returns:
            List of formatted summary strings
        """
        formatted_points = []
        
        for point in selected_points:
            # Clean up the text
            text = point.text.strip()
            
            # Ensure proper capitalization
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]
            
            # Ensure proper ending punctuation
            if text and text[-1] not in '.!?':
                text += '.'
            
            # Remove redundant phrases
            text = re.sub(r'\b(?:we are|our company is|this is)\s+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Ensure minimum quality
            if len(text) >= 15:
                formatted_points.append(text)
        
        return formatted_points
    
    def _calculate_summary_quality(self, selected_points: List[SummaryPoint], 
                                 sections: Dict[str, str]) -> float:
        """
        Calculate overall quality score for the generated summary.
        
        Args:
            selected_points: Selected summary points
            sections: Original section content
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not selected_points:
            return 0.0
        
        # Coverage score (how many important sections are covered)
        important_sections = ['problem', 'solution', 'market', 'traction']
        covered_sections = set(point.section for point in selected_points)
        coverage_score = len(covered_sections.intersection(important_sections)) / len(important_sections)
        
        # Average importance score of selected points
        avg_importance = sum(point.importance_score for point in selected_points) / len(selected_points)
        
        # Diversity score (keyword and section diversity)
        all_keywords = set()
        for point in selected_points:
            all_keywords.update(point.keywords)
        diversity_score = min(len(all_keywords) / 10.0, 1.0)  # Normalize to max 10 unique keywords
        
        # Length appropriateness (not too short, not too long)
        avg_length = sum(len(point.text) for point in selected_points) / len(selected_points)
        length_score = 1.0 if 30 <= avg_length <= 150 else 0.7
        
        # Combine scores
        quality_score = (coverage_score * 0.3 + avg_importance * 0.4 + 
                        diversity_score * 0.2 + length_score * 0.1)
        
        return min(quality_score, 1.0)
    
    def _extract_key_themes(self, selected_points: List[SummaryPoint]) -> List[str]:
        """
        Extract key themes from selected summary points.
        
        Args:
            selected_points: Selected summary points
            
        Returns:
            List of key themes
        """
        all_keywords = []
        for point in selected_points:
            all_keywords.extend(point.keywords)
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        # Return top themes
        return [keyword for keyword, count in keyword_counts.most_common(5)]
    
    def _calculate_confidence_level(self, selected_points: List[SummaryPoint], 
                                  sections: Dict[str, str]) -> float:
        """
        Calculate confidence level in the generated summary.
        
        Args:
            selected_points: Selected summary points
            sections: Original section content
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        if not selected_points:
            return 0.0
        
        # Base confidence from point quality
        avg_importance = sum(point.importance_score for point in selected_points) / len(selected_points)
        
        # Boost for having content from multiple sections
        section_count = len(set(point.section for point in selected_points))
        section_bonus = min(section_count / 4.0, 1.0)
        
        # Boost for having quantitative information
        quantitative_count = sum(1 for point in selected_points 
                               if re.search(r'\d+', point.text))
        quantitative_bonus = min(quantitative_count / 4.0, 0.3)
        
        # Penalty for using fallback content
        fallback_penalty = 0.0  # Will be set if fallback is used
        
        confidence = avg_importance * 0.6 + section_bonus * 0.3 + quantitative_bonus * 0.1 - fallback_penalty
        
        return min(confidence, 1.0)
    
    def _determine_fallback_type(self, sections: Dict[str, str]) -> str:
        """
        Determine the most appropriate fallback template type.
        
        Args:
            sections: Section content to analyze
            
        Returns:
            Fallback template type
        """
        all_content = ' '.join(sections.values()).lower()
        
        # Check for technology focus
        tech_keywords = ['platform', 'technology', 'software', 'algorithm', 'ai', 'ml', 'api']
        tech_count = sum(1 for keyword in tech_keywords if keyword in all_content)
        
        # Check for market focus
        market_keywords = ['market', 'customers', 'users', 'segment', 'demand', 'opportunity']
        market_count = sum(1 for keyword in market_keywords if keyword in all_content)
        
        if tech_count > market_count:
            return 'tech_focused'
        elif market_count > tech_count:
            return 'market_focused'
        else:
            return 'generic'
    
    def _generate_fallback_summary(self, fallback_type: str) -> DeckSummary:
        """
        Generate fallback summary when extraction fails.
        
        Args:
            fallback_type: Type of fallback template to use
            
        Returns:
            DeckSummary with fallback content
        """
        template = self.fallback_templates.get(fallback_type, self.fallback_templates['generic'])
        
        return DeckSummary(
            summary_points=template,
            quality_score=0.4,  # Low quality score for fallback
            coverage_sections=['fallback'],
            key_themes=['general', 'business', 'startup'],
            confidence_level=0.3  # Low confidence for fallback
        )