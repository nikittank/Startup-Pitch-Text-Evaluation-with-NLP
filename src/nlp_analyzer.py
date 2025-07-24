"""
NLP Analyzer for pitch deck content analysis.
Provides sentiment analysis, keyword extraction, readability calculation,
and confidence indicator detection.
"""
import re
import math
from typing import List, Dict, Any
from collections import Counter
from textblob import TextBlob
from ..models.interfaces import NLPAnalyzerInterface


class NLPAnalyzer(NLPAnalyzerInterface):
    """
    Core NLP analyzer for pitch deck content analysis.
    Implements sentiment analysis, keyword extraction, readability metrics,
    and confidence indicator detection.
    """
    
    def __init__(self):
        """Initialize NLP analyzer with confidence indicators and patterns."""
        self.confidence_indicators = [
            # Strong confidence indicators
            'proven', 'demonstrated', 'validated', 'confirmed', 'established',
            'successful', 'achieved', 'accomplished', 'delivered', 'secured',
            'guaranteed', 'certain', 'definitive', 'conclusive', 'verified',
            
            # Growth and traction indicators
            'growing', 'increasing', 'expanding', 'scaling', 'accelerating',
            'momentum', 'traction', 'adoption', 'engagement', 'retention',
            
            # Market position indicators
            'leading', 'dominant', 'competitive advantage', 'first-mover',
            'market leader', 'industry standard', 'breakthrough', 'innovative',
            
            # Financial confidence
            'profitable', 'revenue', 'monetization', 'sustainable', 'scalable',
            'recurring', 'predictable', 'stable', 'strong margins'
        ]
        
        # Comprehensive patterns for metrics extraction
        self.metrics_patterns = {
            'growth': [
                # Growth percentages and rates
                r'(\d+(?:\.\d+)?)\s*%\s*(?:growth|increase|improvement|rise|boost|jump|surge)',
                r'(?:growth|increase|improvement|rise|boost|jump|surge)\s*(?:of\s*|by\s*)?(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*x\s*(?:growth|increase|faster|more)',
                r'(?:grew|increased|improved|rose|boosted)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:MoM|YoY|month-over-month|year-over-year)',
                r'(?:MoM|YoY|month-over-month|year-over-year)\s*(?:growth\s*(?:of\s*|by\s*)?)?(\d+(?:\.\d+)?)\s*%',
            ],
            'user_metrics': [
                # User and customer numbers
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:active\s*)?(?:users|customers|subscribers|members|downloads|installs)',
                r'(?:users|customers|subscribers|members|downloads|installs).*?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:daily|monthly|weekly)\s*(?:active\s*)?(?:users|customers)',
                r'(?:DAU|MAU|WAU).*?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?',
                r'user\s*base.*?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?',
            ],
            'financial': [
                # Revenue and financial metrics
                r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B|k|thousand)?\s*(?:in\s*)?(?:revenue|sales|income|ARR|MRR)',
                r'(?:revenue|sales|income|ARR|MRR).*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B|k|thousand)?',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B|k|thousand)?\s*(?:in\s*)?(?:funding|investment|raised)',
                r'(?:funding|investment|raised).*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B|k|thousand)?',
                r'valuation.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B|k|thousand)?',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:margin|profit|EBITDA)',
            ],
            'traction': [
                # Traction and engagement metrics
                r'(\d+(?:\.\d+)?)\s*%\s*(?:retention|engagement|conversion|churn)',
                r'(?:retention|engagement|conversion|churn)\s*(?:rate\s*)?(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:page\s*views|sessions|transactions)',
                r'(?:NPS|Net\s*Promoter\s*Score).*?(\d+)',
                r'(\d+(?:\.\d+)?)\s*(?:months?|years?)\s*(?:payback|LTV|lifetime\s*value)',
            ],
            'market': [
                # Market size and opportunity metrics
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)\s*(?:TAM|SAM|SOM|market)',
                r'(?:TAM|SAM|SOM|market\s*size).*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)',
                r'market\s*opportunity.*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:market\s*share|penetration)',
            ],
            'temporal': [
                # Time-based metrics
                r'(\d+(?:\.\d+)?)\s*(?:months?|years?)\s*(?:experience|background|track\s*record)',
                r'(?:founded|established|started|launched)\s*(?:in\s*)?(\d{4})',
                r'(\d+(?:\.\d+)?)\s*(?:months?|years?)\s*(?:ago|since)',
                r'(?:over|more\s*than)\s*(\d+(?:\.\d+)?)\s*(?:months?|years?)',
            ]
        }
        
        # Enhanced team experience and domain expertise patterns
        self.team_experience_patterns = [
            # Leadership roles and companies
            r'(?:founder|co-founder|CEO|CTO|CRO|VP|director|manager|head\s+of).*?(?:at|from|of)\s+([A-Z][a-zA-Z\s&\.]+(?:Inc|Corp|LLC|Ltd)?)',
            r'(?:former|ex-|previously)(?:\s+(?:at|with))?\s+([A-Z][a-zA-Z\s&\.]+(?:Inc|Corp|LLC|Ltd)?)',
            
            # Years of experience
            r'(\d+)\s*(?:\+)?\s*years?\s*(?:of\s*)?(?:experience|background|track\s*record)',
            r'(?:over|more\s+than)\s*(\d+)\s*years?\s*(?:of\s*)?(?:experience|background)',
            
            # Education and credentials
            r'(?:PhD|MBA|MS|BS|Bachelor|Master|Doctor).*?(?:from|at|in)\s+([A-Z][a-zA-Z\s&\.]+(?:University|College|Institute|School))',
            r'(?:graduated|degree).*?(?:from|at)\s+([A-Z][a-zA-Z\s&\.]+(?:University|College|Institute|School))',
            
            # Work experience
            r'(?:worked|experience|served).*?(?:at|with|for)\s+([A-Z][a-zA-Z\s&\.]+(?:Inc|Corp|LLC|Ltd)?)',
            r'(?:led|managed|built).*?(?:at|for)\s+([A-Z][a-zA-Z\s&\.]+(?:Inc|Corp|LLC|Ltd)?)',
            
            # Domain expertise indicators
            r'(?:expert|specialist|authority).*?(?:in|on)\s+([a-zA-Z\s]+(?:technology|industry|market|field))',
            r'(?:deep\s+)?(?:expertise|knowledge|understanding).*?(?:in|of)\s+([a-zA-Z\s]+)',
            
            # Industry recognition
            r'(?:recognized|awarded|featured).*?(?:by|in)\s+([A-Z][a-zA-Z\s&\.]+)',
            r'(?:speaker|keynote|presenter).*?(?:at|for)\s+([A-Z][a-zA-Z\s&\.]+)',
        ]
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text content using TextBlob.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment polarity score between -1 (negative) and 1 (positive)
        """
        if not text or not text.strip():
            return 0.0
            
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0.0
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract key terms and phrases from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted keywords sorted by frequency
        """
        if not text or not text.strip():
            return []
            
        try:
            # Clean and tokenize text
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = cleaned_text.split()
            
            # Filter out common stop words and short words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'can', 'this', 'that', 'these', 'those', 'we', 'our',
                'us', 'you', 'your', 'they', 'their', 'them', 'it', 'its'
            }
            
            filtered_words = [
                word for word in words 
                if len(word) > 2 and word not in stop_words
            ]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Return top keywords
            return [word for word, count in word_counts.most_common(20)]
            
        except Exception:
            return []
    
    def calculate_readability(self, text: str) -> float:
        """
        Calculate text readability score using Flesch Reading Ease formula.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Readability score (0-100, higher is more readable)
        """
        if not text or not text.strip():
            return 0.0
            
        try:
            # Count sentences, words, and syllables
            sentences = len(re.findall(r'[.!?]+', text))
            words = len(text.split())
            
            if sentences == 0 or words == 0:
                return 0.0
            
            # Estimate syllables (simple approximation)
            syllables = self._count_syllables(text)
            
            # Flesch Reading Ease formula
            if sentences > 0 and words > 0:
                score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
                return max(0.0, min(100.0, score))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _count_syllables(self, text: str) -> int:
        """
        Estimate syllable count in text (simple approximation).
        
        Args:
            text: Input text
            
        Returns:
            Estimated syllable count
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        syllable_count = 0
        
        for word in words:
            # Count vowel groups
            vowels = 'aeiouy'
            syllables_in_word = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllables_in_word += 1
                prev_was_vowel = is_vowel
            
            # Handle silent e
            if word.endswith('e') and syllables_in_word > 1:
                syllables_in_word -= 1
            
            # Ensure at least 1 syllable per word
            syllables_in_word = max(1, syllables_in_word)
            syllable_count += syllables_in_word
        
        return syllable_count
    
    def detect_confidence_indicators(self, text: str) -> List[str]:
        """
        Detect confidence indicators in business language.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected confidence indicators
        """
        if not text or not text.strip():
            return []
            
        try:
            text_lower = text.lower()
            detected_indicators = []
            
            for indicator in self.confidence_indicators:
                if indicator.lower() in text_lower:
                    detected_indicators.append(indicator)
            
            return detected_indicators
            
        except Exception:
            return []
    
    def extract_metrics(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numerical metrics and growth indicators from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted metrics with type and value information
        """
        if not text or not text.strip():
            return []
            
        try:
            metrics = []
            
            # Extract metrics by category
            for category, patterns in self.metrics_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        metric_value = match.group(1)
                        context = text[max(0, match.start()-50):match.end()+50]
                        
                        metrics.append({
                            'value': metric_value,
                            'type': category,
                            'context': context.strip(),
                            'position': match.start(),
                            'raw_match': match.group(0)
                        })
            
            # Extract team experience metrics
            team_metrics = self._extract_team_metrics(text)
            metrics.extend(team_metrics)
            
            # Extract traction signals
            traction_signals = self._extract_traction_signals(text)
            metrics.extend(traction_signals)
            
            # Remove duplicates and sort by position
            metrics = self._deduplicate_metrics(metrics)
            metrics.sort(key=lambda x: x['position'])
            
            return metrics
            
        except Exception:
            return []
    

    
    def _extract_team_metrics(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract team experience indicators and domain expertise from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of team-related metrics
        """
        team_metrics = []
        
        try:
            for pattern in self.team_experience_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    context = text[max(0, match.start()-40):match.end()+40]
                    metric_type = self._classify_team_metric(pattern, context)
                    
                    team_metrics.append({
                        'value': match.group(1),
                        'type': metric_type,
                        'context': context.strip(),
                        'position': match.start(),
                        'raw_match': match.group(0)
                    })
        
        except Exception:
            pass
        
        return team_metrics
    
    def _classify_team_metric(self, pattern: str, context: str) -> str:
        """
        Classify team metric based on pattern and context.
        
        Args:
            pattern: Regex pattern that matched
            context: Text context around the match
            
        Returns:
            Team metric classification
        """
        pattern_lower = pattern.lower()
        context_lower = context.lower()
        
        if 'years' in pattern_lower:
            return 'experience_years'
        elif any(word in pattern_lower for word in ['university', 'college', 'phd', 'mba', 'degree']):
            return 'education'
        elif any(word in context_lower for word in ['founder', 'ceo', 'cto', 'vp', 'director']):
            return 'leadership_background'
        elif any(word in pattern_lower for word in ['expert', 'specialist', 'expertise', 'authority']):
            return 'domain_expertise'
        elif any(word in pattern_lower for word in ['recognized', 'awarded', 'speaker', 'keynote']):
            return 'industry_recognition'
        else:
            return 'background_company'
    
    def _extract_traction_signals(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract traction signals including MoM growth, user numbers, and revenue indicators.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of traction signal metrics
        """
        traction_signals = []
        
        # Specific traction signal patterns
        traction_patterns = [
            # Month-over-month and year-over-year growth
            r'(\d+(?:\.\d+)?)\s*%\s*(?:MoM|month-over-month|monthly)\s*(?:growth|increase)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:YoY|year-over-year|yearly|annual)\s*(?:growth|increase)',
            
            # User acquisition and engagement
            r'(?:acquired|gained|added)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:new\s*)?(?:users|customers)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:user\s*)?(?:retention|engagement|activation)',
            
            # Revenue traction
            r'(?:reached|achieved|hit)\s*\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:in\s*)?(?:revenue|ARR|MRR)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:revenue\s*)?(?:growth|increase)',
            
            # Partnership and distribution
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:partners|partnerships|integrations|channels)',
            r'(?:launched\s*in|available\s*in|expanded\s*to)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:countries|markets|cities)',
            
            # Product metrics
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:downloads|installs|sign-ups)',
            r'(\d+(?:\.\d+)?)\s*(?:star|stars)\s*(?:rating|review)',
            
            # Funding and validation
            r'(?:raised|secured|closed)\s*\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:seed|series|round)',
            r'(?:backed\s*by|invested\s*by|supported\s*by)\s*([A-Z][a-zA-Z\s&]+(?:Capital|Ventures|Partners|Fund))',
        ]
        
        try:
            for pattern in traction_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    metric_value = match.group(1)
                    context = text[max(0, match.start()-40):match.end()+40]
                    
                    # Classify traction signal type
                    signal_type = self._classify_traction_signal(context, pattern)
                    
                    traction_signals.append({
                        'value': metric_value,
                        'type': f'traction_{signal_type}',
                        'context': context.strip(),
                        'position': match.start(),
                        'raw_match': match.group(0)
                    })
        
        except Exception:
            pass
        
        return traction_signals
    
    def _classify_traction_signal(self, context: str, pattern: str) -> str:
        """
        Classify the type of traction signal based on context and pattern.
        
        Args:
            context: Text context around the signal
            pattern: Regex pattern that matched
            
        Returns:
            Traction signal classification
        """
        context_lower = context.lower()
        pattern_lower = pattern.lower()
        
        if any(word in context_lower for word in ['mom', 'month-over-month', 'yoy', 'year-over-year']):
            return 'growth_rate'
        elif any(word in context_lower for word in ['users', 'customers', 'acquired', 'gained']):
            return 'user_acquisition'
        elif any(word in context_lower for word in ['revenue', 'arr', 'mrr', 'sales']):
            return 'revenue_growth'
        elif any(word in context_lower for word in ['retention', 'engagement', 'activation']):
            return 'engagement'
        elif any(word in context_lower for word in ['partners', 'partnerships', 'integrations']):
            return 'partnerships'
        elif any(word in context_lower for word in ['countries', 'markets', 'cities', 'launched']):
            return 'market_expansion'
        elif any(word in context_lower for word in ['downloads', 'installs', 'sign-ups']):
            return 'product_adoption'
        elif any(word in context_lower for word in ['raised', 'secured', 'funding', 'investment']):
            return 'funding'
        elif any(word in context_lower for word in ['rating', 'review', 'star']):
            return 'product_quality'
        else:
            return 'general'
    
    def _deduplicate_metrics(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate metrics based on value and position proximity.
        
        Args:
            metrics: List of extracted metrics
            
        Returns:
            Deduplicated list of metrics
        """
        if not metrics:
            return []
        
        # Sort by position first
        metrics.sort(key=lambda x: x['position'])
        
        deduplicated = []
        for metric in metrics:
            # Check if this metric is too similar to existing ones
            is_duplicate = False
            for existing in deduplicated:
                # Same value and close position (within 20 characters)
                if (metric['value'] == existing['value'] and 
                    abs(metric['position'] - existing['position']) < 20):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(metric)
        
        return deduplicated