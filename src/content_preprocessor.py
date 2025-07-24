"""
Content preprocessing pipeline for cleaning and normalizing extracted text.
"""
import re
import logging
from typing import Dict, Any, List
import unicodedata

from ..models.interfaces import ContentPreprocessorInterface


class ContentPreprocessor(ContentPreprocessorInterface):
    """Content preprocessor for cleaning and normalizing extracted text."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common PDF artifacts to remove
        self.pdf_artifacts = [
            r'\x0c',  # Form feed characters
            r'\x00',  # Null characters
            r'[\x01-\x08\x0B\x0E-\x1F\x7F]',  # Control characters
            r'[^\x00-\x7F]+',  # Non-ASCII characters (optional, might remove legitimate content)
        ]
        
        # Patterns for cleaning formatting
        self.formatting_patterns = [
            (r'\.{3,}', '...'),  # Multiple dots to ellipsis
            (r'-{3,}', '---'),   # Multiple dashes to em dash equivalent
            (r'_{3,}', '___'),   # Multiple underscores
            (r'={3,}', '==='),   # Multiple equals signs
            (r'\*{3,}', '***'),  # Multiple asterisks
        ]
        
        # Common business/pitch deck terms for quality assessment
        self.business_keywords = [
            'market', 'revenue', 'growth', 'customer', 'product', 'solution',
            'problem', 'team', 'funding', 'investment', 'traction', 'business',
            'model', 'strategy', 'competitive', 'advantage', 'opportunity'
        ]
    
    def clean_text(self, raw_text: str) -> str:
        """
        Clean and normalize extracted text content.
        
        Args:
            raw_text: Raw text extracted from PDF
            
        Returns:
            Cleaned and normalized text
        """
        if not raw_text:
            return ""
            
        text = raw_text
        
        # Step 1: Remove PDF artifacts
        text = self.remove_artifacts(text)
        
        # Step 2: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 3: Clean formatting
        text = self._clean_formatting(text)
        
        # Step 4: Normalize unicode
        text = self._normalize_unicode(text)
        
        # Step 5: Remove excessive punctuation
        text = self._clean_punctuation(text)
        
        return text.strip()
    
    def remove_artifacts(self, text: str) -> str:
        """
        Remove PDF formatting artifacts.
        
        Args:
            text: Text containing potential artifacts
            
        Returns:
            Text with artifacts removed
        """
        if not text:
            return ""
            
        cleaned_text = text
        
        # Remove common PDF artifacts
        for artifact_pattern in self.pdf_artifacts:
            cleaned_text = re.sub(artifact_pattern, '', cleaned_text)
        
        # Remove page numbers and headers/footers (common patterns)
        page_patterns = [
            r'^\d+\s*$',  # Standalone page numbers
            r'Page\s+\d+',  # "Page X" patterns
            r'\d+\s*/\s*\d+',  # "X / Y" page patterns
        ]
        
        lines = cleaned_text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            is_artifact = False
            
            # Check if line matches page number patterns
            for pattern in page_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_artifact = True
                    break
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3 and not line.isalnum():
                is_artifact = True
            
            if not is_artifact:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace and formatting.
        
        Args:
            text: Text with irregular whitespace
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return ""
        
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks - convert multiple line breaks to double line break
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = text.split('\n')
        normalized_lines = [line.rstrip() for line in lines]
        
        # Remove empty lines at the beginning and end
        while normalized_lines and not normalized_lines[0].strip():
            normalized_lines.pop(0)
        while normalized_lines and not normalized_lines[-1].strip():
            normalized_lines.pop()
        
        return '\n'.join(normalized_lines)
    
    def _clean_formatting(self, text: str) -> str:
        """Clean excessive formatting characters."""
        cleaned_text = text
        
        for pattern, replacement in self.formatting_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        return cleaned_text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to standard forms."""
        # Normalize unicode to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common unicode characters with ASCII equivalents
        unicode_replacements = {
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '–': '-',  # En dash
            '—': '--', # Em dash
            '…': '...', # Horizontal ellipsis
            '•': '*',  # Bullet point
            '→': '->',  # Right arrow
            '←': '<-',  # Left arrow
        }
        
        for unicode_char, ascii_replacement in unicode_replacements.items():
            text = text.replace(unicode_char, ascii_replacement)
        
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean excessive or problematic punctuation."""
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamation marks
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'[,]{2,}', ',', text)  # Multiple commas
        text = re.sub(r'[;]{2,}', ';', text)  # Multiple semicolons
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', text)  # Add space between punctuation
        
        return text
    
    def assess_text_quality(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """
        Assess text quality and measure extraction completeness.
        
        Args:
            original_text: Original extracted text
            cleaned_text: Cleaned and processed text
            
        Returns:
            Dictionary with quality assessment metrics
        """
        quality_metrics = {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'compression_ratio': 0.0,
            'word_count': 0,
            'business_keyword_count': 0,
            'estimated_quality': 0.0,
            'has_meaningful_content': False,
            'potential_issues': []
        }
        
        if not cleaned_text:
            quality_metrics['potential_issues'].append("No content after cleaning")
            return quality_metrics
        
        # Calculate compression ratio
        if quality_metrics['original_length'] > 0:
            quality_metrics['compression_ratio'] = quality_metrics['cleaned_length'] / quality_metrics['original_length']
        
        # Count words
        words = cleaned_text.lower().split()
        quality_metrics['word_count'] = len(words)
        
        # Count business-related keywords
        word_set = set(words)
        business_matches = sum(1 for keyword in self.business_keywords if keyword in word_set)
        quality_metrics['business_keyword_count'] = business_matches
        
        # Assess if content is meaningful
        if quality_metrics['word_count'] >= 50 and business_matches >= 3:
            quality_metrics['has_meaningful_content'] = True
        
        # Identify potential issues
        if quality_metrics['compression_ratio'] < 0.5:
            quality_metrics['potential_issues'].append("High compression ratio - possible over-cleaning")
        
        if quality_metrics['word_count'] < 100:
            quality_metrics['potential_issues'].append("Low word count - possible incomplete extraction")
        
        if business_matches < 2:
            quality_metrics['potential_issues'].append("Few business keywords - content may not be pitch deck")
        
        # Calculate overall quality score (0-1)
        length_score = min(quality_metrics['word_count'] / 500, 1.0)  # Normalize to 500 words
        keyword_score = min(business_matches / 10, 1.0)  # Normalize to 10 keywords
        compression_score = 1.0 if 0.7 <= quality_metrics['compression_ratio'] <= 1.0 else 0.5
        
        quality_metrics['estimated_quality'] = (length_score * 0.4 + keyword_score * 0.4 + compression_score * 0.2)
        
        return quality_metrics
    
    def extract_sections_hints(self, text: str) -> Dict[str, List[str]]:
        """
        Extract potential section indicators for downstream classification.
        
        Args:
            text: Cleaned text content
            
        Returns:
            Dictionary mapping section types to potential indicators found
        """
        section_hints = {
            'problem': [],
            'solution': [],
            'market': [],
            'traction': [],
            'team': [],
            'business_model': [],
            'funding': []
        }
        
        # Define section indicator patterns
        section_patterns = {
            'problem': [
                r'problem\s+statement', r'the\s+problem', r'pain\s+point',
                r'challenge', r'issue', r'difficulty'
            ],
            'solution': [
                r'our\s+solution', r'the\s+solution', r'how\s+we\s+solve',
                r'product', r'platform', r'technology'
            ],
            'market': [
                r'market\s+size', r'tam', r'sam', r'som', r'addressable\s+market',
                r'market\s+opportunity', r'target\s+market'
            ],
            'traction': [
                r'traction', r'growth', r'users?', r'customers?', r'revenue',
                r'metrics', r'kpis?', r'month\s+over\s+month', r'mom'
            ],
            'team': [
                r'team', r'founders?', r'ceo', r'cto', r'experience',
                r'background', r'expertise'
            ],
            'business_model': [
                r'business\s+model', r'revenue\s+model', r'monetization',
                r'pricing', r'subscription', r'freemium'
            ],
            'funding': [
                r'funding', r'investment', r'raise', r'round', r'capital',
                r'investors?', r'valuation'
            ]
        }
        
        text_lower = text.lower()
        
        for section, patterns in section_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    section_hints[section].extend(matches)
        
        return section_hints