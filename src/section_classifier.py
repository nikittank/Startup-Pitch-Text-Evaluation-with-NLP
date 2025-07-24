"""
Section classifier for categorizing pitch deck content into structured sections.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from ..models.interfaces import SectionClassifierInterface
from ..config.config_manager import ConfigManager


class SectionClassifier(SectionClassifierInterface):
    """
    Keyword-based section classifier for pitch deck content.
    
    Categorizes text into standard pitch deck sections:
    - Problem Statement
    - Solution
    - Market Information
    - Traction Data
    - Team Information
    - Business Model
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize classifier with configuration."""
        self.config = config_manager
        self.keyword_mappings = {
            "problem": config_manager.get_keywords("problem"),
            "solution": config_manager.get_keywords("solution"),
            "market": config_manager.get_keywords("market"),
            "traction": config_manager.get_keywords("traction"),
            "team": config_manager.get_keywords("team"),
            "business_model": config_manager.get_keywords("business_model")
        }
        self.logger = logging.getLogger(__name__)
        
        # Extended keyword mappings for better section detection
        self._extended_keywords = {
            "problem": self.keyword_mappings["problem"] + [
                "pain", "need", "gap", "inefficiency", "broken", "lacking",
                "current state", "status quo", "existing", "traditional",
                "struggle", "face", "difficulty", "lose", "cost", "waste"
            ],
            "solution": self.keyword_mappings["solution"] + [
                "we built", "we created", "our product", "our platform",
                "introducing", "presents", "offers", "provides", "enables",
                "automated", "ai-powered", "technology", "system"
            ],
            "market": self.keyword_mappings["market"] + [
                "industry", "sector", "segment", "target market", "addressable market",
                "market size", "opportunity size", "$", "billion", "million", "b", "m",
                "global", "worldwide", "cagr", "growing"
            ],
            "traction": self.keyword_mappings["traction"] + [
                "progress", "milestones", "achievements", "results", "performance",
                "adoption", "engagement", "retention", "conversion", "sales",
                "arr", "mrr", "active users", "satisfaction", "partnerships"
            ],
            "team": self.keyword_mappings["team"] + [
                "founders", "leadership", "management", "advisors", "board",
                "co-founder", "executive", "staff", "employees", "talent",
                "former", "ex-", "years", "phd", "stanford", "google", "oracle"
            ],
            "business_model": self.keyword_mappings["business_model"] + [
                "revenue model", "pricing model", "go-to-market", "gtm",
                "sales strategy", "distribution", "channels", "partnerships",
                "saas", "subscription", "per month", "one-time", "premium"
            ]
        }
    
    def classify_sections(self, text: str) -> Dict[str, str]:
        """
        Categorize content into pitch deck sections.
        
        Args:
            text: Full pitch deck text content
            
        Returns:
            Dictionary mapping section names to extracted content
        """
        self.logger.info("Starting section classification")
        self.logger.debug(f"Input text length: {len(text)} characters")
        
        # Split text into paragraphs for analysis
        paragraphs = self._split_into_paragraphs(text)
        self.logger.debug(f"Split text into {len(paragraphs)} paragraphs")
        
        sections = {
            "problem": "",
            "solution": "",
            "market": "",
            "traction": "",
            "team": "",
            "business_model": ""
        }
        
        # Classify each paragraph
        for paragraph in paragraphs:
            if len(paragraph.strip()) < 20:  # Skip very short paragraphs
                self.logger.debug(f"Skipping short paragraph: {paragraph.strip()[:30]}...")
                continue
                
            section_scores = self._score_paragraph_for_sections(paragraph)
            best_section = max(section_scores.items(), key=lambda x: x[1])
            
            # Only assign if confidence is above threshold
            if best_section[1] > 0.15:  # Lower threshold for better detection
                if sections[best_section[0]]:
                    sections[best_section[0]] += "\n\n" + paragraph
                else:
                    sections[best_section[0]] = paragraph
                    
                self.logger.debug(f"Assigned paragraph to {best_section[0]} (confidence: {best_section[1]:.2f})")
            else:
                self.logger.debug(f"Paragraph below confidence threshold ({best_section[1]:.2f}): {paragraph[:50]}...")
        
        # Apply fallback logic for missing sections
        empty_sections_before = [name for name, content in sections.items() if not content]
        sections = self._apply_fallback_logic(text, sections)
        empty_sections_after = [name for name, content in sections.items() if not content]
        
        if empty_sections_before != empty_sections_after:
            recovered_sections = set(empty_sections_before) - set(empty_sections_after)
            self.logger.info(f"Fallback logic recovered sections: {recovered_sections}")
        
        populated_sections = [name for name, content in sections.items() if content]
        self.logger.info(f"Section classification complete. Found {len(populated_sections)} sections: {populated_sections}")
        
        if empty_sections_after:
            self.logger.warning(f"Missing sections: {empty_sections_after}")
        
        return sections
    
    def extract_problem_statement(self, text: str) -> str:
        """Extract problem statement section."""
        return self._extract_section_by_keywords(text, "problem")
    
    def extract_solution(self, text: str) -> str:
        """Extract solution section."""
        return self._extract_section_by_keywords(text, "solution")
    
    def extract_market_info(self, text: str) -> str:
        """Extract market information section."""
        return self._extract_section_by_keywords(text, "market")
    
    def extract_traction_data(self, text: str) -> str:
        """Extract traction data section."""
        return self._extract_section_by_keywords(text, "traction")
    
    def extract_team_info(self, text: str) -> str:
        """Extract team information section."""
        return self._extract_section_by_keywords(text, "team")
    
    def extract_business_model(self, text: str) -> str:
        """Extract business model section."""
        return self._extract_section_by_keywords(text, "business_model")
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs."""
        # Split by double newlines or slide breaks, but also handle headers
        paragraphs = re.split(r'\n\s*\n|\f|\r\n\r\n', text)
        
        # Clean and filter paragraphs, combining headers with following content
        cleaned_paragraphs = []
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()
            
            # Skip very short content
            if len(para) < 10:
                i += 1
                continue
            
            # Check if this looks like a header (short line ending with colon)
            if (len(para) < 50 and 
                (para.endswith(':') or para.isupper() or 
                 any(header in para.lower() for header in ['problem', 'solution', 'market', 'team', 'traction', 'business']))):
                
                # Try to combine with next paragraph
                if i + 1 < len(paragraphs):
                    next_para = paragraphs[i + 1].strip()
                    if len(next_para) > 10:
                        combined = para + "\n" + next_para
                        cleaned_paragraphs.append(combined)
                        i += 2
                        continue
                
            cleaned_paragraphs.append(para)
            i += 1
        
        return cleaned_paragraphs
    
    def _score_paragraph_for_sections(self, paragraph: str) -> Dict[str, float]:
        """Score a paragraph's relevance to each section."""
        paragraph_lower = paragraph.lower()
        scores = {}
        
        # Check for section headers first (strong signal)
        header_bonus = {}
        section_headers = {
            "problem": ["problem statement", "problem:", "the problem", "current problem", "problem"],
            "solution": ["solution:", "our solution", "the solution", "what we do", "solution"],
            "market": ["market:", "market opportunity", "market size", "tam", "market"],
            "traction": ["traction:", "key metrics", "growth", "results", "traction"],
            "team": ["team:", "our team", "about us", "founders", "team"],
            "business_model": ["business model:", "revenue model", "how we make money", "business model"]
        }
        
        for section, headers in section_headers.items():
            header_bonus[section] = 0.0
            for header in headers:
                if header in paragraph_lower:
                    header_bonus[section] = 10.0  # Extremely strong bonus for headers
                    break
        
        for section, keywords in self._extended_keywords.items():
            score = 0.0
            word_count = len(paragraph_lower.split())
            
            for keyword in keywords:
                # Count keyword occurrences
                count = paragraph_lower.count(keyword.lower())
                if count > 0:
                    # Weight by keyword importance and frequency
                    keyword_weight = 2.0 if keyword in self.keyword_mappings[section] else 1.0
                    score += count * keyword_weight
            
            # Add header bonus
            score += header_bonus[section]
            
            # Normalize by paragraph length but give minimum boost for any matches
            if score > 0:
                scores[section] = (score / max(word_count, 1)) + 0.1  # Base boost for matches
            else:
                scores[section] = 0.0
        
        return scores
    
    def _extract_section_by_keywords(self, text: str, section_type: str) -> str:
        """Extract specific section using keyword matching."""
        if section_type not in self._extended_keywords:
            return ""
        
        keywords = self._extended_keywords[section_type]
        paragraphs = self._split_into_paragraphs(text)
        
        best_paragraph = ""
        best_score = 0.0
        
        for paragraph in paragraphs:
            score = 0.0
            paragraph_lower = paragraph.lower()
            
            for keyword in keywords:
                if keyword.lower() in paragraph_lower:
                    score += 1.0 if keyword in self.keyword_mappings[section_type] else 0.7
            
            if score > best_score:
                best_score = score
                best_paragraph = paragraph
        
        return best_paragraph if best_score > 0 else ""
    
    def _apply_fallback_logic(self, full_text: str, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Apply fallback logic for non-standard deck structures.
        
        Uses positional heuristics and alternative keywords when standard
        classification fails.
        """
        self.logger.debug("Applying fallback logic for missing sections")
        text_lower = full_text.lower()
        
        # Fallback for problem section
        if not sections["problem"]:
            problem_indicators = [
                "current situation", "today", "currently", "existing solution",
                "why now", "market problem", "customer pain"
            ]
            fallback_content = self._find_content_by_indicators(full_text, problem_indicators)
            if fallback_content:
                sections["problem"] = fallback_content
                self.logger.debug("Fallback logic found problem section using alternative indicators")
        
        # Fallback for solution section  
        if not sections["solution"]:
            solution_indicators = [
                "our approach", "what we do", "how it works", "our technology",
                "product overview", "key features", "value proposition"
            ]
            sections["solution"] = self._find_content_by_indicators(full_text, solution_indicators)
        
        # Fallback for market section
        if not sections["market"]:
            market_indicators = [
                "market opportunity", "target customer", "customer segment",
                "total addressable", "serviceable", "competitive landscape"
            ]
            sections["market"] = self._find_content_by_indicators(full_text, market_indicators)
        
        # Fallback for traction section
        if not sections["traction"]:
            traction_indicators = [
                "key metrics", "growth metrics", "user base", "customer base",
                "revenue growth", "monthly active", "daily active", "retention rate"
            ]
            sections["traction"] = self._find_content_by_indicators(full_text, traction_indicators)
        
        # Fallback for team section
        if not sections["team"]:
            team_indicators = [
                "about us", "our team", "leadership team", "founding team",
                "management", "key personnel", "advisors", "board members"
            ]
            sections["team"] = self._find_content_by_indicators(full_text, team_indicators)
        
        # Fallback for business model section
        if not sections["business_model"]:
            model_indicators = [
                "how we make money", "revenue streams", "pricing strategy",
                "unit economics", "customer acquisition", "sales process"
            ]
            sections["business_model"] = self._find_content_by_indicators(full_text, model_indicators)
        
        return sections
    
    def _find_content_by_indicators(self, text: str, indicators: List[str]) -> str:
        """Find content using alternative indicators."""
        paragraphs = self._split_into_paragraphs(text)
        
        best_paragraph = ""
        best_score = 0
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            score = 0
            for indicator in indicators:
                if indicator.lower() in paragraph_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_paragraph = paragraph
        
        return best_paragraph
    
    def get_classification_confidence(self, text: str, sections: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate confidence scores for section classifications.
        
        Args:
            text: Original full text
            sections: Classified sections
            
        Returns:
            Dictionary mapping section names to confidence scores (0-1)
        """
        confidence_scores = {}
        
        for section_name, section_content in sections.items():
            if not section_content:
                confidence_scores[section_name] = 0.0
                continue
            
            # Calculate confidence based on keyword density and content length
            keywords = self._extended_keywords.get(section_name, [])
            content_lower = section_content.lower()
            
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            keyword_density = keyword_matches / len(keywords) if keywords else 0
            
            # Content length factor (longer content generally more confident)
            length_factor = min(len(section_content.split()) / 50, 1.0)
            
            # Combined confidence score
            confidence_scores[section_name] = (keyword_density * 0.7 + length_factor * 0.3)
        
        return confidence_scores