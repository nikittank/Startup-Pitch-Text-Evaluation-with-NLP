"""
Industry classification system for startup pitch decks.
Categorizes startups into industry categories based on keyword analysis.
"""
import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class IndustryClassification:
    """Result of industry classification analysis."""
    primary_industry: str
    secondary_industries: List[str]
    confidence_score: float
    matched_keywords: Dict[str, List[str]]


class IndustryClassifier:
    """
    Keyword-based industry categorizer for startup pitch decks.
    Supports multiple industry tags per deck with confidence scoring.
    """
    
    def __init__(self):
        """Initialize the industry classifier with keyword mappings."""
        self.industry_keywords = {
            'Fintech': [
                'financial', 'finance', 'banking', 'payment', 'payments', 'fintech',
                'cryptocurrency', 'crypto', 'blockchain', 'lending', 'loan', 'loans',
                'credit', 'investment', 'investing', 'trading', 'wallet', 'money',
                'transaction', 'transactions', 'remittance', 'insurance', 'insurtech',
                'wealth management', 'robo-advisor', 'peer-to-peer', 'p2p', 'defi',
                'neobank', 'digital bank', 'mobile banking', 'payroll', 'accounting',
                'tax', 'compliance', 'regulatory', 'kyc', 'aml'
            ],
            'HealthTech': [
                'health', 'healthcare', 'medical', 'medicine', 'hospital', 'clinic',
                'patient', 'patients', 'doctor', 'doctors', 'physician', 'nurse',
                'telemedicine', 'telehealth', 'healthtech', 'medtech', 'biotech',
                'pharmaceutical', 'drug', 'drugs', 'therapy', 'treatment', 'diagnosis',
                'diagnostic', 'wellness', 'fitness', 'mental health', 'therapy',
                'rehabilitation', 'surgery', 'surgical', 'medical device', 'wearable',
                'health monitoring', 'clinical', 'trial', 'fda', 'medical records',
                'ehr', 'emr', 'genomics', 'precision medicine'
            ],
            'SaaS': [
                'saas', 'software as a service', 'cloud', 'platform', 'api', 'apis',
                'enterprise', 'b2b', 'business software', 'productivity', 'workflow',
                'automation', 'crm', 'erp', 'hr software', 'project management',
                'collaboration', 'communication', 'analytics', 'dashboard', 'reporting',
                'integration', 'subscription', 'recurring revenue', 'mrr', 'arr',
                'churn', 'retention', 'onboarding', 'user management', 'admin',
                'scalable', 'multi-tenant', 'white-label', 'customizable'
            ],
            'B2C': [
                'consumer', 'b2c', 'retail', 'e-commerce', 'ecommerce', 'marketplace',
                'shopping', 'mobile app', 'social', 'social media', 'community',
                'entertainment', 'gaming', 'games', 'streaming', 'content', 'media',
                'lifestyle', 'fashion', 'food', 'delivery', 'travel', 'booking',
                'reservation', 'dating', 'social network', 'user-generated',
                'viral', 'engagement', 'daily active users', 'dau', 'mau',
                'freemium', 'advertising', 'monetization', 'consumer behavior'
            ],
            'EdTech': [
                'education', 'educational', 'learning', 'teaching', 'student', 'students',
                'teacher', 'teachers', 'school', 'schools', 'university', 'college',
                'course', 'courses', 'curriculum', 'training', 'skill', 'skills',
                'online learning', 'e-learning', 'edtech', 'mooc', 'lms',
                'learning management', 'assessment', 'certification', 'degree',
                'academic', 'classroom', 'virtual classroom', 'tutoring', 'mentoring'
            ],
            'PropTech': [
                'real estate', 'property', 'properties', 'housing', 'rental', 'rent',
                'lease', 'landlord', 'tenant', 'proptech', 'construction', 'building',
                'architecture', 'home', 'residential', 'commercial', 'office space',
                'coworking', 'facility management', 'smart building', 'iot',
                'property management', 'listing', 'mls', 'mortgage', 'appraisal'
            ],
            'Mobility': [
                'transportation', 'transport', 'mobility', 'automotive', 'car', 'cars',
                'vehicle', 'vehicles', 'ride', 'rideshare', 'ridesharing', 'taxi',
                'uber', 'lyft', 'scooter', 'bike', 'bicycle', 'electric vehicle',
                'ev', 'autonomous', 'self-driving', 'logistics', 'delivery',
                'shipping', 'freight', 'supply chain', 'fleet', 'navigation',
                'mapping', 'gps', 'traffic', 'parking'
            ],
            'FoodTech': [
                'food', 'restaurant', 'restaurants', 'dining', 'meal', 'meals',
                'delivery', 'food delivery', 'kitchen', 'cooking', 'recipe', 'recipes',
                'grocery', 'groceries', 'agriculture', 'farming', 'farm', 'organic',
                'nutrition', 'diet', 'foodtech', 'culinary', 'beverage', 'drink',
                'catering', 'hospitality', 'menu', 'ordering', 'takeout'
            ]
        }
        
        # Compile regex patterns for efficient matching
        self.compiled_patterns = {}
        for industry, keywords in self.industry_keywords.items():
            pattern = r'\b(?:' + '|'.join(re.escape(keyword.lower()) for keyword in keywords) + r')\b'
            self.compiled_patterns[industry] = re.compile(pattern, re.IGNORECASE)
    
    def classify_industry(self, text: str) -> IndustryClassification:
        """
        Classify the industry of a startup based on pitch deck text.
        
        Args:
            text: Full text content of the pitch deck
            
        Returns:
            IndustryClassification with primary industry, secondary industries, and confidence
        """
        if not text or not text.strip():
            return IndustryClassification(
                primary_industry="Unknown",
                secondary_industries=[],
                confidence_score=0.0,
                matched_keywords={}
            )
        
        # Normalize text for analysis
        normalized_text = text.lower()
        
        # Count keyword matches for each industry
        industry_scores = {}
        matched_keywords = {}
        
        for industry, pattern in self.compiled_patterns.items():
            matches = pattern.findall(normalized_text)
            if matches:
                # Count unique matches (avoid double counting repeated keywords)
                unique_matches = list(set(matches))
                industry_scores[industry] = len(unique_matches)
                matched_keywords[industry] = unique_matches
            else:
                industry_scores[industry] = 0
                matched_keywords[industry] = []
        
        # Calculate total matches for normalization
        total_matches = sum(industry_scores.values())
        
        if total_matches == 0:
            return IndustryClassification(
                primary_industry="Unknown",
                secondary_industries=[],
                confidence_score=0.0,
                matched_keywords={}
            )
        
        # Sort industries by score
        sorted_industries = sorted(industry_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine primary industry (highest score)
        primary_industry = sorted_industries[0][0]
        primary_score = sorted_industries[0][1]
        
        # Determine secondary industries (score > 0 and at least 20% of primary score)
        secondary_threshold = max(1, primary_score * 0.2)
        secondary_industries = [
            industry for industry, score in sorted_industries[1:]
            if score >= secondary_threshold
        ]
        
        # Calculate confidence score based on match density and clarity
        confidence_score = self._calculate_confidence(
            primary_score, total_matches, len(secondary_industries), len(text)
        )
        
        return IndustryClassification(
            primary_industry=primary_industry,
            secondary_industries=secondary_industries,
            confidence_score=confidence_score,
            matched_keywords={k: v for k, v in matched_keywords.items() if v}
        )
    
    def _calculate_confidence(self, primary_score: int, total_matches: int, 
                            secondary_count: int, text_length: int) -> float:
        """
        Calculate confidence score for industry classification.
        
        Args:
            primary_score: Number of matches for primary industry
            total_matches: Total keyword matches across all industries
            secondary_count: Number of secondary industries identified
            text_length: Length of analyzed text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if total_matches == 0:
            return 0.0
        
        # Base confidence from primary industry dominance
        dominance_ratio = primary_score / total_matches
        base_confidence = dominance_ratio
        
        # Adjust for match density (matches per 1000 characters)
        match_density = (total_matches / max(text_length, 1)) * 1000
        density_factor = min(match_density / 5.0, 1.0)  # Cap at 5 matches per 1000 chars
        
        # Penalty for too many secondary industries (indicates ambiguity)
        ambiguity_penalty = max(0, secondary_count - 1) * 0.1
        
        # Bonus for having some matches but not too scattered
        if primary_score >= 3:
            clarity_bonus = 0.1
        else:
            clarity_bonus = 0.0
        
        # Final confidence calculation
        confidence = base_confidence * density_factor + clarity_bonus - ambiguity_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def get_supported_industries(self) -> List[str]:
        """Get list of supported industry categories."""
        return list(self.industry_keywords.keys())
    
    def add_industry_keywords(self, industry: str, keywords: List[str]) -> None:
        """
        Add or update keywords for an industry category.
        
        Args:
            industry: Industry name
            keywords: List of keywords to add
        """
        if industry in self.industry_keywords:
            self.industry_keywords[industry].extend(keywords)
        else:
            self.industry_keywords[industry] = keywords
        
        # Recompile pattern for this industry
        pattern = r'\b(?:' + '|'.join(re.escape(keyword.lower()) for keyword in self.industry_keywords[industry]) + r')\b'
        self.compiled_patterns[industry] = re.compile(pattern, re.IGNORECASE)