"""
Multi-dimensional scorer for pitch deck evaluation.
Implements scoring across multiple business dimensions including problem clarity,
market potential, traction strength, team experience, business model, vision/moat,
and overall confidence.
"""
import re
import math
from typing import Dict, List, Any, Optional
from ..models.interfaces import MultiDimensionalScorerInterface
from ..models.data_models import ScoringConfiguration
from ..config.config_manager import ConfigManager
from ..analyzers.nlp_analyzer import NLPAnalyzer


class MultiDimensionalScorer(MultiDimensionalScorerInterface):
    """
    Multi-dimensional scorer that evaluates pitch decks across various business dimensions.
    Uses NLP analysis, keyword matching, and rule-based scoring to generate objective scores.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the multi-dimensional scorer.
        
        Args:
            config_manager: Configuration manager for scoring parameters
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.load_config()
        self.nlp_analyzer = NLPAnalyzer()
        
        # Initialize scoring components
        self._init_scoring_patterns()
    
    def _init_scoring_patterns(self):
        """Initialize patterns and keywords for scoring different dimensions."""
        
        # Problem clarity indicators
        self.problem_clarity_indicators = {
            'specificity': [
                'specific', 'particular', 'exact', 'precise', 'detailed',
                'clearly defined', 'well-defined', 'identified', 'pinpointed'
            ],
            'quantification': [
                r'\d+(?:\.\d+)?%', r'\$\d+(?:,\d{3})*(?:\.\d+)?', r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|M|B|k)',
                r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:users|customers|people|hours|days|minutes)'
            ],
            'pain_indicators': [
                'pain', 'problem', 'challenge', 'difficulty', 'struggle', 'frustration',
                'inefficient', 'costly', 'time-consuming', 'manual', 'broken', 'outdated'
            ],
            'urgency': [
                'urgent', 'critical', 'immediate', 'pressing', 'essential', 'crucial',
                'must', 'need', 'required', 'necessary', 'vital', 'important'
            ]
        }
        
        # Market potential indicators
        self.market_potential_indicators = {
            'size_metrics': [
                r'(?:TAM|total\s+addressable\s+market).*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)',
                r'(?:SAM|serviceable\s+addressable\s+market).*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)',
                r'(?:SOM|serviceable\s+obtainable\s+market).*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)',
                r'market\s+size.*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|B|M)\s+market'
            ],
            'growth_indicators': [
                'growing', 'expanding', 'increasing', 'rising', 'booming', 'emerging',
                'fast-growing', 'rapidly growing', 'accelerating', 'scaling'
            ],
            'opportunity_keywords': [
                'opportunity', 'potential', 'addressable', 'untapped', 'underserved',
                'whitespace', 'gap', 'niche', 'segment', 'vertical'
            ],
            'validation': [
                'validated', 'proven', 'demonstrated', 'confirmed', 'established',
                'research shows', 'studies indicate', 'data suggests', 'evidence'
            ]
        }
        
        # Traction strength indicators
        self.traction_indicators = {
            'growth_metrics': [
                r'(\d+(?:\.\d+)?)\s*%\s*(?:MoM|month-over-month|monthly)\s*(?:growth|increase)',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:YoY|year-over-year|yearly|annual)\s*(?:growth|increase)',
                r'(\d+(?:\.\d+)?)\s*x\s*(?:growth|increase|faster|more)',
                r'(?:grew|increased|improved|rose)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%'
            ],
            'user_metrics': [
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:active\s*)?(?:users|customers|subscribers)',
                r'(?:DAU|daily\s+active\s+users).*?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?',
                r'(?:MAU|monthly\s+active\s+users).*?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?'
            ],
            'revenue_metrics': [
                r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?\s*(?:in\s*)?(?:revenue|ARR|MRR)',
                r'(?:revenue|ARR|MRR).*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|k|thousand)?',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:revenue\s*)?(?:growth|increase)'
            ],
            'engagement_metrics': [
                r'(\d+(?:\.\d+)?)\s*%\s*(?:retention|engagement|conversion)',
                r'(?:retention|engagement|conversion)\s*(?:rate\s*)?(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*(?:NPS|Net\s*Promoter\s*Score)'
            ]
        }

    def score_problem_clarity(self, problem_text: str) -> float:
        """
        Score problem clarity based on specificity and definition.
        
        Evaluates:
        - Specificity of problem description
        - Quantification of problem impact
        - Clear pain point identification
        - Urgency indicators
        
        Args:
            problem_text: Text content describing the problem
            
        Returns:
            Problem clarity score (0-10 scale)
        """
        if not problem_text or not problem_text.strip():
            return 0.0
        
        score = 0.0
        text_lower = problem_text.lower()
        
        # Base score from text length and structure (0-2 points)
        word_count = len(problem_text.split())
        if word_count >= 50:
            score += 2.0
        elif word_count >= 20:
            score += 1.5
        elif word_count >= 10:
            score += 1.0
        else:
            score += 0.5
        
        # Specificity indicators (0-2 points)
        specificity_count = sum(
            1 for indicator in self.problem_clarity_indicators['specificity']
            if indicator in text_lower
        )
        score += min(2.0, specificity_count * 0.4)
        
        # Quantification of problem impact (0-2 points)
        quantification_score = 0.0
        for pattern in self.problem_clarity_indicators['quantification']:
            if re.search(pattern, problem_text, re.IGNORECASE):
                quantification_score += 0.5
        score += min(2.0, quantification_score)
        
        # Pain point identification (0-2 points)
        pain_count = sum(
            1 for indicator in self.problem_clarity_indicators['pain_indicators']
            if indicator in text_lower
        )
        score += min(2.0, pain_count * 0.3)
        
        # Urgency indicators (0-2 points)
        urgency_count = sum(
            1 for indicator in self.problem_clarity_indicators['urgency']
            if indicator in text_lower
        )
        score += min(2.0, urgency_count * 0.4)
        
        # Sentiment analysis bonus (0-0.5 points)
        sentiment = self.nlp_analyzer.analyze_sentiment(problem_text)
        if sentiment < -0.2:  # Negative sentiment indicates problem awareness
            score += 0.5
        
        return min(10.0, score)

    def score_market_potential(self, market_text: str) -> float:
        """
        Score market potential using TAM/SAM mentions and quantified data.
        
        Evaluates:
        - Market size quantification (TAM/SAM/SOM)
        - Growth indicators and trends
        - Market opportunity identification
        - Validation and research backing
        
        Args:
            market_text: Text content describing market information
            
        Returns:
            Market potential score (0-10 scale)
        """
        if not market_text or not market_text.strip():
            return 0.0
        
        score = 0.0
        text_lower = market_text.lower()
        
        # Base score from content presence (0-1 point)
        word_count = len(market_text.split())
        if word_count >= 30:
            score += 1.0
        elif word_count >= 15:
            score += 0.7
        elif word_count >= 5:
            score += 0.4
        
        # Market size quantification (0-4 points)
        size_score = 0.0
        for pattern in self.market_potential_indicators['size_metrics']:
            matches = re.findall(pattern, market_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract numeric value
                    value_str = match if isinstance(match, str) else match[0] if match else "0"
                    value = float(re.sub(r'[^\d.]', '', value_str))
                    
                    # Score based on market size
                    if value >= 100:  # $100B+ market
                        size_score += 2.0
                    elif value >= 10:  # $10B+ market
                        size_score += 1.5
                    elif value >= 1:  # $1B+ market
                        size_score += 1.0
                    else:
                        size_score += 0.5
                except (ValueError, IndexError):
                    size_score += 0.3  # Partial credit for mentioning metrics
        
        score += min(4.0, size_score)
        
        # Growth indicators (0-2 points)
        growth_count = sum(
            1 for indicator in self.market_potential_indicators['growth_indicators']
            if indicator in text_lower
        )
        score += min(2.0, growth_count * 0.4)
        
        # Opportunity identification (0-2 points)
        opportunity_count = sum(
            1 for keyword in self.market_potential_indicators['opportunity_keywords']
            if keyword in text_lower
        )
        score += min(2.0, opportunity_count * 0.3)
        
        # Validation and research backing (0-1 point)
        validation_count = sum(
            1 for indicator in self.market_potential_indicators['validation']
            if indicator in text_lower
        )
        score += min(1.0, validation_count * 0.25)
        
        return min(10.0, score)

    def score_traction_strength(self, traction_text: str) -> float:
        """
        Score traction strength based on growth metrics and user numbers.
        
        Evaluates:
        - Growth rate metrics (MoM, YoY)
        - User acquisition and engagement
        - Revenue traction
        - Key performance indicators
        
        Args:
            traction_text: Text content describing traction data
            
        Returns:
            Traction strength score (0-10 scale)
        """
        if not traction_text or not traction_text.strip():
            return 0.0
        
        score = 0.0
        text_lower = traction_text.lower()
        
        # Base score from content presence (0-1 point)
        word_count = len(traction_text.split())
        if word_count >= 40:
            score += 1.0
        elif word_count >= 20:
            score += 0.7
        elif word_count >= 10:
            score += 0.4
        
        # Growth metrics scoring (0-3 points)
        growth_score = 0.0
        for pattern in self.traction_indicators['growth_metrics']:
            matches = re.findall(pattern, traction_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract growth percentage
                    growth_value = float(re.sub(r'[^\d.]', '', str(match)))
                    
                    # Score based on growth rate
                    if growth_value >= 50:  # 50%+ growth
                        growth_score += 1.5
                    elif growth_value >= 20:  # 20%+ growth
                        growth_score += 1.0
                    elif growth_value >= 10:  # 10%+ growth
                        growth_score += 0.7
                    elif growth_value >= 5:  # 5%+ growth
                        growth_score += 0.4
                    else:
                        growth_score += 0.2
                except (ValueError, TypeError):
                    growth_score += 0.3  # Partial credit for mentioning growth
        
        score += min(3.0, growth_score)
        
        # User metrics scoring (0-2.5 points)
        user_score = 0.0
        for pattern in self.traction_indicators['user_metrics']:
            matches = re.findall(pattern, traction_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract user count
                    user_value = float(re.sub(r'[^\d.]', '', str(match)))
                    
                    # Determine scale multiplier
                    if 'million' in traction_text.lower() or 'M' in traction_text:
                        user_value *= 1000000
                    elif 'thousand' in traction_text.lower() or 'k' in traction_text.lower():
                        user_value *= 1000
                    
                    # Score based on user count
                    if user_value >= 1000000:  # 1M+ users
                        user_score += 1.5
                    elif user_value >= 100000:  # 100K+ users
                        user_score += 1.0
                    elif user_value >= 10000:  # 10K+ users
                        user_score += 0.7
                    elif user_value >= 1000:  # 1K+ users
                        user_score += 0.4
                    else:
                        user_score += 0.2
                except (ValueError, TypeError):
                    user_score += 0.3  # Partial credit for mentioning users
        
        score += min(2.5, user_score)
        
        # Revenue metrics scoring (0-2.5 points)
        revenue_score = 0.0
        for pattern in self.traction_indicators['revenue_metrics']:
            matches = re.findall(pattern, traction_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract revenue value
                    revenue_value = float(re.sub(r'[^\d.]', '', str(match)))
                    
                    # Determine scale multiplier
                    if 'million' in traction_text.lower() or 'M' in traction_text:
                        revenue_value *= 1000000
                    elif 'thousand' in traction_text.lower() or 'k' in traction_text.lower():
                        revenue_value *= 1000
                    
                    # Score based on revenue
                    if revenue_value >= 10000000:  # $10M+ revenue
                        revenue_score += 1.5
                    elif revenue_value >= 1000000:  # $1M+ revenue
                        revenue_score += 1.0
                    elif revenue_value >= 100000:  # $100K+ revenue
                        revenue_score += 0.7
                    elif revenue_value >= 10000:  # $10K+ revenue
                        revenue_score += 0.4
                    else:
                        revenue_score += 0.2
                except (ValueError, TypeError):
                    revenue_score += 0.3  # Partial credit for mentioning revenue
        
        score += min(2.5, revenue_score)
        
        # Engagement metrics scoring (0-1 point)
        engagement_score = 0.0
        for pattern in self.traction_indicators['engagement_metrics']:
            matches = re.findall(pattern, traction_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract engagement percentage
                    engagement_value = float(re.sub(r'[^\d.]', '', str(match)))
                    
                    # Score based on engagement metrics
                    if engagement_value >= 80:  # 80%+ retention/engagement
                        engagement_score += 0.5
                    elif engagement_value >= 60:  # 60%+ retention/engagement
                        engagement_score += 0.3
                    elif engagement_value >= 40:  # 40%+ retention/engagement
                        engagement_score += 0.2
                    else:
                        engagement_score += 0.1
                except (ValueError, TypeError):
                    engagement_score += 0.1  # Partial credit for mentioning engagement
        
        score += min(1.0, engagement_score)
        
        return min(10.0, score)

    def score_team_experience(self, team_text: str) -> float:
        """
        Score team experience based on background and founder indicators.
        
        Evaluates:
        - Years of relevant experience
        - Previous company backgrounds
        - Educational credentials
        - Domain expertise indicators
        - Leadership experience
        
        Args:
            team_text: Text content describing team information
            
        Returns:
            Team experience score (0-10 scale)
        """
        if not team_text or not team_text.strip():
            return 0.0
        
        score = 0.0
        text_lower = team_text.lower()
        
        # Base score from content presence (0-1 point)
        word_count = len(team_text.split())
        if word_count >= 50:
            score += 1.0
        elif word_count >= 25:
            score += 0.7
        elif word_count >= 10:
            score += 0.4
        
        # Years of experience scoring (0-2.5 points)
        experience_score = 0.0
        experience_patterns = [
            r'(\d+)\s*(?:\+)?\s*years?\s*(?:of\s*)?(?:experience|background)',
            r'(?:over|more\s+than)\s*(\d+)\s*years?\s*(?:of\s*)?(?:experience|background)',
            r'(\d+)\s*years?\s*(?:at|with|in)\s*[A-Z][a-zA-Z\s&\.]+'
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, team_text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    if years >= 15:  # 15+ years experience
                        experience_score += 1.0
                    elif years >= 10:  # 10+ years experience
                        experience_score += 0.8
                    elif years >= 5:  # 5+ years experience
                        experience_score += 0.6
                    elif years >= 3:  # 3+ years experience
                        experience_score += 0.4
                    else:
                        experience_score += 0.2
                except ValueError:
                    continue
        
        score += min(2.5, experience_score)
        
        # Previous company backgrounds (0-2 points)
        company_score = 0.0
        prestigious_companies = [
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta', 'netflix',
            'tesla', 'uber', 'airbnb', 'stripe', 'salesforce', 'oracle', 'ibm',
            'goldman sachs', 'mckinsey', 'bain', 'bcg', 'jpmorgan', 'morgan stanley'
        ]
        
        # Check for prestigious company mentions
        for company in prestigious_companies:
            if company in text_lower:
                company_score += 0.4
        
        # Check for startup/tech company patterns
        startup_patterns = [
            r'(?:founder|co-founder|ceo|cto|vp).*?(?:at|of)\s+([A-Z][a-zA-Z\s&\.]+)',
            r'(?:former|ex-).*?(?:at|with)\s+([A-Z][a-zA-Z\s&\.]+)',
            r'(?:led|managed|built).*?(?:at|for)\s+([A-Z][a-zA-Z\s&\.]+)'
        ]
        
        for pattern in startup_patterns:
            matches = re.findall(pattern, team_text, re.IGNORECASE)
            company_score += len(matches) * 0.2
        
        score += min(2.0, company_score)
        
        # Educational credentials (0-1.5 points)
        education_score = 0.0
        education_patterns = [
            r'(?:PhD|Doctor).*?(?:from|at|in)\s+([A-Z][a-zA-Z\s&\.]+(?:University|College|Institute))',
            r'(?:MBA|Master).*?(?:from|at|in)\s+([A-Z][a-zA-Z\s&\.]+(?:University|College|Institute))',
            r'(?:BS|Bachelor|degree).*?(?:from|at|in)\s+([A-Z][a-zA-Z\s&\.]+(?:University|College|Institute))'
        ]
        
        prestigious_schools = [
            'harvard', 'stanford', 'mit', 'berkeley', 'caltech', 'princeton',
            'yale', 'columbia', 'chicago', 'wharton', 'sloan', 'kellogg'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, team_text, re.IGNORECASE)
            for match in matches:
                education_score += 0.3
                # Bonus for prestigious schools
                if any(school in match.lower() for school in prestigious_schools):
                    education_score += 0.2
        
        score += min(1.5, education_score)
        
        # Domain expertise indicators (0-2 points)
        expertise_score = 0.0
        expertise_keywords = [
            'expert', 'specialist', 'authority', 'expertise', 'deep knowledge',
            'domain expert', 'thought leader', 'recognized', 'published',
            'speaker', 'keynote', 'advisor', 'consultant', 'veteran'
        ]
        
        for keyword in expertise_keywords:
            if keyword in text_lower:
                expertise_score += 0.3
        
        score += min(2.0, expertise_score)
        
        # Leadership experience (0-1 point)
        leadership_score = 0.0
        leadership_keywords = [
            'ceo', 'cto', 'cfo', 'founder', 'co-founder', 'president',
            'vp', 'vice president', 'director', 'head of', 'manager',
            'led team', 'managed team', 'built team', 'scaled team'
        ]
        
        for keyword in leadership_keywords:
            if keyword in text_lower:
                leadership_score += 0.2
        
        score += min(1.0, leadership_score)
        
        return min(10.0, score)

    def score_business_model(self, model_text: str) -> float:
        """
        Score business model clarity based on monetization strategy.
        
        Evaluates:
        - Clear revenue streams
        - Pricing strategy definition
        - Monetization approach
        - Scalability indicators
        - Unit economics mentions
        
        Args:
            model_text: Text content describing business model
            
        Returns:
            Business model score (0-10 scale)
        """
        if not model_text or not model_text.strip():
            return 0.0
        
        score = 0.0
        text_lower = model_text.lower()
        
        # Base score from content presence (0-1 point)
        word_count = len(model_text.split())
        if word_count >= 40:
            score += 1.0
        elif word_count >= 20:
            score += 0.7
        elif word_count >= 10:
            score += 0.4
        
        # Revenue stream identification (0-3 points)
        revenue_streams = [
            'subscription', 'saas', 'recurring revenue', 'monthly recurring',
            'annual recurring', 'freemium', 'premium', 'transaction fee',
            'commission', 'marketplace', 'advertising', 'licensing',
            'enterprise', 'b2b', 'b2c', 'usage-based', 'per-seat'
        ]
        
        revenue_score = 0.0
        for stream in revenue_streams:
            if stream in text_lower:
                revenue_score += 0.4
        
        score += min(3.0, revenue_score)
        
        # Pricing strategy clarity (0-2 points)
        pricing_indicators = [
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:per|/)\s*(?:month|year|user|seat|transaction)',
            r'(?:pricing|price|cost|fee).*?\$(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:commission|fee|take\s*rate)',
            'tiered pricing', 'pricing tiers', 'pricing model', 'pricing strategy'
        ]
        
        pricing_score = 0.0
        for indicator in pricing_indicators:
            if isinstance(indicator, str):
                if indicator in text_lower:
                    pricing_score += 0.5
            else:
                if re.search(indicator, model_text, re.IGNORECASE):
                    pricing_score += 0.5
        
        score += min(2.0, pricing_score)
        
        # Monetization approach clarity (0-2 points)
        monetization_keywords = [
            'monetization', 'revenue model', 'business model', 'how we make money',
            'revenue generation', 'income stream', 'profit model', 'value capture'
        ]
        
        monetization_score = 0.0
        for keyword in monetization_keywords:
            if keyword in text_lower:
                monetization_score += 0.4
        
        score += min(2.0, monetization_score)
        
        # Scalability indicators (0-1.5 points)
        scalability_keywords = [
            'scalable', 'scalability', 'scale', 'marginal cost', 'network effects',
            'viral', 'automated', 'self-service', 'platform', 'marketplace',
            'high margin', 'low marginal cost', 'economies of scale'
        ]
        
        scalability_score = 0.0
        for keyword in scalability_keywords:
            if keyword in text_lower:
                scalability_score += 0.3
        
        score += min(1.5, scalability_score)
        
        # Unit economics mentions (0-1.5 points)
        unit_economics_keywords = [
            'unit economics', 'ltv', 'lifetime value', 'cac', 'customer acquisition cost',
            'payback period', 'gross margin', 'contribution margin', 'arpu',
            'average revenue per user', 'churn rate', 'retention rate'
        ]
        
        unit_economics_score = 0.0
        for keyword in unit_economics_keywords:
            if keyword in text_lower:
                unit_economics_score += 0.3
        
        score += min(1.5, unit_economics_score)
        
        return min(10.0, score)

    def score_vision_moat(self, vision_text: str) -> float:
        """
        Score vision/moat based on competitive advantage and IP mentions.
        
        Evaluates:
        - Competitive advantage clarity
        - Intellectual property mentions
        - Defensibility indicators
        - Unique value proposition
        - Barriers to entry
        
        Args:
            vision_text: Text content describing vision and competitive moat
            
        Returns:
            Vision/moat score (0-10 scale)
        """
        if not vision_text or not vision_text.strip():
            return 0.0
        
        score = 0.0
        text_lower = vision_text.lower()
        
        # Base score from content presence (0-1 point)
        word_count = len(vision_text.split())
        if word_count >= 30:
            score += 1.0
        elif word_count >= 15:
            score += 0.7
        elif word_count >= 8:
            score += 0.4
        
        # Competitive advantage indicators (0-2.5 points)
        competitive_advantage_keywords = [
            'competitive advantage', 'competitive edge', 'differentiation',
            'unique', 'proprietary', 'exclusive', 'first-mover',
            'market leader', 'industry leader', 'breakthrough',
            'innovative', 'disruptive', 'revolutionary'
        ]
        
        advantage_score = 0.0
        for keyword in competitive_advantage_keywords:
            if keyword in text_lower:
                advantage_score += 0.4
        
        score += min(2.5, advantage_score)
        
        # Intellectual property mentions (0-2 points)
        ip_keywords = [
            'patent', 'patents', 'intellectual property', 'ip', 'trademark',
            'copyright', 'trade secret', 'proprietary technology',
            'proprietary algorithm', 'proprietary data', 'patent pending'
        ]
        
        ip_score = 0.0
        for keyword in ip_keywords:
            if keyword in text_lower:
                ip_score += 0.5
        
        score += min(2.0, ip_score)
        
        # Defensibility indicators (0-2 points)
        defensibility_keywords = [
            'defensible', 'moat', 'barriers to entry', 'switching costs',
            'network effects', 'data advantage', 'scale advantage',
            'brand loyalty', 'customer lock-in', 'high switching cost',
            'difficult to replicate', 'hard to copy'
        ]
        
        defensibility_score = 0.0
        for keyword in defensibility_keywords:
            if keyword in text_lower:
                defensibility_score += 0.4
        
        score += min(2.0, defensibility_score)
        
        # Unique value proposition (0-1.5 points)
        uvp_keywords = [
            'unique value', 'value proposition', 'unique selling',
            'only solution', 'first to', 'never been done',
            'game changer', 'paradigm shift', 'new category'
        ]
        
        uvp_score = 0.0
        for keyword in uvp_keywords:
            if keyword in text_lower:
                uvp_score += 0.3
        
        score += min(1.5, uvp_score)
        
        # Technology and data advantages (0-1 point)
        tech_advantage_keywords = [
            'ai advantage', 'machine learning', 'data advantage',
            'proprietary dataset', 'algorithmic advantage',
            'technical expertise', 'deep tech', 'advanced technology'
        ]
        
        tech_score = 0.0
        for keyword in tech_advantage_keywords:
            if keyword in text_lower:
                tech_score += 0.25
        
        score += min(1.0, tech_score)
        
        return min(10.0, score)

    def score_overall_confidence(self, full_text: str) -> float:
        """
        Score overall confidence based on language tone and decisiveness.
        
        Evaluates:
        - Confidence indicators in language
        - Decisiveness of statements
        - Certainty vs. uncertainty markers
        - Positive sentiment
        - Action-oriented language
        
        Args:
            full_text: Complete text content of the pitch deck
            
        Returns:
            Overall confidence score (0-10 scale)
        """
        if not full_text or not full_text.strip():
            return 0.0
        
        score = 0.0
        text_lower = full_text.lower()
        
        # Base score from content completeness (0-1 point)
        word_count = len(full_text.split())
        if word_count >= 500:
            score += 1.0
        elif word_count >= 300:
            score += 0.8
        elif word_count >= 150:
            score += 0.6
        elif word_count >= 50:
            score += 0.3
        
        # Confidence indicators from NLP analyzer (0-3 points)
        confidence_indicators = self.nlp_analyzer.detect_confidence_indicators(full_text)
        confidence_score = min(3.0, len(confidence_indicators) * 0.2)
        score += confidence_score
        
        # Sentiment analysis (0-2 points)
        sentiment = self.nlp_analyzer.analyze_sentiment(full_text)
        if sentiment >= 0.3:  # Strong positive sentiment
            score += 2.0
        elif sentiment >= 0.1:  # Moderate positive sentiment
            score += 1.5
        elif sentiment >= 0.0:  # Neutral to slightly positive
            score += 1.0
        elif sentiment >= -0.1:  # Slightly negative (acceptable for problem description)
            score += 0.5
        
        # Decisiveness markers (0-2 points)
        decisive_keywords = [
            'will', 'going to', 'committed to', 'determined to',
            'focused on', 'dedicated to', 'proven', 'demonstrated',
            'achieved', 'accomplished', 'delivered', 'executed'
        ]
        
        decisive_score = 0.0
        for keyword in decisive_keywords:
            if keyword in text_lower:
                decisive_score += 0.3
        
        score += min(2.0, decisive_score)
        
        # Uncertainty markers (penalty: -1 point max)
        uncertainty_keywords = [
            'maybe', 'perhaps', 'might', 'could be', 'possibly',
            'hopefully', 'trying to', 'attempting to', 'unsure',
            'uncertain', 'unclear', 'confused', 'difficult'
        ]
        
        uncertainty_penalty = 0.0
        for keyword in uncertainty_keywords:
            if keyword in text_lower:
                uncertainty_penalty += 0.2
        
        score -= min(1.0, uncertainty_penalty)
        
        # Action-oriented language (0-1.5 points)
        action_keywords = [
            'launch', 'build', 'create', 'develop', 'implement',
            'execute', 'deliver', 'scale', 'grow', 'expand',
            'capture', 'dominate', 'lead', 'transform'
        ]
        
        action_score = 0.0
        for keyword in action_keywords:
            if keyword in text_lower:
                action_score += 0.2
        
        score += min(1.5, action_score)
        
        # Quantified statements bonus (0-0.5 points)
        metrics = self.nlp_analyzer.extract_metrics(full_text)
        if len(metrics) >= 5:
            score += 0.5
        elif len(metrics) >= 3:
            score += 0.3
        elif len(metrics) >= 1:
            score += 0.1
        
        return max(0.0, min(10.0, score))

    def calculate_composite_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate composite final score using weighted scoring algorithm.
        
        Combines individual dimension scores using configurable weights
        and normalizes to 0-100 scale.
        
        Args:
            dimension_scores: Dictionary of dimension names to scores (0-10 scale)
            
        Returns:
            Composite score normalized to 0-100 scale
        """
        if not dimension_scores:
            return 0.0
        
        # Get dimension weights from configuration
        weights = self.config.dimension_weights
        
        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            if dimension in weights:
                weight = weights[dimension]
                weighted_sum += score * weight
                total_weight += weight
        
        # Handle case where no valid dimensions found
        if total_weight == 0.0:
            return 0.0
        
        # Calculate weighted average (0-10 scale)
        weighted_average = weighted_sum / total_weight
        
        # Normalize to 0-100 scale
        normalized_score = (weighted_average / 10.0) * 100.0
        
        # Ensure score is within bounds
        return max(0.0, min(100.0, normalized_score))

    def score_all_dimensions(self, sections: Dict[str, str]) -> Dict[str, float]:
        """
        Score all dimensions for a complete pitch deck evaluation.
        
        Args:
            sections: Dictionary of section names to text content
            
        Returns:
            Dictionary of dimension names to scores (0-10 scale)
        """
        scores = {}
        
        # Get section content with fallbacks
        problem_text = sections.get('problem', '') or sections.get('Problem', '') or ''
        market_text = sections.get('market', '') or sections.get('Market', '') or sections.get('market_size', '') or ''
        traction_text = sections.get('traction', '') or sections.get('Traction', '') or ''
        team_text = sections.get('team', '') or sections.get('Team', '') or ''
        business_model_text = sections.get('business_model', '') or sections.get('Business Model', '') or sections.get('monetization', '') or ''
        vision_text = sections.get('vision', '') or sections.get('Vision', '') or sections.get('competitive_advantage', '') or sections.get('solution', '') or ''
        
        # Combine all text for overall confidence scoring
        full_text = ' '.join(sections.values())
        
        # Score each dimension
        scores['problem_clarity'] = self.score_problem_clarity(problem_text)
        scores['market_potential'] = self.score_market_potential(market_text)
        scores['traction_strength'] = self.score_traction_strength(traction_text)
        scores['team_experience'] = self.score_team_experience(team_text)
        scores['business_model'] = self.score_business_model(business_model_text)
        scores['vision_moat'] = self.score_vision_moat(vision_text)
        scores['overall_confidence'] = self.score_overall_confidence(full_text)
        
        return scores

    def get_scoring_insights(self, dimension_scores: Dict[str, float]) -> Dict[str, str]:
        """
        Generate insights and recommendations based on dimension scores.
        
        Args:
            dimension_scores: Dictionary of dimension names to scores
            
        Returns:
            Dictionary of insights for each dimension
        """
        insights = {}
        thresholds = self.config.scoring_thresholds
        
        for dimension, score in dimension_scores.items():
            if dimension in thresholds:
                dimension_thresholds = thresholds[dimension]
                
                if score >= dimension_thresholds['excellent']:
                    insights[dimension] = f"Excellent {dimension.replace('_', ' ')} - strong competitive advantage"
                elif score >= dimension_thresholds['good']:
                    insights[dimension] = f"Good {dimension.replace('_', ' ')} - solid foundation"
                elif score >= dimension_thresholds['fair']:
                    insights[dimension] = f"Fair {dimension.replace('_', ' ')} - room for improvement"
                else:
                    insights[dimension] = f"Weak {dimension.replace('_', ' ')} - needs significant development"
        
        return insights

    def generate_investability_insight(self, dimension_scores: Dict[str, float], composite_score: float) -> str:
        """
        Generate overall investability insight based on scores.
        
        Args:
            dimension_scores: Dictionary of dimension scores
            composite_score: Overall composite score (0-100)
            
        Returns:
            Investability insight string
        """
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for dimension, score in dimension_scores.items():
            dimension_name = dimension.replace('_', ' ').title()
            if score >= 7.0:
                strengths.append(dimension_name)
            elif score < 4.0:
                weaknesses.append(dimension_name)
        
        # Generate insight based on composite score
        if composite_score >= 80:
            investment_level = "Highly Investable"
            recommendation = "Strong investment opportunity with excellent fundamentals"
        elif composite_score >= 65:
            investment_level = "Investable"
            recommendation = "Good investment potential with solid foundation"
        elif composite_score >= 50:
            investment_level = "Moderately Investable"
            recommendation = "Moderate investment potential, requires further development"
        elif composite_score >= 35:
            investment_level = "Low Investment Potential"
            recommendation = "Significant improvements needed before investment consideration"
        else:
            investment_level = "Not Investment Ready"
            recommendation = "Major fundamental issues need to be addressed"
        
        # Build insight string
        insight_parts = [f"{investment_level}: {recommendation}"]
        
        if strengths:
            insight_parts.append(f"Key strengths: {', '.join(strengths[:3])}")
        
        if weaknesses:
            insight_parts.append(f"Areas for improvement: {', '.join(weaknesses[:3])}")
        
        return ". ".join(insight_parts) + "."

    def validate_scores(self, dimension_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate that all dimension scores are within expected ranges.
        
        Args:
            dimension_scores: Dictionary of dimension scores to validate
            
        Returns:
            Dictionary of dimension names to validation status
        """
        validation_results = {}
        
        for dimension, score in dimension_scores.items():
            # Check if score is within valid range (0-10)
            is_valid = isinstance(score, (int, float)) and 0.0 <= score <= 10.0
            validation_results[dimension] = is_valid
        
        return validation_results

    def get_score_distribution(self, dimension_scores: Dict[str, float]) -> Dict[str, int]:
        """
        Get distribution of scores across quality levels.
        
        Args:
            dimension_scores: Dictionary of dimension scores
            
        Returns:
            Dictionary with counts for each quality level
        """
        distribution = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0
        }
        
        thresholds = self.config.scoring_thresholds
        
        for dimension, score in dimension_scores.items():
            if dimension in thresholds:
                dimension_thresholds = thresholds[dimension]
                
                if score >= dimension_thresholds['excellent']:
                    distribution['excellent'] += 1
                elif score >= dimension_thresholds['good']:
                    distribution['good'] += 1
                elif score >= dimension_thresholds['fair']:
                    distribution['fair'] += 1
                else:
                    distribution['poor'] += 1
        
        return distribution