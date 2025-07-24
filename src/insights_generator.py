"""
Investability insights generation system.
Creates personalized comments based on pitch deck scoring patterns.
"""
from typing import Dict, List, Tuple
from ..models.data_models import EvaluationResult, SectionScores


class InvestabilityInsightsGenerator:
    """
    Generates investability insights based on scoring patterns and deck analysis.
    Uses rule-based templates to create personalized comments.
    """
    
    def __init__(self):
        """Initialize the insights generator with scoring thresholds."""
        # Define scoring thresholds for different performance levels
        self.thresholds = {
            'excellent': 8.5,
            'good': 7.0,
            'moderate': 5.5,
            'poor': 4.0
        }
        
        # Define composite score thresholds
        self.composite_thresholds = {
            'excellent': 85.0,
            'good': 70.0,
            'moderate': 55.0,
            'poor': 40.0
        }
    
    def generate_insight(self, result: EvaluationResult) -> str:
        """
        Generate investability insight for a single evaluation result.
        
        Args:
            result: EvaluationResult containing scores and analysis
            
        Returns:
            Personalized investability insight comment
        """
        # Analyze scoring patterns
        strengths = self._identify_strengths(result.section_scores)
        weaknesses = self._identify_weaknesses(result.section_scores)
        overall_level = self._get_performance_level(result.composite_score, self.composite_thresholds)
        
        # Generate insight based on patterns
        insight = self._build_insight_comment(
            deck_name=result.deck_name,
            overall_level=overall_level,
            composite_score=result.composite_score,
            strengths=strengths,
            weaknesses=weaknesses,
            confidence_level=result.confidence_level
        )
        
        return insight
    
    def generate_batch_insights(self, results: List[EvaluationResult]) -> Dict[str, str]:
        """
        Generate insights for multiple evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary mapping deck names to insight comments
        """
        insights = {}
        for result in results:
            insights[result.deck_name] = self.generate_insight(result)
        
        return insights
    
    def _identify_strengths(self, scores: SectionScores) -> List[Tuple[str, float]]:
        """
        Identify the strongest dimensions in the pitch deck.
        
        Args:
            scores: SectionScores object with dimension scores
            
        Returns:
            List of (dimension_name, score) tuples for strong areas
        """
        dimension_scores = [
            ('problem_clarity', scores.problem_clarity),
            ('market_potential', scores.market_potential),
            ('traction_strength', scores.traction_strength),
            ('team_experience', scores.team_experience),
            ('business_model', scores.business_model),
            ('vision_moat', scores.vision_moat),
            ('overall_confidence', scores.overall_confidence)
        ]
        
        # Filter for good or excellent scores
        strengths = [
            (dim, score) for dim, score in dimension_scores 
            if score >= self.thresholds['good']
        ]
        
        # Sort by score descending
        strengths.sort(key=lambda x: x[1], reverse=True)
        
        return strengths[:3]  # Return top 3 strengths
    
    def _identify_weaknesses(self, scores: SectionScores) -> List[Tuple[str, float]]:
        """
        Identify the weakest dimensions in the pitch deck.
        
        Args:
            scores: SectionScores object with dimension scores
            
        Returns:
            List of (dimension_name, score) tuples for weak areas
        """
        dimension_scores = [
            ('problem_clarity', scores.problem_clarity),
            ('market_potential', scores.market_potential),
            ('traction_strength', scores.traction_strength),
            ('team_experience', scores.team_experience),
            ('business_model', scores.business_model),
            ('vision_moat', scores.vision_moat),
            ('overall_confidence', scores.overall_confidence)
        ]
        
        # Filter for moderate or poor scores
        weaknesses = [
            (dim, score) for dim, score in dimension_scores 
            if score < self.thresholds['good']
        ]
        
        # Sort by score ascending (worst first)
        weaknesses.sort(key=lambda x: x[1])
        
        return weaknesses[:3]  # Return top 3 weaknesses
    
    def _get_performance_level(self, score: float, thresholds: Dict[str, float]) -> str:
        """
        Determine performance level based on score and thresholds.
        
        Args:
            score: Numerical score to evaluate
            thresholds: Dictionary of performance level thresholds
            
        Returns:
            Performance level string
        """
        if score >= thresholds['excellent']:
            return 'excellent'
        elif score >= thresholds['good']:
            return 'good'
        elif score >= thresholds['moderate']:
            return 'moderate'
        else:
            return 'poor'
    
    def _build_insight_comment(self, 
                              deck_name: str,
                              overall_level: str,
                              composite_score: float,
                              strengths: List[Tuple[str, float]],
                              weaknesses: List[Tuple[str, float]],
                              confidence_level: float) -> str:
        """
        Build the final insight comment using templates and analysis.
        
        Args:
            deck_name: Name of the pitch deck
            overall_level: Overall performance level
            composite_score: Composite score value
            strengths: List of strength dimensions
            weaknesses: List of weakness dimensions
            confidence_level: Analysis confidence level
            
        Returns:
            Formatted insight comment
        """
        # Start with overall assessment
        overall_templates = {
            'excellent': f"{deck_name} demonstrates exceptional investment potential with a composite score of {composite_score:.1f}.",
            'good': f"{deck_name} shows strong investment potential with a composite score of {composite_score:.1f}.",
            'moderate': f"{deck_name} presents moderate investment potential with a composite score of {composite_score:.1f}.",
            'poor': f"{deck_name} shows limited investment potential with a composite score of {composite_score:.1f}."
        }
        
        comment_parts = [overall_templates[overall_level]]
        
        # Add strengths analysis
        if strengths:
            strength_text = self._format_strengths(strengths)
            comment_parts.append(f"Key strengths include {strength_text}.")
        
        # Add weaknesses analysis
        if weaknesses:
            weakness_text = self._format_weaknesses(weaknesses)
            comment_parts.append(f"Areas for improvement: {weakness_text}.")
        
        # Add investment recommendation
        recommendation = self._get_investment_recommendation(overall_level, strengths, weaknesses)
        comment_parts.append(recommendation)
        
        # Add confidence qualifier if low
        if confidence_level < 0.7:
            comment_parts.append("Note: Analysis confidence is moderate due to limited extractable content.")
        
        return " ".join(comment_parts)
    
    def _format_strengths(self, strengths: List[Tuple[str, float]]) -> str:
        """
        Format strengths list into readable text.
        
        Args:
            strengths: List of (dimension, score) tuples
            
        Returns:
            Formatted strengths text
        """
        dimension_names = {
            'problem_clarity': 'clear problem definition',
            'market_potential': 'strong market opportunity',
            'traction_strength': 'solid traction metrics',
            'team_experience': 'experienced team',
            'business_model': 'clear business model',
            'vision_moat': 'competitive advantages',
            'overall_confidence': 'confident presentation'
        }
        
        formatted_strengths = []
        for dimension, score in strengths:
            name = dimension_names.get(dimension, dimension.replace('_', ' '))
            if score >= self.thresholds['excellent']:
                formatted_strengths.append(f"exceptional {name}")
            else:
                formatted_strengths.append(f"strong {name}")
        
        if len(formatted_strengths) == 1:
            return formatted_strengths[0]
        elif len(formatted_strengths) == 2:
            return f"{formatted_strengths[0]} and {formatted_strengths[1]}"
        else:
            return f"{', '.join(formatted_strengths[:-1])}, and {formatted_strengths[-1]}"
    
    def _format_weaknesses(self, weaknesses: List[Tuple[str, float]]) -> str:
        """
        Format weaknesses list into readable text.
        
        Args:
            weaknesses: List of (dimension, score) tuples
            
        Returns:
            Formatted weaknesses text
        """
        dimension_names = {
            'problem_clarity': 'problem definition clarity',
            'market_potential': 'market opportunity validation',
            'traction_strength': 'traction demonstration',
            'team_experience': 'team background presentation',
            'business_model': 'business model clarity',
            'vision_moat': 'competitive differentiation',
            'overall_confidence': 'presentation confidence'
        }
        
        formatted_weaknesses = []
        for dimension, score in weaknesses:
            name = dimension_names.get(dimension, dimension.replace('_', ' '))
            if score < self.thresholds['poor']:
                formatted_weaknesses.append(f"significant improvement needed in {name}")
            else:
                formatted_weaknesses.append(f"strengthening {name}")
        
        if len(formatted_weaknesses) == 1:
            return formatted_weaknesses[0]
        elif len(formatted_weaknesses) == 2:
            return f"{formatted_weaknesses[0]} and {formatted_weaknesses[1]}"
        else:
            return f"{', '.join(formatted_weaknesses[:-1])}, and {formatted_weaknesses[-1]}"
    
    def _get_investment_recommendation(self, 
                                     overall_level: str,
                                     strengths: List[Tuple[str, float]],
                                     weaknesses: List[Tuple[str, float]]) -> str:
        """
        Generate investment recommendation based on analysis.
        
        Args:
            overall_level: Overall performance level
            strengths: List of strength dimensions
            weaknesses: List of weakness dimensions
            
        Returns:
            Investment recommendation text
        """
        recommendations = {
            'excellent': "Highly recommended for investment consideration with strong fundamentals across multiple dimensions.",
            'good': "Recommended for further due diligence with solid foundation and clear value proposition.",
            'moderate': "Consider for investment with focused improvements in key areas and additional validation.",
            'poor': "Not recommended for investment without significant improvements in fundamental areas."
        }
        
        base_recommendation = recommendations[overall_level]
        
        # Add specific guidance based on patterns
        if overall_level in ['good', 'moderate'] and len(strengths) >= 2:
            base_recommendation += " The strong performance in key areas provides a solid foundation for growth."
        
        if overall_level in ['moderate', 'poor'] and len(weaknesses) >= 3:
            base_recommendation += " Multiple areas require attention before investment readiness."
        
        return base_recommendation
    
    def generate_comparative_insights(self, results: List[EvaluationResult]) -> Dict[str, str]:
        """
        Generate comparative insights showing how each deck performs relative to others.
        
        Args:
            results: List of all evaluation results for comparison
            
        Returns:
            Dictionary of comparative insights by deck name
        """
        if len(results) < 2:
            return {}
        
        # Sort by composite score
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        comparative_insights = {}
        for i, result in enumerate(sorted_results):
            rank = i + 1
            total = len(results)
            
            if rank == 1:
                comparative_text = f"Ranks #1 out of {total} decks analyzed, demonstrating superior performance across key metrics."
            elif rank <= total // 3:
                comparative_text = f"Ranks #{rank} out of {total} decks, placing in the top tier of analyzed pitches."
            elif rank <= 2 * total // 3:
                comparative_text = f"Ranks #{rank} out of {total} decks, showing middle-tier performance with room for improvement."
            else:
                comparative_text = f"Ranks #{rank} out of {total} decks, indicating significant areas for enhancement needed."
            
            # Add the comparative insight to the existing insight
            existing_insight = self.generate_insight(result)
            comparative_insights[result.deck_name] = f"{existing_insight} {comparative_text}"
        
        return comparative_insights