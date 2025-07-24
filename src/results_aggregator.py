"""
Results aggregation system for pitch deck evaluations.
Handles ranking, top/bottom identification, and result organization.
"""
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, replace
from ..models.data_models import EvaluationResult, SectionScores
from .insights_generator import InvestabilityInsightsGenerator


@dataclass
class RankedResults:
    """Container for ranked evaluation results with metadata."""
    all_results: List[EvaluationResult]
    ranked_results: List[EvaluationResult]
    top_3: List[EvaluationResult]
    bottom_3: List[EvaluationResult]
    average_scores: Dict[str, float]
    score_statistics: Dict[str, Dict[str, float]]


class ResultsAggregator:
    """
    Aggregates and ranks pitch deck evaluation results.
    Provides comprehensive analysis of evaluation outcomes.
    """
    
    def __init__(self):
        """Initialize the results aggregator."""
        self.insights_generator = InvestabilityInsightsGenerator()
    
    def aggregate_results(self, results: List[EvaluationResult]) -> RankedResults:
        """
        Aggregate and rank evaluation results.
        
        Args:
            results: List of evaluation results to aggregate
            
        Returns:
            RankedResults containing ranked data and statistics
        """
        if not results:
            return RankedResults(
                all_results=[],
                ranked_results=[],
                top_3=[],
                bottom_3=[],
                average_scores={},
                score_statistics={}
            )
        
        # Generate investability insights for all results
        results_with_insights = self._generate_insights(results)
        
        # Sort results by composite score (descending)
        ranked_results = self._rank_by_composite_score(results_with_insights)
        
        # Identify top 3 and bottom 3
        top_3, bottom_3 = self._identify_top_bottom_performers(ranked_results)
        
        # Calculate average scores across all dimensions
        average_scores = self._calculate_average_scores(results_with_insights)
        
        # Generate score statistics
        score_statistics = self._calculate_score_statistics(results_with_insights)
        
        return RankedResults(
            all_results=results_with_insights,
            ranked_results=ranked_results,
            top_3=top_3,
            bottom_3=bottom_3,
            average_scores=average_scores,
            score_statistics=score_statistics
        )
    
    def _rank_by_composite_score(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        """
        Rank results by composite score in descending order.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Sorted list of results by composite score
        """
        return sorted(results, key=lambda x: x.composite_score, reverse=True)
    
    def _identify_top_bottom_performers(self, ranked_results: List[EvaluationResult]) -> Tuple[List[EvaluationResult], List[EvaluationResult]]:
        """
        Identify top 3 and bottom 3 performing decks.
        
        Args:
            ranked_results: Results sorted by composite score
            
        Returns:
            Tuple of (top_3, bottom_3) results
        """
        if len(ranked_results) == 0:
            return [], []
        
        # Get top 3 (first 3 in ranked list)
        top_3 = ranked_results[:3]
        
        # Get bottom 3 (last 3 in ranked list, but maintain descending order)
        if len(ranked_results) >= 3:
            bottom_3 = ranked_results[-3:]
        else:
            # If less than 3 results, bottom_3 is empty or contains remaining results
            bottom_3 = ranked_results[len(top_3):]
        
        return top_3, bottom_3
    
    def _calculate_average_scores(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
        Calculate average scores across all dimensions.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of average scores by dimension
        """
        if not results:
            return {}
        
        # Initialize score accumulators
        score_sums = {
            'problem_clarity': 0.0,
            'market_potential': 0.0,
            'traction_strength': 0.0,
            'team_experience': 0.0,
            'business_model': 0.0,
            'vision_moat': 0.0,
            'overall_confidence': 0.0,
            'composite_score': 0.0
        }
        
        # Sum all scores
        for result in results:
            score_sums['problem_clarity'] += result.section_scores.problem_clarity
            score_sums['market_potential'] += result.section_scores.market_potential
            score_sums['traction_strength'] += result.section_scores.traction_strength
            score_sums['team_experience'] += result.section_scores.team_experience
            score_sums['business_model'] += result.section_scores.business_model
            score_sums['vision_moat'] += result.section_scores.vision_moat
            score_sums['overall_confidence'] += result.section_scores.overall_confidence
            score_sums['composite_score'] += result.composite_score
        
        # Calculate averages
        num_results = len(results)
        return {dimension: score_sum / num_results for dimension, score_sum in score_sums.items()}
    
    def _calculate_score_statistics(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics for each scoring dimension.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of statistics by dimension (min, max, mean, std)
        """
        if not results:
            return {}
        
        # Extract scores by dimension
        dimensions = {
            'problem_clarity': [r.section_scores.problem_clarity for r in results],
            'market_potential': [r.section_scores.market_potential for r in results],
            'traction_strength': [r.section_scores.traction_strength for r in results],
            'team_experience': [r.section_scores.team_experience for r in results],
            'business_model': [r.section_scores.business_model for r in results],
            'vision_moat': [r.section_scores.vision_moat for r in results],
            'overall_confidence': [r.section_scores.overall_confidence for r in results],
            'composite_score': [r.composite_score for r in results]
        }
        
        statistics = {}
        for dimension, scores in dimensions.items():
            statistics[dimension] = {
                'min': min(scores),
                'max': max(scores),
                'mean': sum(scores) / len(scores),
                'std': self._calculate_std_dev(scores)
            }
        
        return statistics
    
    def _calculate_std_dev(self, scores: List[float]) -> float:
        """
        Calculate standard deviation of scores.
        
        Args:
            scores: List of numerical scores
            
        Returns:
            Standard deviation
        """
        if len(scores) <= 1:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
        return variance ** 0.5
    
    def get_ranking_summary(self, ranked_results: RankedResults) -> Dict[str, Any]:
        """
        Generate a summary of ranking results.
        
        Args:
            ranked_results: Aggregated ranking results
            
        Returns:
            Dictionary containing ranking summary information
        """
        if not ranked_results.ranked_results:
            return {
                'total_decks': 0,
                'score_range': {'min': 0, 'max': 0},
                'top_performer': None,
                'bottom_performer': None,
                'average_composite_score': 0
            }
        
        return {
            'total_decks': len(ranked_results.ranked_results),
            'score_range': {
                'min': ranked_results.ranked_results[-1].composite_score,
                'max': ranked_results.ranked_results[0].composite_score
            },
            'top_performer': {
                'name': ranked_results.ranked_results[0].deck_name,
                'score': ranked_results.ranked_results[0].composite_score
            },
            'bottom_performer': {
                'name': ranked_results.ranked_results[-1].deck_name,
                'score': ranked_results.ranked_results[-1].composite_score
            },
            'average_composite_score': ranked_results.average_scores.get('composite_score', 0)
        }
    
    def _generate_insights(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        """
        Generate investability insights for all results and return updated results.
        
        Args:
            results: List of evaluation results without insights
            
        Returns:
            List of evaluation results with generated insights
        """
        # Generate comparative insights that include ranking information
        comparative_insights = self.insights_generator.generate_comparative_insights(results)
        
        # Update results with generated insights
        updated_results = []
        for result in results:
            insight = comparative_insights.get(result.deck_name, 
                                             self.insights_generator.generate_insight(result))
            
            # Create new result with updated insight
            updated_result = replace(result, investability_insight=insight)
            updated_results.append(updated_result)
        
        return updated_results