"""
Core data models for pitch deck processing and evaluation.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class PitchDeckData:
    """Represents extracted and processed pitch deck content."""
    name: str
    raw_text: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]
    extraction_quality: float


@dataclass
class SectionScores:
    """Scores for individual evaluation dimensions."""
    problem_clarity: float
    market_potential: float
    traction_strength: float
    team_experience: float
    business_model: float
    vision_moat: float
    overall_confidence: float


@dataclass
class EvaluationResult:
    """Complete evaluation result for a pitch deck."""
    deck_name: str
    section_scores: SectionScores
    composite_score: float
    investability_insight: str
    industry_category: str
    summary_points: List[str]
    confidence_level: float


@dataclass
class ScoringConfiguration:
    """Configuration parameters for scoring algorithms."""
    dimension_weights: Dict[str, float]
    keyword_mappings: Dict[str, List[str]]
    scoring_thresholds: Dict[str, Dict[str, float]]
    confidence_indicators: List[str]