"""
Configuration management system for scoring parameters and thresholds.
"""
import json
import os
from typing import Dict, List, Any
from ..models.data_models import ScoringConfiguration


class ConfigManager:
    """Manages configuration parameters for the pitch deck evaluation system."""
    
    def __init__(self, config_path: str = "config/scoring_config.json"):
        self.config_path = config_path
        self._config = None
        self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load default configuration parameters."""
        default_config = {
            "dimension_weights": {
                "problem_clarity": 0.15,
                "market_potential": 0.20,
                "traction_strength": 0.25,
                "team_experience": 0.15,
                "business_model": 0.10,
                "vision_moat": 0.10,
                "overall_confidence": 0.05
            },
            "keyword_mappings": {
                "problem": ["problem", "pain point", "challenge", "issue", "difficulty"],
                "solution": ["solution", "approach", "product", "service", "platform"],
                "market": ["market", "tam", "sam", "som", "addressable", "opportunity"],
                "traction": ["traction", "growth", "users", "revenue", "customers", "mom", "yoy"],
                "team": ["team", "founder", "ceo", "cto", "experience", "background"],
                "business_model": ["business model", "monetization", "revenue", "pricing", "subscription"],
                "vision": ["vision", "moat", "competitive", "advantage", "ip", "patent", "defensible"]
            },
            "scoring_thresholds": {
                "problem_clarity": {
                    "excellent": 8.0,
                    "good": 6.0,
                    "fair": 4.0,
                    "poor": 2.0
                },
                "market_potential": {
                    "excellent": 8.0,
                    "good": 6.0,
                    "fair": 4.0,
                    "poor": 2.0
                },
                "traction_strength": {
                    "excellent": 8.0,
                    "good": 6.0,
                    "fair": 4.0,
                    "poor": 2.0
                },
                "team_experience": {
                    "excellent": 8.0,
                    "good": 6.0,
                    "fair": 4.0,
                    "poor": 2.0
                },
                "business_model": {
                    "excellent": 8.0,
                    "good": 6.0,
                    "fair": 4.0,
                    "poor": 2.0
                },
                "vision_moat": {
                    "excellent": 8.0,
                    "good": 6.0,
                    "fair": 4.0,
                    "poor": 2.0
                },
                "overall_confidence": {
                    "excellent": 8.0,
                    "good": 6.0,
                    "fair": 4.0,
                    "poor": 2.0
                }
            },
            "confidence_indicators": [
                "will", "proven", "demonstrated", "validated", "confirmed",
                "established", "successful", "achieved", "accomplished",
                "strong", "solid", "robust", "significant", "substantial"
            ]
        }
        self._config = default_config
    
    def load_config(self, config_path: str = None) -> ScoringConfiguration:
        """Load configuration from file or use default."""
        if config_path:
            self.config_path = config_path
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                # Merge with defaults
                self._merge_config(file_config)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load config file {self.config_path}: {e}")
                print("Using default configuration.")
        
        return ScoringConfiguration(
            dimension_weights=self._config["dimension_weights"],
            keyword_mappings=self._config["keyword_mappings"],
            scoring_thresholds=self._config["scoring_thresholds"],
            confidence_indicators=self._config["confidence_indicators"]
        )
    
    def _merge_config(self, file_config: Dict[str, Any]) -> None:
        """Merge file configuration with defaults."""
        for key, value in file_config.items():
            if key in self._config:
                if isinstance(value, dict) and isinstance(self._config[key], dict):
                    self._config[key].update(value)
                else:
                    self._config[key] = value
    
    def save_config(self, config: ScoringConfiguration, config_path: str = None) -> None:
        """Save configuration to file."""
        if config_path:
            self.config_path = config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        config_dict = {
            "dimension_weights": config.dimension_weights,
            "keyword_mappings": config.keyword_mappings,
            "scoring_thresholds": config.scoring_thresholds,
            "confidence_indicators": config.confidence_indicators
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_dimension_weight(self, dimension: str) -> float:
        """Get weight for a specific scoring dimension."""
        return self._config["dimension_weights"].get(dimension, 0.0)
    
    def get_keywords(self, category: str) -> List[str]:
        """Get keywords for a specific category."""
        return self._config["keyword_mappings"].get(category, [])
    
    def get_threshold(self, dimension: str, level: str) -> float:
        """Get threshold value for a dimension and quality level."""
        return self._config["scoring_thresholds"].get(dimension, {}).get(level, 0.0)
    
    def get_confidence_indicators(self) -> List[str]:
        """Get list of confidence indicator words."""
        return self._config["confidence_indicators"]