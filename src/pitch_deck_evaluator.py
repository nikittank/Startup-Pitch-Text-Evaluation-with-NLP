"""
Main entry point for the Startup Pitch NLP Evaluator system.
"""
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .models.data_models import PitchDeckData, EvaluationResult, ScoringConfiguration
from .models.interfaces import (
    PDFExtractorInterface,
    ContentPreprocessorInterface,
    SectionClassifierInterface,
    NLPAnalyzerInterface,
    MultiDimensionalScorerInterface,
    VisualizationEngineInterface
)
from .config.config_manager import ConfigManager
from .extractors.pdf_extractor import PDFExtractor
from .extractors.content_preprocessor import ContentPreprocessor
from .analyzers.section_classifier import SectionClassifier
from .analyzers.nlp_analyzer import NLPAnalyzer
from .analyzers.industry_classifier import IndustryClassifier
from .analyzers.deck_summarizer import DeckSummarizer
from .scorers.multi_dimensional_scorer import MultiDimensionalScorer
from .results.results_aggregator import ResultsAggregator
from .visualization.visualization_engine import VisualizationEngine
from .dashboard.html_dashboard import HTMLDashboard
from .dashboard.summary_table_generator import SummaryTableGenerator


class PitchDeckEvaluator:
    """
    Main orchestrator class for pitch deck evaluation pipeline.
    Coordinates all components to process and evaluate pitch decks.
    """
    
    def __init__(self, config_path: str = "config/scoring_config.json", output_dir: str = "results"):
        """
        Initialize the pitch deck evaluator with all components.
        
        Args:
            config_path: Path to scoring configuration file
            output_dir: Directory for output files and visualizations
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self._initialize_components()
        
        # Processing statistics and quality metrics
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'processing_errors': [],
            'quality_metrics': {
                'average_extraction_quality': 0.0,
                'average_confidence_level': 0.0,
                'section_classification_success_rate': 0.0,
                'scoring_completeness': 0.0
            },
            'processing_times': {
                'pdf_extraction': [],
                'text_processing': [],
                'scoring': [],
                'total_per_deck': []
            }
        }
    
    def _setup_logging(self):
        """Set up logging configuration for the pipeline."""
        log_file = self.output_dir / "processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")
        
        # Core processing components
        self.pdf_extractor = PDFExtractor()
        self.content_preprocessor = ContentPreprocessor()
        self.section_classifier = SectionClassifier(self.config_manager)
        self.nlp_analyzer = NLPAnalyzer()
        self.scorer = MultiDimensionalScorer(self.config_manager)
        
        # Analysis components
        self.industry_classifier = IndustryClassifier()
        self.deck_summarizer = DeckSummarizer()
        
        # Results and visualization components
        self.results_aggregator = ResultsAggregator()
        self.visualizer = VisualizationEngine(str(self.output_dir / "visualizations"))
        self.dashboard_generator = HTMLDashboard()
        self.table_generator = SummaryTableGenerator()
        
        self.logger.info("All components initialized successfully")
    
    def set_components(self,
                      pdf_extractor: PDFExtractorInterface = None,
                      content_preprocessor: ContentPreprocessorInterface = None,
                      section_classifier: SectionClassifierInterface = None,
                      nlp_analyzer: NLPAnalyzerInterface = None,
                      scorer: MultiDimensionalScorerInterface = None,
                      visualizer: VisualizationEngineInterface = None) -> None:
        """Set component implementations for dependency injection (optional override)."""
        if pdf_extractor:
            self.pdf_extractor = pdf_extractor
        if content_preprocessor:
            self.content_preprocessor = content_preprocessor
        if section_classifier:
            self.section_classifier = section_classifier
        if nlp_analyzer:
            self.nlp_analyzer = nlp_analyzer
        if scorer:
            self.scorer = scorer
        if visualizer:
            self.visualizer = visualizer
    
    def evaluate_pitch_deck(self, pdf_path: str) -> Optional[EvaluationResult]:
        """
        Evaluate a single pitch deck through the complete pipeline with comprehensive error handling.
        
        Args:
            pdf_path: Path to the PDF pitch deck file
            
        Returns:
            EvaluationResult containing scores and analysis, or None if processing failed
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        deck_name = pdf_path.stem
        
        self.logger.info(f"Starting evaluation of {deck_name}")
        
        # Initialize quality metrics for this deck
        quality_metrics = {
            'extraction_quality': 0.0,
            'section_classification_success': 0.0,
            'scoring_completeness': 0.0,
            'confidence_level': 0.0
        }
        
        try:
            # Step 1: Extract text from PDF with error handling
            self.logger.info(f"Extracting text from {deck_name}")
            extraction_start = time.time()
            
            try:
                extraction_result = self.pdf_extractor.extract_text(str(pdf_path))
                extraction_time = time.time() - extraction_start
                self.processing_stats['processing_times']['pdf_extraction'].append(extraction_time)
                
                quality_metrics['extraction_quality'] = extraction_result.get('quality', 0.0)
                
                if not extraction_result.get('text'):
                    raise ValueError("No text extracted from PDF")
                
                if extraction_result.get('quality', 0) < 0.1:
                    self.logger.warning(f"Poor text extraction quality ({extraction_result.get('quality', 0):.3f}) for {deck_name}")
                    self._record_processing_error(deck_name, "Poor text extraction quality", "extraction")
                    return None
                    
            except Exception as e:
                self.logger.error(f"PDF extraction failed for {deck_name}: {str(e)}")
                self._record_processing_error(deck_name, f"PDF extraction failed: {str(e)}", "extraction")
                return None
            
            # Step 2: Preprocess content with error handling
            self.logger.info(f"Preprocessing content for {deck_name}")
            processing_start = time.time()
            
            try:
                cleaned_text = self.content_preprocessor.clean_text(extraction_result['text'])
                if not cleaned_text or len(cleaned_text.strip()) < 50:
                    raise ValueError("Insufficient text content after preprocessing")
            except Exception as e:
                self.logger.error(f"Content preprocessing failed for {deck_name}: {str(e)}")
                self._record_processing_error(deck_name, f"Content preprocessing failed: {str(e)}", "preprocessing")
                return None
            
            # Step 3: Classify sections with error handling
            self.logger.info(f"Classifying sections for {deck_name}")
            
            try:
                sections = self.section_classifier.classify_sections(cleaned_text)
                # Calculate section classification success rate
                non_empty_sections = sum(1 for section_text in sections.values() if section_text and section_text.strip())
                quality_metrics['section_classification_success'] = non_empty_sections / len(sections) if sections else 0.0
                
                if non_empty_sections == 0:
                    self.logger.warning(f"No sections successfully classified for {deck_name}")
                    # Continue with empty sections - graceful degradation
                    
            except Exception as e:
                self.logger.error(f"Section classification failed for {deck_name}: {str(e)}")
                self._record_processing_error(deck_name, f"Section classification failed: {str(e)}", "classification")
                sections = {}  # Continue with empty sections
            
            # Step 4: NLP analysis with error handling
            self.logger.info(f"Performing NLP analysis for {deck_name}")
            
            try:
                nlp_features = self._extract_nlp_features(cleaned_text, sections)
            except Exception as e:
                self.logger.error(f"NLP analysis failed for {deck_name}: {str(e)}")
                self._record_processing_error(deck_name, f"NLP analysis failed: {str(e)}", "nlp_analysis")
                nlp_features = {}  # Continue with empty features
            
            # Step 5: Score dimensions with error handling
            self.logger.info(f"Scoring dimensions for {deck_name}")
            scoring_start = time.time()
            
            try:
                section_scores = self._score_dimensions(sections, nlp_features)
                scoring_time = time.time() - scoring_start
                self.processing_stats['processing_times']['scoring'].append(scoring_time)
                
                # Calculate scoring completeness
                non_zero_scores = sum(1 for score in section_scores.__dict__.values() if score > 0)
                quality_metrics['scoring_completeness'] = non_zero_scores / len(section_scores.__dict__)
                
            except Exception as e:
                self.logger.error(f"Dimension scoring failed for {deck_name}: {str(e)}")
                self._record_processing_error(deck_name, f"Dimension scoring failed: {str(e)}", "scoring")
                # Use default scores as fallback
                from .models.data_models import SectionScores
                section_scores = SectionScores(
                    problem_clarity=1.0, market_potential=1.0, traction_strength=1.0,
                    team_experience=1.0, business_model=1.0, vision_moat=1.0, overall_confidence=1.0
                )
                quality_metrics['scoring_completeness'] = 0.0
            
            # Step 6: Industry classification and summarization with error handling
            self.logger.info(f"Classifying industry and generating summary for {deck_name}")
            
            try:
                industry_category = self.industry_classifier.classify_industry(cleaned_text)
            except Exception as e:
                self.logger.warning(f"Industry classification failed for {deck_name}: {str(e)}")
                industry_category = "Unknown"
            
            try:
                summary_points = self.deck_summarizer.generate_summary(cleaned_text, sections)
            except Exception as e:
                self.logger.warning(f"Summary generation failed for {deck_name}: {str(e)}")
                summary_points = [f"Summary generation failed for {deck_name}"]
            
            # Step 7: Create evaluation result with error handling
            try:
                composite_score = self.scorer.calculate_composite_score(section_scores.__dict__)
            except Exception as e:
                self.logger.error(f"Composite score calculation failed for {deck_name}: {str(e)}")
                composite_score = sum(section_scores.__dict__.values()) / len(section_scores.__dict__) * 10
            
            quality_metrics['confidence_level'] = extraction_result.get('quality', 0.8)
            
            result = EvaluationResult(
                deck_name=deck_name,
                section_scores=section_scores,
                composite_score=composite_score,
                investability_insight="",  # Will be generated by results aggregator
                industry_category=industry_category,
                summary_points=summary_points,
                confidence_level=quality_metrics['confidence_level']
            )
            
            # Record processing time
            total_time = time.time() - start_time
            self.processing_stats['processing_times']['total_per_deck'].append(total_time)
            self.processing_stats['processing_times']['text_processing'].append(
                time.time() - processing_start - scoring_time
            )
            
            # Update quality metrics
            self._update_quality_metrics(quality_metrics)
            
            self.processing_stats['successful_extractions'] += 1
            self.logger.info(f"Successfully evaluated {deck_name} with score {composite_score:.2f} "
                           f"(quality: {quality_metrics['confidence_level']:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error evaluating {deck_name}: {str(e)}")
            self._record_processing_error(deck_name, f"Unexpected error: {str(e)}", "general")
            return None
        finally:
            self.processing_stats['total_processed'] += 1
    
    def evaluate_multiple_decks(self, pdf_paths: List[str], max_workers: int = 3) -> List[EvaluationResult]:
        """
        Evaluate multiple pitch decks with parallel processing and progress tracking.
        
        Args:
            pdf_paths: List of paths to PDF pitch deck files
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of EvaluationResult objects (successful evaluations only)
        """
        self.logger.info(f"Starting batch evaluation of {len(pdf_paths)} pitch decks")
        
        # Reset processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'processing_errors': [],
            'quality_metrics': {
                'average_extraction_quality': 0.0,
                'average_confidence_level': 0.0,
                'section_classification_success_rate': 0.0,
                'scoring_completeness': 0.0
            },
            'processing_times': {
                'pdf_extraction': [],
                'text_processing': [],
                'scoring': [],
                'total_per_deck': []
            }
        }
        
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.evaluate_pitch_deck, pdf_path): pdf_path 
                for pdf_path in pdf_paths
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(pdf_paths), desc="Evaluating pitch decks") as pbar:
                for future in as_completed(future_to_path):
                    pdf_path = future_to_path[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {pdf_path}: {str(e)}")
                        pbar.update(1)
        
        self.logger.info(f"Batch evaluation complete: {len(results)} successful, "
                        f"{self.processing_stats['failed_extractions']} failed")
        
        return results
    
    def generate_dashboard(self, results: List[EvaluationResult]) -> str:
        """
        Generate HTML dashboard with evaluation results and visualizations.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Path to generated HTML dashboard file
        """
        if not results:
            self.logger.warning("No results to generate dashboard")
            return ""
        
        self.logger.info("Generating comprehensive dashboard and visualizations")
        
        # Aggregate and rank results
        ranked_results = self.results_aggregator.aggregate_results(results)
        
        # Generate visualizations
        viz_paths = {}
        try:
            viz_paths['radar'] = self.visualizer.create_radar_chart(
                {r.deck_name: r.section_scores.__dict__ for r in ranked_results.ranked_results}
            )
            viz_paths['histogram'] = self.visualizer.create_score_histogram(
                [r.composite_score for r in ranked_results.ranked_results]
            )
            viz_paths['heatmap'] = self.visualizer.create_correlation_heatmap(ranked_results.ranked_results)
            viz_paths['ranking'] = self.visualizer.create_ranking_chart(ranked_results.ranked_results)
        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {str(e)}")
        
        # Generate HTML dashboard
        dashboard_path = self.dashboard_generator.generate_dashboard(
            ranked_results, viz_paths, str(self.output_dir)
        )
        
        self.logger.info(f"Dashboard generated at: {dashboard_path}")
        return dashboard_path
    
    def export_results(self, results: List[EvaluationResult], format: str = "csv") -> str:
        """
        Export evaluation results to specified format.
        
        Args:
            results: List of evaluation results
            format: Export format ("csv", "excel", "json")
            
        Returns:
            Path to exported results file
        """
        if not results:
            self.logger.warning("No results to export")
            return ""
        
        # Aggregate results for export
        ranked_results = self.results_aggregator.aggregate_results(results)
        
        # Export using table generator
        if format.lower() == "excel":
            export_path = self.table_generator.export_to_excel(
                ranked_results, str(self.output_dir)
            )
        else:  # Default to CSV
            export_path = self.table_generator.export_to_csv(
                ranked_results, str(self.output_dir)
            )
        
        self.logger.info(f"Results exported to: {export_path}")
        return export_path
    
    def _extract_nlp_features(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract NLP features from text and sections."""
        features = {}
        
        # Overall text analysis
        features['sentiment'] = self.nlp_analyzer.analyze_sentiment(text)
        features['keywords'] = self.nlp_analyzer.extract_keywords(text)
        features['readability'] = self.nlp_analyzer.calculate_readability(text)
        features['confidence_indicators'] = self.nlp_analyzer.detect_confidence_indicators(text)
        features['metrics'] = self.nlp_analyzer.extract_metrics(text)
        
        # Section-specific analysis
        features['section_analysis'] = {}
        for section_name, section_text in sections.items():
            if section_text:
                features['section_analysis'][section_name] = {
                    'sentiment': self.nlp_analyzer.analyze_sentiment(section_text),
                    'keywords': self.nlp_analyzer.extract_keywords(section_text),
                    'confidence': self.nlp_analyzer.detect_confidence_indicators(section_text)
                }
        
        return features
    
    def _score_dimensions(self, sections: Dict[str, str], nlp_features: Dict[str, Any]):
        """Score all dimensions using the multi-dimensional scorer."""
        from .models.data_models import SectionScores
        
        return SectionScores(
            problem_clarity=self.scorer.score_problem_clarity(sections.get('problem', '')),
            market_potential=self.scorer.score_market_potential(sections.get('market', '')),
            traction_strength=self.scorer.score_traction_strength(sections.get('traction', '')),
            team_experience=self.scorer.score_team_experience(sections.get('team', '')),
            business_model=self.scorer.score_business_model(sections.get('business_model', '')),
            vision_moat=self.scorer.score_vision_moat(sections.get('solution', '') + ' ' + sections.get('market', '')),
            overall_confidence=self.scorer.score_overall_confidence(' '.join(sections.values()))
        )
    
    def _record_processing_error(self, deck_name: str, error_message: str, error_type: str):
        """Record a processing error with categorization."""
        error_record = {
            'deck': deck_name,
            'error': error_message,
            'type': error_type,
            'timestamp': time.time()
        }
        self.processing_stats['processing_errors'].append(error_record)
        self.processing_stats['failed_extractions'] += 1
    
    def _update_quality_metrics(self, deck_quality_metrics: Dict[str, float]):
        """Update overall quality metrics with data from a single deck."""
        current_metrics = self.processing_stats['quality_metrics']
        successful_count = self.processing_stats['successful_extractions']
        
        if successful_count == 0:
            # First successful deck
            for key, value in deck_quality_metrics.items():
                current_metrics[key.replace('_', '_')] = value
        else:
            # Running average
            for key, value in deck_quality_metrics.items():
                metric_key = key.replace('_', '_')
                if metric_key in current_metrics:
                    current_metrics[metric_key] = (
                        (current_metrics[metric_key] * (successful_count - 1) + value) / successful_count
                    )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics for the current batch."""
        stats = self.processing_stats.copy()
        
        # Calculate additional statistics
        if stats['processing_times']['total_per_deck']:
            stats['performance_metrics'] = {
                'average_processing_time': sum(stats['processing_times']['total_per_deck']) / len(stats['processing_times']['total_per_deck']),
                'fastest_processing_time': min(stats['processing_times']['total_per_deck']),
                'slowest_processing_time': max(stats['processing_times']['total_per_deck'])
            }
        
        # Calculate success rate
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_extractions'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
        
        # Categorize errors
        error_categories = {}
        for error in stats['processing_errors']:
            error_type = error.get('type', 'unknown')
            error_categories[error_type] = error_categories.get(error_type, 0) + 1
        stats['error_categories'] = error_categories
        
        return stats
    
    def generate_processing_quality_report(self) -> str:
        """Generate a comprehensive processing quality report."""
        stats = self.get_processing_statistics()
        
        report_lines = [
            "=== PITCH DECK EVALUATION PROCESSING QUALITY REPORT ===",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PROCESSING SUMMARY:",
            f"  Total Decks Processed: {stats['total_processed']}",
            f"  Successful Evaluations: {stats['successful_extractions']}",
            f"  Failed Evaluations: {stats['failed_extractions']}",
            f"  Success Rate: {stats['success_rate']:.1%}",
            ""
        ]
        
        # Quality metrics
        if stats['successful_extractions'] > 0:
            quality = stats['quality_metrics']
            report_lines.extend([
                "QUALITY METRICS:",
                f"  Average Extraction Quality: {quality['average_extraction_quality']:.3f}",
                f"  Average Confidence Level: {quality['average_confidence_level']:.3f}",
                f"  Section Classification Success Rate: {quality['section_classification_success_rate']:.1%}",
                f"  Scoring Completeness: {quality['scoring_completeness']:.1%}",
                ""
            ])
        
        # Performance metrics
        if 'performance_metrics' in stats:
            perf = stats['performance_metrics']
            report_lines.extend([
                "PERFORMANCE METRICS:",
                f"  Average Processing Time: {perf['average_processing_time']:.2f}s",
                f"  Fastest Processing Time: {perf['fastest_processing_time']:.2f}s",
                f"  Slowest Processing Time: {perf['slowest_processing_time']:.2f}s",
                ""
            ])
        
        # Error analysis
        if stats['processing_errors']:
            report_lines.extend([
                "ERROR ANALYSIS:",
                f"  Total Errors: {len(stats['processing_errors'])}",
                ""
            ])
            
            if stats['error_categories']:
                report_lines.append("  Error Categories:")
                for error_type, count in stats['error_categories'].items():
                    report_lines.append(f"    {error_type}: {count}")
                report_lines.append("")
            
            report_lines.append("  Recent Errors:")
            for error in stats['processing_errors'][-5:]:  # Show last 5 errors
                report_lines.append(f"    {error['deck']}: {error['error']}")
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
        ])
        
        if stats['success_rate'] < 0.8:
            report_lines.append("  - Success rate is below 80%. Consider reviewing PDF quality or extraction methods.")
        
        if stats['quality_metrics']['average_extraction_quality'] < 0.5:
            report_lines.append("  - Low extraction quality detected. PDFs may be image-based or corrupted.")
        
        if stats['quality_metrics']['section_classification_success_rate'] < 0.6:
            report_lines.append("  - Section classification success is low. Consider updating keyword mappings.")
        
        if not stats['processing_errors']:
            report_lines.append("  - No processing errors detected. System is operating normally.")
        
        return "\n".join(report_lines)
    
    def save_processing_quality_report(self, output_path: Optional[str] = None) -> str:
        """Save processing quality report to file."""
        if output_path is None:
            output_path = str(self.output_dir / "processing_quality_report.txt")
        
        report_content = self.generate_processing_quality_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Processing quality report saved to: {output_path}")
        return output_path
    
    def run_complete_evaluation(self, pdf_paths: List[str], generate_dashboard: bool = True, 
                               export_format: str = "csv") -> Dict[str, str]:
        """
        Run complete evaluation pipeline with all outputs.
        
        Args:
            pdf_paths: List of PDF file paths to evaluate
            generate_dashboard: Whether to generate HTML dashboard
            export_format: Format for results export ("csv" or "excel")
            
        Returns:
            Dictionary with paths to generated files
        """
        self.logger.info("Starting complete evaluation pipeline")
        
        # Evaluate all decks
        results = self.evaluate_multiple_decks(pdf_paths)
        
        if not results:
            self.logger.error("No successful evaluations - cannot generate outputs")
            return {}
        
        output_paths = {}
        
        # Export results
        export_path = self.export_results(results, export_format)
        if export_path:
            output_paths['results_export'] = export_path
        
        # Generate dashboard
        if generate_dashboard:
            dashboard_path = self.generate_dashboard(results)
            if dashboard_path:
                output_paths['dashboard'] = dashboard_path
        
        # Generate and save processing quality report
        quality_report_path = self.save_processing_quality_report()
        output_paths['quality_report'] = quality_report_path
        
        # Log processing statistics
        stats = self.get_processing_statistics()
        self.logger.info(f"Pipeline complete - Processed: {stats['total_processed']}, "
                        f"Successful: {stats['successful_extractions']}, "
                        f"Failed: {stats['failed_extractions']}, "
                        f"Success Rate: {stats['success_rate']:.1%}")
        
        return output_paths