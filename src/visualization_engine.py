"""
Visualization engine for pitch deck evaluation results.
Generates radar charts, histograms, heatmaps, and ranking visualizations.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Optional
import os
from ..models.data_models import EvaluationResult, SectionScores
from ..results.results_aggregator import RankedResults


class VisualizationEngine:
    """
    Creates various visualizations for pitch deck evaluation results.
    Supports radar charts, histograms, correlation heatmaps, and ranking charts.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualization engine.
        
        Args:
            output_dir: Directory to save generated visualizations
        """
        self.output_dir = output_dir
        self._ensure_output_directory()
        
        # Set up matplotlib style for high-quality outputs
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_radar_chart(self, results: List[EvaluationResult], 
                          selected_decks: Optional[List[str]] = None,
                          filename: str = "radar_chart.png") -> str:
        """
        Create a radar chart showing scoring dimensions for selected decks.
        
        Args:
            results: List of evaluation results
            selected_decks: Optional list of deck names to include (default: all)
            filename: Output filename for the chart
            
        Returns:
            Path to the saved chart file
        """
        if not results:
            raise ValueError("No results provided for radar chart generation")
        
        # Filter results if specific decks are selected
        if selected_decks:
            results = [r for r in results if r.deck_name in selected_decks]
        
        if not results:
            raise ValueError("No matching results found for selected decks")
        
        # Define dimension labels and order
        dimensions = [
            'Problem\nClarity',
            'Market\nPotential', 
            'Traction\nStrength',
            'Team\nExperience',
            'Business\nModel',
            'Vision/\nMoat',
            'Overall\nConfidence'
        ]
        
        # Number of dimensions
        num_dims = len(dimensions)
        
        # Calculate angles for each dimension
        angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create figure and polar subplot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Color palette for different decks
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        
        # Plot each deck
        for i, result in enumerate(results):
            # Extract dimension scores
            scores = [
                result.section_scores.problem_clarity,
                result.section_scores.market_potential,
                result.section_scores.traction_strength,
                result.section_scores.team_experience,
                result.section_scores.business_model,
                result.section_scores.vision_moat,
                result.section_scores.overall_confidence
            ]
            scores += scores[:1]  # Complete the circle
            
            # Plot the radar chart for this deck
            ax.plot(angles, scores, 'o-', linewidth=2, 
                   label=result.deck_name, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'])
        ax.grid(True)
        
        # Add title and legend
        plt.title('Pitch Deck Evaluation - Multi-Dimensional Scoring', 
                 size=16, fontweight='bold', pad=20)
        
        # Position legend outside the plot
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the chart
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def create_multi_deck_comparison_radar(self, results: List[EvaluationResult],
                                         filename: str = "multi_deck_radar.png") -> str:
        """
        Create a radar chart comparing all decks on a single chart.
        
        Args:
            results: List of evaluation results
            filename: Output filename for the chart
            
        Returns:
            Path to the saved chart file
        """
        return self.create_radar_chart(results, selected_decks=None, filename=filename)
    
    def create_individual_radar_charts(self, results: List[EvaluationResult]) -> List[str]:
        """
        Create individual radar charts for each deck.
        
        Args:
            results: List of evaluation results
            
        Returns:
            List of paths to saved chart files
        """
        chart_paths = []
        
        for result in results:
            # Create safe filename from deck name
            safe_name = "".join(c for c in result.deck_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            filename = f"radar_{safe_name}.png"
            
            # Create radar chart for single deck
            path = self.create_radar_chart([result], filename=filename)
            chart_paths.append(path)
        
        return chart_paths
    
    def create_score_histogram(self, results: List[EvaluationResult],
                              filename: str = "score_histogram.png") -> str:
        """
        Create histogram showing final score distribution across all decks.
        
        Args:
            results: List of evaluation results
            filename: Output filename for the chart
            
        Returns:
            Path to the saved chart file
        """
        if not results:
            raise ValueError("No results provided for histogram generation")
        
        # Extract composite scores
        composite_scores = [result.composite_score for result in results]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n_bins = min(10, len(results))  # Adjust bins based on number of decks
        n, bins, patches = ax.hist(composite_scores, bins=n_bins, 
                                  alpha=0.7, color='skyblue', 
                                  edgecolor='black', linewidth=1)
        
        # Color bars based on score ranges
        for i, (patch, score) in enumerate(zip(patches, bins[:-1])):
            if score >= 80:
                patch.set_facecolor('green')
                patch.set_alpha(0.7)
            elif score >= 60:
                patch.set_facecolor('orange')
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor('red')
                patch.set_alpha(0.7)
        
        # Customize the chart
        ax.set_xlabel('Composite Score', fontsize=12)
        ax.set_ylabel('Number of Decks', fontsize=12)
        ax.set_title('Distribution of Final Composite Scores', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_score = np.mean(composite_scores)
        std_score = np.std(composite_scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_score:.1f}')
        
        # Add legend
        ax.legend()
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_score:.1f}\nStd Dev: {std_score:.1f}\nRange: {min(composite_scores):.1f} - {max(composite_scores):.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the chart
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def create_correlation_heatmap(self, results: List[EvaluationResult],
                                  filename: str = "correlation_heatmap.png") -> str:
        """
        Create correlation heatmap for dimension score relationships using seaborn.
        
        Args:
            results: List of evaluation results
            filename: Output filename for the chart
            
        Returns:
            Path to the saved chart file
        """
        if not results:
            raise ValueError("No results provided for correlation heatmap generation")
        
        # Create DataFrame with all dimension scores
        data = []
        for result in results:
            data.append({
                'Problem Clarity': result.section_scores.problem_clarity,
                'Market Potential': result.section_scores.market_potential,
                'Traction Strength': result.section_scores.traction_strength,
                'Team Experience': result.section_scores.team_experience,
                'Business Model': result.section_scores.business_model,
                'Vision/Moat': result.section_scores.vision_moat,
                'Overall Confidence': result.section_scores.overall_confidence,
                'Composite Score': result.composite_score
            })
        
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap using seaborn
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        # Customize the chart
        ax.set_title('Correlation Matrix of Scoring Dimensions', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save the chart
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def create_ranking_chart(self, ranked_results: RankedResults,
                           filename: str = "ranking_chart.png") -> str:
        """
        Create ranking bar chart for visual deck comparison.
        
        Args:
            ranked_results: Ranked results from ResultsAggregator
            filename: Output filename for the chart
            
        Returns:
            Path to the saved chart file
        """
        if not ranked_results.ranked_results:
            raise ValueError("No ranked results provided for ranking chart generation")
        
        results = ranked_results.ranked_results
        
        # Extract data for plotting
        deck_names = [result.deck_name for result in results]
        composite_scores = [result.composite_score for result in results]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create color map based on ranking
        colors = []
        for i, score in enumerate(composite_scores):
            if i < 3:  # Top 3
                colors.append('green')
            elif i >= len(composite_scores) - 3:  # Bottom 3
                colors.append('red')
            else:
                colors.append('skyblue')
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(deck_names)), composite_scores, color=colors, alpha=0.7)
        
        # Customize the chart
        ax.set_yticks(range(len(deck_names)))
        ax.set_yticklabels(deck_names)
        ax.set_xlabel('Composite Score', fontsize=12)
        ax.set_title('Pitch Deck Ranking by Composite Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, composite_scores)):
            ax.text(score + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        # Add ranking indicators
        for i, bar in enumerate(bars):
            rank = i + 1
            ax.text(2, bar.get_y() + bar.get_height()/2, 
                   f'#{rank}', ha='left', va='center', 
                   fontweight='bold', color='white', fontsize=10)
        
        # Invert y-axis to show highest scores at top
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Top 3'),
            Patch(facecolor='skyblue', alpha=0.7, label='Middle'),
            Patch(facecolor='red', alpha=0.7, label='Bottom 3')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Save the chart
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def generate_all_visualizations(self, ranked_results: RankedResults) -> Dict[str, str]:
        """
        Generate all visualization types and return paths to saved files.
        
        Args:
            ranked_results: Complete ranked results from ResultsAggregator
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        if not ranked_results.all_results:
            raise ValueError("No results provided for visualization generation")
        
        visualization_paths = {}
        
        try:
            # Generate radar chart (multi-deck comparison)
            visualization_paths['radar_chart'] = self.create_multi_deck_comparison_radar(
                ranked_results.all_results
            )
            
            # Generate score histogram
            visualization_paths['score_histogram'] = self.create_score_histogram(
                ranked_results.all_results
            )
            
            # Generate correlation heatmap
            visualization_paths['correlation_heatmap'] = self.create_correlation_heatmap(
                ranked_results.all_results
            )
            
            # Generate ranking chart
            visualization_paths['ranking_chart'] = self.create_ranking_chart(
                ranked_results
            )
            
            # Generate individual radar charts
            individual_charts = self.create_individual_radar_charts(ranked_results.all_results)
            visualization_paths['individual_radar_charts'] = individual_charts
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            raise
        
        return visualization_paths