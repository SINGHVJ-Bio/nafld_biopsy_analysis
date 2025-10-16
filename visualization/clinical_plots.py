"""
Clinical data and correlation visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class ClinicalVisualizer:
    """Visualize clinical data and correlations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.set_style()
    
    def set_style(self):
        """Set consistent plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.colors = {
            'healthy': '#2ecc71',
            'nafld': '#e74c3c', 
            'controls': '#3498db'
        }
    
    def plot_clinical_distributions(self, clinical_data: pd.DataFrame, 
                                  group_column: str = 'group',
                                  save: bool = True):
        """Plot distributions of clinical variables across groups"""
        # Select numeric clinical variables
        numeric_cols = clinical_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != group_column]
        
        if not numeric_cols:
            logger.warning("No numeric clinical variables found")
            return None
        
        # Create subplots
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, clinical_var in enumerate(numeric_cols):
            if idx >= len(axes):
                break
                
            # Boxplot
            sns.boxplot(data=clinical_data, x=group_column, y=clinical_var, 
                       ax=axes[idx], palette=self.colors)
            axes[idx].set_title(f'Distribution of {clinical_var}')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'clinical_distributions.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, features_df: pd.DataFrame, 
                                clinical_data: pd.DataFrame,
                                top_features: int = 20,
                                save: bool = True):
        """Plot correlation heatmap between features and clinical variables"""
        # Merge data
        merged_data = features_df.merge(clinical_data, on='patient_id', how='inner')
        
        # Select top imaging features by variance
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        if len(feature_cols) > top_features:
            variances = merged_data[feature_cols].var().sort_values(ascending=False)
            top_feature_cols = variances.head(top_features).index.tolist()
        else:
            top_feature_cols = feature_cols
        
        # Select clinical variables
        clinical_cols = [col for col in clinical_data.columns 
                        if col not in ['patient_id', 'group'] and 
                        merged_data[col].dtype in [np.number]]
        
        if not clinical_cols:
            logger.warning("No numeric clinical variables found for correlation")
            return None
        
        # Calculate correlation matrix
        correlation_data = merged_data[top_feature_cols + clinical_cols].corr()
        
        # Extract feature-clinical correlations
        feature_clinical_corr = correlation_data.loc[top_feature_cols, clinical_cols]
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(feature_clinical_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(clinical_cols)))
        ax.set_yticks(np.arange(len(top_feature_cols)))
        ax.set_xticklabels(clinical_cols, rotation=45, ha='right')
        ax.set_yticklabels(top_feature_cols)
        
        # Add correlation values as text
        for i in range(len(top_feature_cols)):
            for j in range(len(clinical_cols)):
                text = ax.text(j, i, f'{feature_clinical_corr.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Feature-Clinical Variable Correlations')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'feature_clinical_correlations.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig, feature_clinical_corr
    
    def plot_biomarker_performance(self, biomarker_results: Dict, 
                                 save: bool = True):
        """Plot biomarker performance and importance"""
        if 'performance' not in biomarker_results:
            logger.warning("No performance data in biomarker results")
            return None
        
        performance = biomarker_results['performance']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation plot
        if 'correlations' in performance:
            correlations = performance['correlations']
            features = list(correlations.keys())
            corr_values = list(correlations.values())
            
            # Sort by absolute correlation
            sorted_indices = np.argsort(np.abs(corr_values))[::-1]
            features = [features[i] for i in sorted_indices]
            corr_values = [corr_values[i] for i in sorted_indices]
            
            bars = axes[0].barh(features[:10], corr_values[:10], 
                               color=['red' if x < 0 else 'blue' for x in corr_values[:10]])
            axes[0].set_xlabel('Correlation')
            axes[0].set_title('Top Biomarker Correlations')
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # Feature importance plot
        if 'feature_importances' in biomarker_results:
            importances = biomarker_results['feature_importances']
            features = list(importances.keys())
            importance_values = list(importances.values())
            
            # Sort by importance
            sorted_indices = np.argsort(np.abs(importance_values))[::-1]
            features = [features[i] for i in sorted_indices]
            importance_values = [importance_values[i] for i in sorted_indices]
            
            axes[1].barh(features[:10], importance_values[:10], color='green')
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Biomarker Importance Scores')
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'biomarker_performance.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_survival_analysis(self, survival_results: Dict, 
                             save: bool = True):
        """Plot survival analysis results"""
        if 'survival_curves' not in survival_results:
            logger.warning("No survival curves in results")
            return None
        
        survival_curves = survival_results['survival_curves']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(survival_curves)))
        
        for (group_name, curve_data), color in zip(survival_curves.items(), colors):
            times = curve_data['times']
            survival_probs = curve_data['survival_probabilities']
            
            ax.step(times, survival_probs, where='post', 
                   label=f'{group_name} (n={curve_data["sample_size"]})', 
                   color=color, linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Kaplan-Meier Survival Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add hazard ratio information if available
        if 'feature_hazard_ratios' in survival_results.get('survival_analysis', {}):
            hr_data = survival_results['survival_analysis']['feature_hazard_ratios']
            top_hr = sorted(hr_data.items(), key=lambda x: abs(np.log(x[1])), reverse=True)[:3]
            
            hr_text = "Top HRs: " + ", ".join([f"{feat}: {hr:.2f}" 
                                             for feat, hr in top_hr])
            ax.text(0.02, 0.02, hr_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'survival_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_clinical_dashboard(self, clinical_data: pd.DataFrame, 
                                features_df: pd.DataFrame,
                                analysis_results: Dict):
        """Create interactive clinical dashboard"""
        # Create Plotly dashboard with multiple tabs
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Clinical Distributions', 'Feature Correlations',
                          'Biomarker Performance', 'Survival Analysis'),
            specs=[[{"type": "box"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "xy"}]]
        )
        
        # Clinical distributions
        numeric_cols = clinical_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            clinical_var = numeric_cols[0]
            for group in clinical_data['group'].unique():
                group_data = clinical_data[clinical_data['group'] == group][clinical_var]
                fig.add_trace(
                    go.Box(y=group_data, name=group, boxpoints='outliers'),
                    row=1, col=1
                )
        
        # Feature correlations (simplified)
        if 'correlation_analysis' in analysis_results:
            corr_data = analysis_results['correlation_analysis']
            # Add simplified correlation visualization
        
        # Biomarker performance
        if 'biomarker_analysis' in analysis_results:
            biomarker_data = analysis_results['biomarker_analysis']
            # Add biomarker visualization
        
        # Survival analysis
        if 'survival_analysis' in analysis_results:
            survival_data = analysis_results['survival_analysis']
            # Add survival curves
        
        fig.update_layout(height=800, title_text="Clinical Analysis Dashboard")
        fig.write_html(self.output_dir / "clinical_dashboard.html")
        
        return fig