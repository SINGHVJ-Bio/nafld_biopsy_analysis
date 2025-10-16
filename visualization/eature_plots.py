import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

class FeatureVisualizer:
    """Visualize extracted features and analysis results"""
    
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
    
    def plot_feature_distributions(self, features_df: pd.DataFrame, clinical_data: pd.DataFrame, save: bool = True):
        """Plot distributions of key features across groups"""
        # Merge with clinical data
        merged_df = features_df.merge(clinical_data, on='patient_id')
        
        # Select key features for visualization
        key_features = [
            'steatosis_area_percentage', 'ballooning_count', 
            'inflammation_area_percentage', 'fibrosis_area_percentage',
            'steatosis_mean_size', 'ballooning_mean_area'
        ]
        
        # Create subplots
        n_features = len(key_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(key_features):
            if feature in merged_df.columns:
                # Violin plot
                sns.violinplot(data=merged_df, x='group', y=feature, ax=axes[idx], palette=self.colors)
                axes[idx].set_title(f'Distribution of {feature}')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].set_visible(False)
        
        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_correlations(self, features_df: pd.DataFrame, clinical_data: pd.DataFrame, save: bool = True):
        """Plot correlation matrices and clinical correlations"""
        # Merge data
        merged_df = features_df.merge(clinical_data, on='patient_id')
        
        # Select numeric features only
        numeric_features = merged_df.select_dtypes(include=[np.number]).columns
        feature_corr = merged_df[numeric_features].corr()
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Feature-feature correlation
        sns.heatmap(feature_corr, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Feature-Feature Correlation Matrix')
        
        # Feature-clinical correlation (if clinical variables available)
        clinical_vars = ['NAS', 'steatosis_grade', 'ballooning_grade', 'inflammation_grade']  # Example
        available_clinical = [var for var in clinical_vars if var in merged_df.columns]
        
        if available_clinical:
            # Select top imaging features
            imaging_features = [col for col in features_df.columns if col not in ['patient_id']]
            top_features = imaging_features[:10]  # Top 10 for visualization
            
            clinical_corr = merged_df[top_features + available_clinical].corr()
            clinical_corr = clinical_corr.loc[available_clinical, top_features]
            
            sns.heatmap(clinical_corr, cmap='coolwarm', center=0, annot=True, fmt='.2f', ax=axes[1])
            axes[1].set_title('Feature-Clinical Correlation')
        else:
            axes[1].text(0.5, 0.5, 'Clinical variables not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Feature-Clinical Correlation')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dimensionality_reduction(self, features_df: pd.DataFrame, clinical_data: pd.DataFrame, save: bool = True):
        """Plot PCA, t-SNE, and UMAP projections"""
        # Merge data
        merged_df = features_df.merge(clinical_data, on='patient_id')
        
        # Prepare feature matrix
        feature_cols = [col for col in features_df.columns if col not in ['patient_id']]
        X = merged_df[feature_cols].fillna(0)
        
        # Apply dimensionality reduction
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=42)
        umap_reducer = umap.UMAP(random_state=42)
        
        X_pca = pca.fit_transform(X)
        X_tsne = tsne.fit_transform(X)
        X_umap = umap_reducer.fit_transform(X)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # PCA plot
        for group in merged_df['group'].unique():
            mask = merged_df['group'] == group
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           label=group, alpha=0.7, s=60, color=self.colors[group])
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[0].set_title('PCA Projection')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE plot
        for group in merged_df['group'].unique():
            mask = merged_df['group'] == group
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           label=group, alpha=0.7, s=60, color=self.colors[group])
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('t-SNE Projection')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # UMAP plot
        for group in merged_df['group'].unique():
            mask = merged_df['group'] == group
            axes[2].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                           label=group, alpha=0.7, s=60, color=self.colors[group])
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].set_title('UMAP Projection')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        
        return fig, {'pca': X_pca, 'tsne': X_tsne, 'umap': X_umap}
    
    def create_interactive_feature_explorer(self, features_df: pd.DataFrame, clinical_data: pd.DataFrame):
        """Create interactive feature explorer using Plotly"""
        merged_df = features_df.merge(clinical_data, on='patient_id')
        
        # Select features for the explorer
        feature_cols = [col for col in features_df.columns if col not in ['patient_id']]
        
        # Create interactive scatter matrix
        fig = px.scatter_matrix(
            merged_df,
            dimensions=feature_cols[:6],  # First 6 features
            color='group',
            title="Feature Scatter Matrix",
            hover_data=['patient_id']
        )
        
        fig.write_html(self.output_dir / "interactive_feature_explorer.html")
        
        return fig