import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class QCVisualizer:
    """Generate quality control visualizations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.set_style()
    
    def set_style(self):
        """Set consistent plotting style"""
        plt.style.use('default')
        sns.set_palette("viridis")
        
        self.colors = {
            'PASS': '#2ecc71',
            'WARNING': '#f39c12', 
            'FAIL': '#e74c3c'
        }
    
    def plot_qc_summary(self, qc_data: pd.DataFrame, save: bool = True):
        """Create comprehensive QC summary dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Whole Slide Image Quality Control Summary', fontsize=16, fontweight='bold')
        
        # 1. Quality score distribution
        axes[0,0].hist(qc_data['quality_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(0.8, color='red', linestyle='--', alpha=0.7, label='Pass Threshold')
        axes[0,0].axvline(0.6, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        axes[0,0].set_xlabel('Quality Score')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Quality Score Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. QC status breakdown
        status_counts = qc_data['qc_status'].value_counts()
        axes[0,1].pie(status_counts.values, labels=status_counts.index, 
                     autopct='%1.1f%%', colors=[self.colors[s] for s in status_counts.index])
        axes[0,1].set_title('QC Status Distribution')
        
        # 3. Focus score vs tissue percentage
        colors = [self.colors[status] for status in qc_data['qc_status']]
        scatter = axes[0,2].scatter(qc_data['focus_score_level0'], qc_data['tissue_percentage_level0'],
                                   c=colors, alpha=0.6, s=60)
        axes[0,2].set_xlabel('Focus Score')
        axes[0,2].set_ylabel('Tissue Percentage')
        axes[0,2].set_title('Focus vs Tissue Content')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Artifact score distribution by group
        sns.boxplot(data=qc_data, x='group', y='artifact_score_level0', ax=axes[1,0])
        axes[1,0].set_title('Artifact Scores by Group')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Quality score by group
        sns.violinplot(data=qc_data, x='group', y='quality_score', ax=axes[1,1])
        axes[1,1].set_title('Quality Scores by Group')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Correlation heatmap
        numeric_cols = ['quality_score', 'focus_score_level0', 'tissue_percentage_level0', 'artifact_score_level0']
        corr_matrix = qc_data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('QC Metrics Correlation')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'qc_summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'qc_summary_dashboard.pdf', bbox_inches='tight')
        
        return fig
    
    def create_interactive_qc_dashboard(self, qc_data: pd.DataFrame):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Quality Score Distribution', 'Focus vs Tissue Content', 
                          'Artifact Scores', 'Quality by Group', 'Metrics Correlation'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}, {"type": "box"}],
                   [{"type": "violin"}, {"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # 1. Quality score distribution
        fig.add_trace(
            go.Histogram(x=qc_data['quality_score'], name='Quality Scores'),
            row=1, col=1
        )
        
        # 2. Focus vs tissue with status coloring
        for status in ['PASS', 'WARNING', 'FAIL']:
            status_data = qc_data[qc_data['qc_status'] == status]
            fig.add_trace(
                go.Scatter(x=status_data['focus_score_level0'], 
                          y=status_data['tissue_percentage_level0'],
                          mode='markers', name=status,
                          marker_color=self.colors[status]),
                row=1, col=2
            )
        
        # 3. Artifact scores by group
        for group in qc_data['group'].unique():
            group_data = qc_data[qc_data['group'] == group]
            fig.add_trace(
                go.Box(y=group_data['artifact_score_level0'], name=group),
                row=1, col=3
            )
        
        # 4. Quality scores by group
        for group in qc_data['group'].unique():
            group_data = qc_data[qc_data['group'] == group]
            fig.add_trace(
                go.Violin(y=group_data['quality_score'], name=group, box_visible=True),
                row=2, col=1
            )
        
        # 5. Correlation heatmap
        numeric_cols = ['quality_score', 'focus_score_level0', 'tissue_percentage_level0', 'artifact_score_level0']
        corr_matrix = qc_data[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, 
                      x=corr_matrix.columns, 
                      y=corr_matrix.columns,
                      colorscale='RdBu', zmid=0),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive QC Dashboard")
        
        # Save interactive plot
        fig.write_html(self.output_dir / "interactive_qc_dashboard.html")
        
        return fig