"""
Model performance and interpretation visualizations
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
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                           precision_recall_curve, classification_report)
import logging

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """Visualize model performance and interpretations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.set_style()
    
    def set_style(self):
        """Set consistent plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.colors = {
            'train': '#3498db',
            'val': '#e74c3c', 
            'test': '#2ecc71'
        }
    
    def plot_training_history(self, train_losses: List[float], val_losses: List[float],
                            train_accuracies: List[float] = None, 
                            val_accuracies: List[float] = None,
                            model_name: str = "Model", save: bool = True):
        """Plot training and validation metrics over epochs"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title(f'{model_name} - Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot (if available)
        if train_accuracies and val_accuracies:
            axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            axes[1].set_title(f'{model_name} - Training and Validation Accuracy')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{model_name}_training_history.png', 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f'{model_name}_training_history.pdf', 
                       bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, y_true: List, y_pred: List, 
                            class_names: List[str] = None,
                            model_name: str = "Model", save: bool = True):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=f'{model_name} - Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{model_name}_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, y_true: List, y_probs: List, 
                       class_names: List[str] = None,
                       model_name: str = "Model", save: bool = True):
        """Plot ROC curves for multi-class classification"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels for multi-class ROC
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{model_name}_roc_curves.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig, roc_auc
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: List[float],
                              model_name: str = "Model", 
                              top_k: int = 20, save: bool = True):
        """Plot feature importance scores"""
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]
        top_indices = indices[:top_k]
        
        top_features = [feature_names[i] for i in top_indices]
        top_scores = [importance_scores[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_scores, align='center', color='skyblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_name} - Top {top_k} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{model_name}_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_model_dashboard(self, model_results: Dict, model_name: str = "Model"):
        """Create comprehensive model performance dashboard"""
        # Create interactive Plotly dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training History', 'Confusion Matrix', 
                          'ROC Curves', 'Feature Importance'),
            specs=[[{"type": "xy"}, {"type": "heatmap"}],
                   [{"type": "xy"}, {"type": "bar"}]]
        )
        
        # Training history
        if 'train_losses' in model_results and 'val_losses' in model_results:
            epochs = list(range(1, len(model_results['train_losses']) + 1))
            
            fig.add_trace(
                go.Scatter(x=epochs, y=model_results['train_losses'], 
                          name='Train Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=model_results['val_losses'], 
                          name='Val Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Confusion matrix
        if 'confusion_matrix' in model_results:
            cm = model_results['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=False),
                row=1, col=2
            )
        
        # ROC curves
        if 'roc_curves' in model_results:
            for class_name, curve_data in model_results['roc_curves'].items():
                fig.add_trace(
                    go.Scatter(x=curve_data['fpr'], y=curve_data['tpr'],
                              name=f'{class_name} (AUC: {curve_data["auc"]:.2f})'),
                    row=2, col=1
                )
        
        # Feature importance
        if 'feature_importance' in model_results:
            importance_data = model_results['feature_importance']
            top_features = list(importance_data.keys())[:10]
            top_scores = list(importance_data.values())[:10]
            
            fig.add_trace(
                go.Bar(x=top_scores, y=top_features, orientation='h'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text=f"{model_name} Performance Dashboard")
        fig.write_html(self.output_dir / f"{model_name}_dashboard.html")
        
        return fig