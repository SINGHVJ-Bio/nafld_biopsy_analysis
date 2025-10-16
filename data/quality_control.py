import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from skimage import measure, filters
import matplotlib.pyplot as plt
from ..config.parameters import data_params
from .preprocessing import TissueDetector
import logging
import json

logger = logging.getLogger(__name__)

class QualityControl:
    """Comprehensive quality control for whole slide images"""
    
    def __init__(self):
        self.tissue_detector = TissueDetector()
        self.qc_metrics = {}
    
    def analyze_slide_quality(self, slide, slide_path: Path) -> Dict:
        """Perform comprehensive QC analysis on slide"""
        metrics = {
            'slide_path': str(slide_path),
            'dimensions': slide.dimensions,
            'level_count': slide.level_count,
        }
        
        # Analyze each level
        for level in range(slide.level_count):
            level_metrics = self._analyze_level(slide, level)
            metrics[f'level_{level}'] = level_metrics
        
        # Overall quality score
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        metrics['qc_status'] = self._determine_qc_status(metrics)
        
        self.qc_metrics[str(slide_path)] = metrics
        return metrics
    
    def _analyze_level(self, slide, level: int) -> Dict:
        """Analyze quality at specific level"""
        try:
            # Get thumbnail for analysis
            thumb_size = slide.level_dimensions[level]
            thumbnail = np.array(slide.read_region((0, 0), level, thumb_size))[:, :, :3]
            
            # Calculate metrics
            focus_score = self._calculate_focus_score(thumbnail)
            tissue_percentage = self.tissue_detector.calculate_tissue_percentage(
                self.tissue_detector.detect_tissue_regions(thumbnail)
            )
            color_stats = self._analyze_color_distribution(thumbnail)
            artifact_score = self._detect_artifacts(thumbnail)
            
            return {
                'focus_score': focus_score,
                'tissue_percentage': tissue_percentage,
                'color_mean': color_stats['mean'],
                'color_std': color_stats['std'],
                'artifact_score': artifact_score,
                'dimensions': slide.level_dimensions[level]
            }
        except Exception as e:
            logger.error(f"Error analyzing level {level}: {e}")
            return {}
    
    def _calculate_focus_score(self, image: np.ndarray) -> float:
        """Calculate image focus/blur score using variance of Laplacian"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _analyze_color_distribution(self, image: np.ndarray) -> Dict:
        """Analyze color distribution in image"""
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        return {'mean': mean.tolist(), 'std': std.tolist()}
    
    def _detect_artifacts(self, image: np.ndarray) -> float:
        """Detect artifacts like folds, tears, bubbles"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for artifact detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture analysis for unusual patterns
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combine metrics
        artifact_score = edge_density * 0.7 + (1 / (1 + laplacian_var)) * 0.3
        return float(artifact_score)
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-1)"""
        scores = []
        
        for level in range(metrics['level_count']):
            level_metrics = metrics.get(f'level_{level}', {})
            if level_metrics:
                # Focus score (normalized)
                focus_norm = min(level_metrics['focus_score'] / 1000, 1.0)
                
                # Tissue percentage
                tissue_score = level_metrics['tissue_percentage']
                
                # Artifact score (inverse)
                artifact_score = 1 - min(level_metrics['artifact_score'], 1.0)
                
                level_score = (focus_norm + tissue_score + artifact_score) / 3
                scores.append(level_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _determine_qc_status(self, metrics: Dict) -> str:
        """Determine QC status based on metrics"""
        quality_score = metrics['quality_score']
        
        if quality_score >= 0.8:
            return "PASS"
        elif quality_score >= 0.6:
            return "WARNING"
        else:
            return "FAIL"
    
    def generate_qc_report(self, output_path: Path):
        """Generate comprehensive QC report"""
        if not self.qc_metrics:
            logger.warning("No QC metrics available. Run analysis first.")
            return
        
        report_data = []
        for slide_path, metrics in self.qc_metrics.items():
            report_data.append({
                'slide': Path(slide_path).name,
                'dimensions': metrics['dimensions'],
                'quality_score': metrics['quality_score'],
                'qc_status': metrics['qc_status'],
                'focus_score_level0': metrics.get('level_0', {}).get('focus_score', 0),
                'tissue_percentage_level0': metrics.get('level_0', {}).get('tissue_percentage', 0),
                'artifact_score_level0': metrics.get('level_0', {}).get('artifact_score', 0),
            })
        
        df = pd.DataFrame(report_data)
        df.to_csv(output_path / "qc_report.csv", index=False)
        
        # Create summary
        summary = {
            'total_slides': len(df),
            'pass_count': len(df[df['qc_status'] == 'PASS']),
            'warning_count': len(df[df['qc_status'] == 'WARNING']),
            'fail_count': len(df[df['qc_status'] == 'FAIL']),
            'mean_quality_score': df['quality_score'].mean(),
        }
        
        with open(output_path / "qc_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return df, summary