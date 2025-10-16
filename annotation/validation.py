"""
Validation tools for annotation quality and inter-rater reliability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import logging
from .tools import AnnotationManager

logger = logging.getLogger(__name__)

class InterRaterReliability:
    """Calculate inter-rater reliability metrics"""
    
    def __init__(self, annotation_manager: AnnotationManager):
        self.annotation_manager = annotation_manager
    
    def calculate_agreement(self, annotations1: Dict, annotations2: Dict, 
                          image_shape: Tuple[int, int]) -> Dict:
        """Calculate agreement between two annotators"""
        # Convert annotations to masks
        mask1 = self.annotation_manager.convert_to_mask(annotations1, image_shape)
        mask2 = self.annotation_manager.convert_to_mask(annotations2, image_shape)
        
        # Flatten masks for comparison
        flat_mask1 = mask1.flatten()
        flat_mask2 = mask2.flatten()
        
        # Calculate metrics
        agreement_metrics = {}
        
        # Overall agreement
        overall_agreement = np.mean(flat_mask1 == flat_mask2)
        agreement_metrics['overall_agreement'] = overall_agreement
        
        # Cohen's kappa
        kappa = cohen_kappa_score(flat_mask1, flat_mask2)
        agreement_metrics['cohens_kappa'] = kappa
        
        # Per-class agreement
        class_agreements = {}
        for class_name, class_id in self.annotation_manager.annotation_schema["classes"].items():
            if class_id == 0:  # Skip background
                continue
            
            class_mask1 = (flat_mask1 == class_id)
            class_mask2 = (flat_mask2 == class_id)
            
            if np.any(class_mask1) or np.any(class_mask2):
                class_agreement = np.mean(class_mask1 == class_mask2)
                class_agreements[class_name] = class_agreement
        
        agreement_metrics['class_agreements'] = class_agreements
        
        # Confusion matrix
        cm = confusion_matrix(flat_mask1, flat_mask2)
        agreement_metrics['confusion_matrix'] = cm.tolist()
        
        return agreement_metrics
    
    def calculate_fleiss_kappa(self, multiple_annotations: List[Dict], 
                             image_shape: Tuple[int, int]) -> float:
        """Calculate Fleiss' kappa for multiple raters"""
        # This is a simplified implementation
        # In practice, you'd need to handle multiple raters per subject
        
        if len(multiple_annotations) < 3:
            logger.warning("Fleiss' kappa requires at least 3 raters")
            return 0.0
        
        # Convert to categorical agreement matrix
        # This is a complex calculation - simplified here
        return 0.0  # Placeholder

class ConsensusAnnotation:
    """Create consensus annotations from multiple raters"""
    
    def __init__(self, annotation_manager: AnnotationManager):
        self.annotation_manager = annotation_manager
    
    def create_consensus(self, multiple_annotations: List[Dict], 
                        image_shape: Tuple[int, int], method: str = "majority_vote") -> np.ndarray:
        """Create consensus mask from multiple annotations"""
        masks = []
        
        for annotation_data in multiple_annotations:
            mask = self.annotation_manager.convert_to_mask(annotation_data, image_shape)
            masks.append(mask)
        
        masks_array = np.array(masks)
        
        if method == "majority_vote":
            consensus_mask = self._majority_vote(masks_array)
        elif method == "union":
            consensus_mask = self._union(masks_array)
        elif method == "intersection":
            consensus_mask = self._intersection(masks_array)
        else:
            raise ValueError(f"Unknown consensus method: {method}")
        
        return consensus_mask
    
    def _majority_vote(self, masks: np.ndarray) -> np.ndarray:
        """Majority vote consensus"""
        from scipy import stats
        
        # For each pixel, take the most frequent class
        mode_result = stats.mode(masks, axis=0)
        return mode_result.mode.squeeze()
    
    def _union(self, masks: np.ndarray) -> np.ndarray:
        """Union of all annotations"""
        union_mask = np.zeros_like(masks[0])
        
        for class_id in self.annotation_manager.annotation_schema["classes"].values():
            if class_id == 0:  # Skip background
                continue
            
            class_present = np.any(masks == class_id, axis=0)
            union_mask[class_present] = class_id
        
        return union_mask
    
    def _intersection(self, masks: np.ndarray) -> np.ndarray:
        """Intersection of all annotations"""
        intersection_mask = np.zeros_like(masks[0])
        
        for class_id in self.annotation_manager.annotation_schema["classes"].values():
            if class_id == 0:  # Skip background
                continue
            
            class_in_all = np.all(masks == class_id, axis=0)
            intersection_mask[class_in_all] = class_id
        
        return intersection_mask
    
    def generate_consensus_report(self, multiple_annotations: List[Dict], 
                                image_shape: Tuple[int, int]) -> pd.DataFrame:
        """Generate report on consensus process"""
        report_data = []
        irr = InterRaterReliability(self.annotation_manager)
        
        # Calculate pairwise agreements
        for i in range(len(multiple_annotations)):
            for j in range(i + 1, len(multiple_annotations)):
                agreement = irr.calculate_agreement(
                    multiple_annotations[i], multiple_annotations[j], image_shape
                )
                
                report_entry = {
                    'annotator_pair': f"{multiple_annotations[i]['annotator']}-{multiple_annotations[j]['annotator']}",
                    'overall_agreement': agreement['overall_agreement'],
                    'cohens_kappa': agreement['cohens_kappa']
                }
                report_data.append(report_entry)
        
        return pd.DataFrame(report_data)