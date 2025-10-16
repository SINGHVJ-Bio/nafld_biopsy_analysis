"""
Annotation tools for creating and managing ground truth data
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..config.paths import Paths
import logging
import cv2

logger = logging.getLogger(__name__)

class AnnotationManager:
    """Manage annotation creation and storage"""
    
    def __init__(self, paths: Paths):
        self.paths = paths
        self.annotation_schema = self._create_annotation_schema()
    
    def _create_annotation_schema(self) -> Dict:
        """Define annotation schema for NAFLD"""
        return {
            "classes": {
                "background": 0,
                "steatosis": 1,
                "ballooning": 2,
                "inflammation": 3,
                "fibrosis": 4
            },
            "class_colors": {
                "background": [0, 0, 0],
                "steatosis": [255, 0, 0],      # Red
                "ballooning": [0, 255, 0],     # Green
                "inflammation": [0, 0, 255],   # Blue
                "fibrosis": [255, 255, 0]      # Yellow
            },
            "annotation_types": {
                "polygon": "For irregular shapes",
                "point": "For cell counting",
                "bounding_box": "For object detection",
                "mask": "For semantic segmentation"
            }
        }
    
    def create_annotation_template(self, slide_path: Path, annotator: str) -> Dict:
        """Create empty annotation template for a slide"""
        template = {
            "slide_id": slide_path.stem,
            "slide_path": str(slide_path),
            "annotator": annotator,
            "annotation_date": pd.Timestamp.now().isoformat(),
            "annotations": [],
            "quality_checks": {},
            "metadata": {}
        }
        return template
    
    def add_polygon_annotation(self, annotation_data: Dict, class_name: str, 
                             points: List[Tuple[float, float]], confidence: float = 1.0) -> Dict:
        """Add polygon annotation to annotation data"""
        annotation = {
            "type": "polygon",
            "class": class_name,
            "class_id": self.annotation_schema["classes"][class_name],
            "points": points,
            "confidence": confidence,
            "properties": {}
        }
        
        annotation_data["annotations"].append(annotation)
        return annotation_data
    
    def add_point_annotation(self, annotation_data: Dict, class_name: str,
                           point: Tuple[float, float], confidence: float = 1.0) -> Dict:
        """Add point annotation for cell counting"""
        annotation = {
            "type": "point",
            "class": class_name,
            "class_id": self.annotation_schema["classes"][class_name],
            "point": point,
            "confidence": confidence,
            "properties": {}
        }
        
        annotation_data["annotations"].append(annotation)
        return annotation_data
    
    def save_annotation(self, annotation_data: Dict, output_path: Path):
        """Save annotation to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        logger.info(f"Annotation saved to {output_path}")
    
    def load_annotation(self, annotation_path: Path) -> Dict:
        """Load annotation from JSON file"""
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        return annotation_data
    
    def convert_to_mask(self, annotation_data: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert annotation to segmentation mask"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for annotation in annotation_data["annotations"]:
            class_id = annotation["class_id"]
            
            if annotation["type"] == "polygon":
                points = np.array(annotation["points"], dtype=np.int32)
                cv2.fillPoly(mask, [points], class_id)
            elif annotation["type"] == "point":
                point = tuple(map(int, annotation["point"]))
                cv2.circle(mask, point, radius=5, color=class_id, thickness=-1)
        
        return mask
    
    def generate_annotation_report(self, annotation_dir: Path) -> pd.DataFrame:
        """Generate report of annotation statistics"""
        annotation_files = list(annotation_dir.glob("*.json"))
        report_data = []
        
        for ann_file in annotation_files:
            annotation_data = self.load_annotation(ann_file)
            
            class_counts = {}
            for class_name in self.annotation_schema["classes"].keys():
                class_counts[class_name] = 0
            
            for annotation in annotation_data["annotations"]:
                class_name = annotation["class"]
                class_counts[class_name] += 1
            
            report_entry = {
                "slide_id": annotation_data["slide_id"],
                "annotator": annotation_data["annotator"],
                "total_annotations": len(annotation_data["annotations"]),
                **class_counts
            }
            report_data.append(report_entry)
        
        return pd.DataFrame(report_data)

class AnnotationValidator:
    """Validate annotation quality and consistency"""
    
    def __init__(self, annotation_manager: AnnotationManager):
        self.annotation_manager = annotation_manager
    
    def validate_annotation(self, annotation_data: Dict) -> Dict:
        """Validate single annotation file"""
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ["slide_id", "annotator", "annotations"]
        for field in required_fields:
            if field not in annotation_data:
                issues.append(f"Missing required field: {field}")
        
        # Validate annotations
        for i, annotation in enumerate(annotation_data["annotations"]):
            annotation_issues = self._validate_single_annotation(annotation, i)
            issues.extend(annotation_issues)
        
        # Check for empty annotations
        if len(annotation_data["annotations"]) == 0:
            warnings.append("No annotations found in file")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "annotation_count": len(annotation_data["annotations"])
        }
    
    def _validate_single_annotation(self, annotation: Dict, index: int) -> List[str]:
        """Validate single annotation object"""
        issues = []
        
        # Check required annotation fields
        required_annotation_fields = ["type", "class", "class_id"]
        for field in required_annotation_fields:
            if field not in annotation:
                issues.append(f"Annotation {index}: Missing field '{field}'")
        
        # Validate class
        if "class" in annotation:
            valid_classes = self.annotation_manager.annotation_schema["classes"].keys()
            if annotation["class"] not in valid_classes:
                issues.append(f"Annotation {index}: Invalid class '{annotation['class']}'")
        
        # Validate type-specific fields
        if annotation["type"] == "polygon":
            if "points" not in annotation or len(annotation["points"]) < 3:
                issues.append(f"Annotation {index}: Polygon must have at least 3 points")
        
        elif annotation["type"] == "point":
            if "point" not in annotation:
                issues.append(f"Annotation {index}: Point annotation missing 'point' field")
        
        return issues