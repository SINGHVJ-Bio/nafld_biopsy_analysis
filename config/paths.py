from pathlib import Path
import os

class Paths:
    """Manage all file paths in the project"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        
        # Raw data
        self.raw_wsis = self.base_dir / "00_raw_wsis"
        self.healthy = self.raw_wsis / "healthy"
        self.nafld = self.raw_wsis / "nafld" 
        self.controls = self.raw_wsis / "controls"
        
        # Processed data
        self.processed = self.base_dir / "01_processed"
        self.normalized = self.processed / "normalized"
        self.patches = self.processed / "patches"
        
        # Annotations
        self.annotations = self.base_dir / "02_annotations"
        self.masks = self.annotations / "masks"
        
        # Models
        self.models = self.base_dir / "03_models"
        self.segmentation_models = self.models / "segmentation"
        self.detection_models = self.models / "detection"
        
        # Results
        self.results = self.base_dir / "04_results"
        self.features = self.results / "features"
        self.analysis = self.results / "analysis"
        self.visualizations = self.results / "visualizations"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        for attr_name in dir(self):
            if not attr_name.startswith('_') and isinstance(getattr(self, attr_name), Path):
                path = getattr(self, attr_name)
                path.mkdir(parents=True, exist_ok=True)
    
    def get_patient_paths(self, patient_id: str) -> dict:
        """Get all relevant paths for a patient"""
        return {
            'raw_wsi': self.raw_wsis / f"{patient_id}.svs",
            'normalized': self.normalized / f"{patient_id}.tiff",
            'patches': self.patches / patient_id,
            'features': self.features / f"{patient_id}_features.csv",
            'annotations': self.annotations / f"{patient_id}.json"
        }