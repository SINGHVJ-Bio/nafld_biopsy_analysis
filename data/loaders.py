import openslide
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from ..config.paths import Paths
from ..config.parameters import data_params
import logging

logger = logging.getLogger(__name__)

class WSILoader:
    """Whole Slide Image loader with quality checks"""
    
    def __init__(self, paths: Paths):
        self.paths = paths
        self.slide_cache = {}
        
    def load_slide(self, slide_path: Path) -> openslide.OpenSlide:
        """Load WSI with error handling"""
        try:
            slide = openslide.OpenSlide(str(slide_path))
            self.slide_cache[str(slide_path)] = slide
            return slide
        except Exception as e:
            logger.error(f"Error loading slide {slide_path}: {e}")
            raise
    
    def get_slide_metadata(self, slide_path: Path) -> Dict:
        """Extract slide metadata"""
        slide = self.load_slide(slide_path)
        metadata = {
            'dimensions': slide.dimensions,
            'level_count': slide.level_count,
            'level_dimensions': slide.level_dimensions,
            'level_downsamples': slide.level_downsamples,
            'properties': dict(slide.properties)
        }
        return metadata
    
    def read_region_at_level(self, slide: openslide.OpenSlide, location: Tuple[int, int], 
                           level: int, size: Tuple[int, int]) -> np.ndarray:
        """Read region from specific level"""
        region = slide.read_region(location, level, size)
        return np.array(region)[:, :, :3]  # Remove alpha channel

class ClinicalDataLoader:
    """Load and validate clinical metadata"""
    
    def __init__(self, clinical_data_path: Path):
        self.data_path = clinical_data_path
        self.data = None
        
    def load_clinical_data(self) -> pd.DataFrame:
        """Load clinical data with validation"""
        if self.data_path.suffix == '.csv':
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.xlsx':
            self.data = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        self._validate_data()
        return self.data
    
    def _validate_data(self):
        """Validate clinical data structure"""
        required_columns = ['patient_id', 'group']  # Basic required columns
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for duplicates
        if self.data['patient_id'].duplicated().any():
            logger.warning("Duplicate patient IDs found in clinical data")
    
    def get_patient_data(self, patient_id: str) -> Dict:
        """Get data for specific patient"""
        if self.data is None:
            self.load_clinical_data()
        
        patient_data = self.data[self.data['patient_id'] == patient_id]
        if len(patient_data) == 0:
            raise ValueError(f"Patient {patient_id} not found in clinical data")
        
        return patient_data.iloc[0].to_dict()

class DataRegistry:
    """Registry for tracking all data files"""
    
    def __init__(self, paths: Paths):
        self.paths = paths
        self.registry = None
        
    def build_registry(self) -> pd.DataFrame:
        """Build comprehensive data registry"""
        registry_data = []
        
        # Scan all WSI directories
        for group_dir, group_name in [
            (self.paths.healthy, "healthy"),
            (self.paths.nafld, "nafld"), 
            (self.paths.controls, "controls")
        ]:
            for wsi_file in group_dir.glob("*.svs"):
                registry_data.append({
                    'patient_id': wsi_file.stem,
                    'wsi_path': wsi_file,
                    'group': group_name,
                    'file_size': wsi_file.stat().st_size
                })
        
        self.registry = pd.DataFrame(registry_data)
        return self.registry
    
    def get_group_files(self, group: str) -> List[Path]:
        """Get all files for a specific group"""
        if self.registry is None:
            self.build_registry()
        
        return self.registry[self.registry['group'] == group]['wsi_path'].tolist()