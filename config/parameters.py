from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import torch
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    """Data loading and preprocessing parameters"""
    base_dir: str
    clinical_data: str
    raw_wsi_extensions: List[str]
    
    # Patch extraction
    patch_size: Tuple[int, int] = (512, 512)
    patch_level: int = 1
    overlap: float = 0.1
    min_tissue_area: float = 0.7
    
    # Stain normalization
    stain_normalization_method: str = "macenko"
    reference_image: str = "control_01.svs"
    
    # Augmentation
    augmentation: bool = True
    rotation_range: float = 180.0
    flip_horizontal: bool = True
    flip_vertical: bool = True
    color_jitter: float = 0.1

@dataclass
class QCConfig:
    """Quality control parameters"""
    min_tissue_percentage: float = 5.0
    max_background_percentage: float = 95.0
    focus_threshold: float = 0.7
    min_slide_size_mb: int = 10
    check_magnification: bool = True
    expected_magnification: float = 20.0

@dataclass
class ModelConfig:
    """Model architecture and training parameters"""
    # Segmentation
    seg_model_type: str = "unet"
    seg_input_channels: int = 3
    seg_output_channels: int = 5
    seg_filters: List[int] = None
    
    # Detection
    detection_num_classes: int = 4
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 15
    use_pretrained: bool = True
    pretrained_path: str = ""
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.seg_filters is None:
            self.seg_filters = [64, 128, 256, 512, 1024]

@dataclass
class FeatureConfig:
    """Feature extraction parameters"""
    extract_morphological: bool = True
    extract_spatial: bool = True
    extract_textural: bool = True
    min_object_size: int = 10
    spatial_bins: int = 10
    
    # Morphological features
    steatosis_features: List[str] = None
    ballooning_features: List[str] = None
    inflammation_features: List[str] = None
    fibrosis_features: List[str] = None
    
    def __post_init__(self):
        if self.steatosis_features is None:
            self.steatosis_features = ['area', 'perimeter', 'circularity', 'eccentricity', 'solidity']
        if self.ballooning_features is None:
            self.ballooning_features = ['area', 'perimeter', 'circularity', 'intensity_mean', 'intensity_std']
        if self.inflammation_features is None:
            self.inflammation_features = ['count', 'density', 'cluster_size_mean', 'cluster_size_std']
        if self.fibrosis_features is None:
            self.fibrosis_features = ['area_percentage', 'collagen_proportionate_area', 'pattern_entropy']

@dataclass
class AnalysisConfig:
    """Analysis parameters"""
    n_clusters_range: Tuple[int, int] = (2, 8)
    clustering_method: str = "kmeans"
    cv_folds: int = 5
    alpha: float = 0.05
    multiple_testing_correction: str = "fdr_bh"
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class PipelineConfig:
    """Pipeline execution parameters"""
    run_quality_control: bool = True
    run_stain_normalization: bool = True
    run_segmentation: bool = True
    run_feature_extraction: bool = True
    run_analysis: bool = True
    generate_reports: bool = True
    
    # Output settings
    save_intermediate_results: bool = True
    generate_plots: bool = True
    generate_interactive_dashboards: bool = True

class ConfigManager:
    """Manage configuration from YAML file"""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self.config_data = self._load_config()
        
        # Initialize configuration objects
        self.data = self._init_data_config()
        self.qc = self._init_qc_config()
        self.model = self._init_model_config()
        self.feature = self._init_feature_config()
        self.analysis = self._init_analysis_config()
        self.pipeline = self._init_pipeline_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_data_config(self) -> DataConfig:
        data_section = self.config_data.get('data', {})
        return DataConfig(
            base_dir=data_section.get('base_dir', ''),
            clinical_data=data_section.get('clinical_data', ''),
            raw_wsi_extensions=data_section.get('raw_wsi_extensions', ['.svs', '.ndpi']),
            patch_size=tuple(data_section.get('patch_size', [512, 512])),
            stain_normalization_method=data_section.get('stain_normalization_method', 'macenko')
        )
    
    def _init_qc_config(self) -> QCConfig:
        qc_section = self.config_data.get('quality_control', {})
        return QCConfig(
            min_tissue_percentage=qc_section.get('min_tissue_percentage', 5.0),
            max_background_percentage=qc_section.get('max_background_percentage', 95.0),
            focus_threshold=qc_section.get('focus_threshold', 0.7),
            min_slide_size_mb=qc_section.get('min_slide_size_mb', 10)
        )
    
    def _init_model_config(self) -> ModelConfig:
        model_section = self.config_data.get('models', {}).get('segmentation', {})
        training_section = self.config_data.get('training', {})
        
        return ModelConfig(
            seg_model_type=model_section.get('model_type', 'unet'),
            seg_output_channels=model_section.get('num_classes', 5),
            batch_size=training_section.get('batch_size', 8),
            learning_rate=training_section.get('learning_rate', 1e-4),
            num_epochs=training_section.get('num_epochs', 100),
            early_stopping_patience=training_section.get('early_stopping_patience', 15),
            use_pretrained=model_section.get('use_pretrained', True),
            pretrained_path=model_section.get('pretrained_path', '')
        )
    
    def _init_feature_config(self) -> FeatureConfig:
        feature_section = self.config_data.get('feature_extraction', {})
        return FeatureConfig(
            extract_morphological=feature_section.get('extract_morphological', True),
            extract_spatial=feature_section.get('extract_spatial', True),
            extract_textural=feature_section.get('extract_textural', True),
            min_object_size=feature_section.get('min_object_size', 10),
            spatial_bins=feature_section.get('spatial_bins', 10)
        )
    
    def _init_analysis_config(self) -> AnalysisConfig:
        analysis_section = self.config_data.get('analysis', {})
        return AnalysisConfig(
            n_clusters_range=tuple(analysis_section.get('n_clusters_range', [2, 8])),
            clustering_method=analysis_section.get('clustering_method', 'kmeans'),
            cv_folds=analysis_section.get('cross_validation_folds', 5),
            alpha=analysis_section.get('statistical_alpha', 0.05),
            multiple_testing_correction=analysis_section.get('multiple_testing_correction', 'fdr_bh')
        )
    
    def _init_pipeline_config(self) -> PipelineConfig:
        pipeline_section = self.config_data.get('pipeline', {})
        output_section = self.config_data.get('output', {})
        
        return PipelineConfig(
            run_quality_control=pipeline_section.get('run_quality_control', True),
            run_stain_normalization=pipeline_section.get('run_stain_normalization', True),
            run_segmentation=pipeline_section.get('run_segmentation', True),
            run_feature_extraction=pipeline_section.get('run_feature_extraction', True),
            run_analysis=pipeline_section.get('run_analysis', True),
            generate_reports=pipeline_section.get('generate_reports', True),
            save_intermediate_results=output_section.get('save_intermediate_results', True),
            generate_plots=output_section.get('generate_plots', True),
            generate_interactive_dashboards=output_section.get('generate_interactive_dashboards', True)
        )
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required paths
        if not Path(self.data.base_dir).exists():
            issues.append(f"Base data directory does not exist: {self.data.base_dir}")
        
        if not Path(self.data.clinical_data).exists():
            issues.append(f"Clinical data file does not exist: {self.data.clinical_data}")
        
        # Check model paths if using pretrained
        if self.model.use_pretrained and self.model.pretrained_path:
            if not Path(self.model.pretrained_path).exists():
                issues.append(f"Pretrained model path does not exist: {self.model.pretrained_path}")
        
        return issues

# Global configuration instance
config_manager = ConfigManager()
data_config = config_manager.data
qc_config = config_manager.qc
model_config = config_manager.model
feature_config = config_manager.feature
analysis_config = config_manager.analysis
pipeline_config = config_manager.pipeline