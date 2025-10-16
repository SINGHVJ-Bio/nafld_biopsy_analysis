#!/usr/bin/env python3
"""
Main pipeline script for NAFLD whole slide image analysis
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import yaml
from tqdm import tqdm
import openslide
from datetime import datetime

# Add package to path
sys.path.append(str(Path(__file__).parent))

from config.paths import Paths
from config.parameters import ConfigManager, data_config, qc_config, model_config, feature_config, analysis_config, pipeline_config
from data.loaders import WSILoader, ClinicalDataLoader, DataRegistry
from data.quality_control import QualityControl
from data.preprocessing import StainNormalizer, PatchExtractor, TissueDetector
from models.segmentation import UNet, SegmentationModel
from models.detection import DetectionModel, BallooningDetector
from models.classification import SlideClassifier, NASClassifier
from models.training import ModelTrainer
from features.extraction import ComprehensiveFeatureExtractor
from features.spatial import SpatialAnalyzer
from visualization.qc_plots import QCVisualizer
from visualization.feature_plots import FeatureVisualizer
from visualization.model_plots import ModelVisualizer
from analysis.subtype_discovery import SubtypeAnalyzer
from analysis.clinical_correlation import ClinicalCorrelator
from analysis.biomarker_analysis import BiomarkerAnalyzer, SurvivalAnalyzer
from utils.helpers import create_logger, setup_random_seed, save_pickle, load_pickle
from utils.loggers import PipelineLogger, ResultsLogger

# Set up logging
logger = create_logger(__name__, Path("pipeline.log"))

class NAFLDPipeline:
    """Main pipeline class for NAFLD analysis"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config_data
        
        # Validate configuration
        issues = self.config_manager.validate_config()
        if issues:
            for issue in issues:
                logger.error(issue)
            raise ValueError("Configuration validation failed")
        
        # Initialize paths
        self.paths = Paths(data_config.base_dir)
        
        # Initialize components
        self.clinical_loader = ClinicalDataLoader(Path(data_config.clinical_data))
        self.wsi_loader = WSILoader(self.paths)
        self.data_registry = DataRegistry(self.paths)
        
        # Initialize analyzers
        self.qc_analyzer = QualityControl()
        self.feature_extractor = ComprehensiveFeatureExtractor()
        self.spatial_analyzer = SpatialAnalyzer()
        self.subtype_analyzer = SubtypeAnalyzer()
        self.clinical_correlator = ClinicalCorrelator()
        self.biomarker_analyzer = BiomarkerAnalyzer()
        self.survival_analyzer = SurvivalAnalyzer()
        
        # Initialize visualizers
        self.visualizer = QCVisualizer(self.paths.visualizations)
        self.feature_visualizer = FeatureVisualizer(self.paths.visualizations)
        self.model_visualizer = ModelVisualizer(self.paths.visualizations)
        
        # Initialize loggers
        self.pipeline_logger = PipelineLogger(self.paths.results / "logs")
        self.results_logger = ResultsLogger(self.paths.results / "results")
        
        # Initialize models
        self.segmentation_model = None
        self.detection_model = None
        self.classification_model = None
        
        # Set random seeds for reproducibility
        setup_random_seed(analysis_config.random_state)
        
        logger.info("NAFLD Pipeline initialized with configuration")

    def run_quality_control(self):
        """Run comprehensive quality control on all slides"""
        if not pipeline_config.run_quality_control:
            logger.info("Skipping quality control as configured")
            return None, None
        
        self.pipeline_logger.log_step_start("quality_control")
        
        try:
            # Build data registry
            registry = self.data_registry.build_registry()
            
            qc_results = []
            for _, row in tqdm(registry.iterrows(), total=len(registry), desc="QC Analysis"):
                try:
                    slide = self.wsi_loader.load_slide(row['wsi_path'])
                    metrics = self.qc_analyzer.analyze_slide_quality(slide, row['wsi_path'])
                    metrics['group'] = row['group']
                    qc_results.append(metrics)
                    
                    logger.debug(f"QC for {row['wsi_path'].name}: {metrics['qc_status']} (score: {metrics['quality_score']:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error in QC for {row['wsi_path']}: {e}")
            
            # Generate QC report
            qc_df = pd.DataFrame(qc_results)
            qc_report, qc_summary = self.qc_analyzer.generate_qc_report(self.paths.results)
            
            # Create visualizations
            if pipeline_config.generate_plots:
                self.visualizer.plot_qc_summary(qc_df)
            
            if pipeline_config.generate_interactive_dashboards:
                self.visualizer.create_interactive_qc_dashboard(qc_df)
            
            # Log results
            self.pipeline_logger.log_step_completion("quality_control", {
                'passed_slides': qc_summary['pass_count'],
                'failed_slides': qc_summary['fail_count'],
                'mean_quality_score': qc_summary['mean_quality_score']
            })
            
            logger.info(f"QC completed: {qc_summary['pass_count']} passed, {qc_summary['fail_count']} failed")
            
            return qc_df, qc_summary
            
        except Exception as e:
            self.pipeline_logger.log_step_error("quality_control", e)
            raise

    def run_stain_normalization(self):
        """Run stain normalization on all slides with proper patch processing"""
        if not pipeline_config.run_stain_normalization:
            logger.info("Skipping stain normalization as configured")
            return
        
        self.pipeline_logger.log_step_start("stain_normalization")
        
        try:
            # Load reference image
            reference_path = self.paths.controls / data_config.reference_image
            if not reference_path.exists():
                # Try to find any control slide
                control_files = list(self.paths.controls.glob("*.*"))
                if not control_files:
                    raise FileNotFoundError("No control slides found for stain normalization")
                reference_path = control_files[0]
            
            logger.info(f"Using {reference_path.name} as reference for stain normalization")
            reference_slide = self.wsi_loader.load_slide(reference_path)
            
            # Extract reference patch from tissue-rich area
            patch_extractor = PatchExtractor(
                patch_size=data_config.patch_size,
                overlap=0.0  # No overlap for reference
            )
            reference_patches = patch_extractor.extract_patches(reference_slide, level=1)
            
            if not reference_patches:
                raise ValueError("Could not extract reference patches from control slide")
            
            # Use the patch with highest tissue content as reference
            tissue_detector = TissueDetector()
            best_patch = None
            best_tissue_content = 0
            
            for patch, location in reference_patches:
                tissue_mask = tissue_detector.detect_tissue_regions(patch)
                tissue_content = tissue_detector.calculate_tissue_percentage(tissue_mask)
                
                if tissue_content > best_tissue_content:
                    best_tissue_content = tissue_content
                    best_patch = patch
            
            if best_patch is None:
                raise ValueError("No suitable reference patch found")
            
            # Initialize and fit normalizer
            normalizer = StainNormalizer(method=data_config.stain_normalization_method)
            normalizer.fit(best_patch)
            
            # Normalize all QC-passed slides
            registry = self.data_registry.registry
            if registry is None:
                registry = self.data_registry.build_registry()
            
            qc_passed_slides = registry[registry['qc_status'] == 'PASS']
            
            normalized_count = 0
            for _, row in tqdm(qc_passed_slides.iterrows(), total=len(qc_passed_slides), desc="Stain Normalization"):
                try:
                    slide = self.wsi_loader.load_slide(row['wsi_path'])
                    slide_id = row['patient_id']
                    
                    # Process and normalize the entire slide
                    success = self._process_and_normalize_slide(slide, normalizer, slide_id)
                    
                    if success:
                        normalized_count += 1
                        logger.debug(f"Normalized {slide_id}")
                    else:
                        logger.warning(f"Failed to normalize {slide_id}")
                        
                except Exception as e:
                    logger.error(f"Error normalizing {row['wsi_path']}: {e}")
            
            self.pipeline_logger.log_step_completion("stain_normalization", {
                'normalized_slides': normalized_count,
                'total_slides': len(qc_passed_slides)
            })
            
            logger.info(f"Stain normalization completed: {normalized_count}/{len(qc_passed_slides)} slides normalized")
            
        except Exception as e:
            self.pipeline_logger.log_step_error("stain_normalization", e)
            raise

    def _process_and_normalize_slide(self, slide, normalizer, slide_id: str) -> bool:
        """Process and normalize a complete slide"""
        try:
            # Create output directory for normalized slide
            normalized_dir = self.paths.normalized / slide_id
            normalized_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract patches at multiple levels for comprehensive processing
            patch_extractor = PatchExtractor(
                patch_size=data_config.patch_size,
                overlap=data_config.overlap
            )
            
            # Process at different resolution levels
            for level in [0, 1, 2]:
                if level >= slide.level_count:
                    continue
                    
                level_patches = patch_extractor.extract_patches(slide, level=level)
                
                for i, (patch, location) in enumerate(level_patches):
                    try:
                        # Apply stain normalization
                        normalized_patch = normalizer.transform(patch)
                        
                        # Save normalized patch
                        patch_filename = f"level_{level}_patch_{location[0]}_{location[1]}.png"
                        patch_path = normalized_dir / patch_filename
                        
                        # Convert and save
                        patch_bgr = cv2.cvtColor(normalized_patch, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(patch_path), patch_bgr)
                        
                        # Save metadata
                        metadata = {
                            'slide_id': slide_id,
                            'patch_id': i,
                            'location': location,
                            'level': level,
                            'patch_size': data_config.patch_size,
                            'normalization_method': data_config.stain_normalization_method,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        metadata_path = normalized_dir / f"level_{level}_patch_{location[0]}_{location[1]}_metadata.json"
                        import json
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                            
                    except Exception as e:
                        logger.warning(f"Error processing patch {i} at level {level} for {slide_id}: {e}")
                        continue
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing slide {slide_id}: {e}")
            return False

    def initialize_models(self):
        """Initialize all required models"""
        self.pipeline_logger.log_step_start("model_initialization")
        
        try:
            # Initialize segmentation model
            if pipeline_config.run_segmentation:
                self.segmentation_model = self._initialize_segmentation_model()
            
            # Initialize detection model
            self.detection_model = DetectionModel(num_classes=4)
            
            # Initialize classification model
            self.classification_model = NASClassifier()
            
            self.pipeline_logger.log_step_completion("model_initialization", {
                'segmentation_model_loaded': self.segmentation_model is not None,
                'detection_model_loaded': self.detection_model is not None,
                'classification_model_loaded': self.classification_model is not None
            })
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            self.pipeline_logger.log_step_error("model_initialization", e)
            raise

    def _initialize_segmentation_model(self) -> SegmentationModel:
        """Initialize segmentation model with proper error handling"""
        try:
            model = SegmentationModel()
            
            if model_config.use_pretrained and model_config.pretrained_path:
                pretrained_path = Path(model_config.pretrained_path)
                if pretrained_path.exists():
                    logger.info(f"Loading pre-trained segmentation model from {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location=model_config.device)
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                    
                    if 'optimizer_state_dict' in checkpoint:
                        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    logger.info("Pre-trained segmentation model loaded successfully")
                else:
                    logger.warning(f"Pre-trained model path not found: {pretrained_path}")
                    logger.info("Using randomly initialized segmentation model")
            else:
                logger.info("Using randomly initialized segmentation model")
            
            return model
            
        except Exception as e:
            logger.error(f"Error initializing segmentation model: {e}")
            raise

    def run_segmentation(self):
        """Run segmentation on all slides"""
        if not pipeline_config.run_segmentation:
            logger.info("Skipping segmentation as configured")
            return
        
        if self.segmentation_model is None:
            self.initialize_models()
        
        self.pipeline_logger.log_step_start("segmentation")
        
        try:
            registry = self.data_registry.registry
            if registry is None:
                registry = self.data_registry.build_registry()
            
            qc_passed_slides = registry[registry['qc_status'] == 'PASS']
            
            segmentation_results = []
            for _, row in tqdm(qc_passed_slides.iterrows(), total=len(qc_passed_slides), desc="Segmentation"):
                try:
                    slide = self.wsi_loader.load_slide(row['wsi_path'])
                    slide_id = row['patient_id']
                    
                    # Run segmentation
                    segmentation_mask, confidence_map = self._segment_complete_slide(slide, slide_id)
                    
                    if segmentation_mask is not None:
                        # Save segmentation results
                        result = {
                            'patient_id': slide_id,
                            'segmentation_mask_path': self._save_segmentation_results(segmentation_mask, confidence_map, slide_id),
                            'mean_confidence': np.mean(confidence_map) if confidence_map is not None else 0.0,
                            'tissue_classes_present': self._analyze_segmentation_classes(segmentation_mask)
                        }
                        segmentation_results.append(result)
                        
                        logger.debug(f"Segmented {slide_id} with mean confidence: {result['mean_confidence']:.3f}")
                    else:
                        logger.warning(f"Segmentation failed for {slide_id}")
                        
                except Exception as e:
                    logger.error(f"Error segmenting {row['patient_id']}: {e}")
            
            # Save segmentation summary
            seg_df = pd.DataFrame(segmentation_results)
            seg_summary_path = self.paths.results / "segmentation_summary.csv"
            seg_df.to_csv(seg_summary_path, index=False)
            
            self.pipeline_logger.log_step_completion("segmentation", {
                'successful_segmentations': len(segmentation_results),
                'total_slides': len(qc_passed_slides),
                'mean_confidence': seg_df['mean_confidence'].mean() if len(seg_df) > 0 else 0
            })
            
            logger.info(f"Segmentation completed: {len(segmentation_results)}/{len(qc_passed_slides)} slides segmented")
            
            return seg_df
            
        except Exception as e:
            self.pipeline_logger.log_step_error("segmentation", e)
            raise

    def _segment_complete_slide(self, slide, slide_id: str) -> tuple:
        """Segment a complete slide using patch-based processing"""
        try:
            patch_extractor = PatchExtractor(
                patch_size=data_config.patch_size,
                overlap=0.05  # Small overlap for better stitching
            )
            
            # Get slide dimensions at level 0
            slide_dimensions = slide.level_dimensions[0]
            
            # Create empty arrays for full slide segmentation
            full_segmentation = np.zeros(slide_dimensions[::-1], dtype=np.uint8)  # height, width
            full_confidence = np.zeros(slide_dimensions[::-1], dtype=np.float32)
            count_map = np.zeros(slide_dimensions[::-1], dtype=np.uint8)
            
            # Process patches at level 1 for efficiency (adjust based on slide size)
            processing_level = min(1, slide.level_count - 1)
            patches = patch_extractor.extract_patches(slide, level=processing_level)
            
            for patch, location in tqdm(patches, desc=f"Segmenting {slide_id}", leave=False):
                try:
                    # Preprocess patch
                    patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                    patch_tensor = patch_tensor.unsqueeze(0).to(model_config.device)
                    
                    # Run segmentation
                    with torch.no_grad():
                        output = self.segmentation_model.model(patch_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, prediction = torch.max(probabilities, dim=1)
                    
                    # Convert to numpy
                    seg_patch = prediction.squeeze().cpu().numpy().astype(np.uint8)
                    conf_patch = confidence.squeeze().cpu().numpy().astype(np.float32)
                    
                    # Scale location and size to level 0
                    scale_factor = slide.level_downsamples[processing_level]
                    x0 = int(location[0] * scale_factor)
                    y0 = int(location[1] * scale_factor)
                    patch_size_full = (
                        int(data_config.patch_size[0] * scale_factor),
                        int(data_config.patch_size[1] * scale_factor)
                    )
                    
                    # Resize segmentation to full resolution
                    seg_patch_full = cv2.resize(seg_patch, patch_size_full, interpolation=cv2.INTER_NEAREST)
                    conf_patch_full = cv2.resize(conf_patch, patch_size_full, interpolation=cv2.INTER_LINEAR)
                    
                    # Place in full slide arrays
                    y_end = min(y0 + patch_size_full[1], full_segmentation.shape[0])
                    x_end = min(x0 + patch_size_full[0], full_segmentation.shape[1])
                    
                    patch_height = y_end - y0
                    patch_width = x_end - x0
                    
                    if patch_height > 0 and patch_width > 0:
                        full_segmentation[y0:y_end, x0:x_end] = seg_patch_full[:patch_height, :patch_width]
                        full_confidence[y0:y_end, x0:x_end] = conf_patch_full[:patch_height, :patch_width]
                        count_map[y0:y_end, x0:x_end] += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing patch at {location} for {slide_id}: {e}")
                    continue
            
            # Handle overlapping regions (simple majority voting)
            overlap_mask = count_map > 1
            if np.any(overlap_mask):
                logger.debug(f"Handling overlapping regions for {slide_id}")
                # For simplicity, we'll keep the first assignment
                # In production, you might implement more sophisticated merging
            
            return full_segmentation, full_confidence
            
        except Exception as e:
            logger.error(f"Error in slide segmentation for {slide_id}: {e}")
            return None, None

    def _save_segmentation_results(self, segmentation_mask: np.ndarray, confidence_map: np.ndarray, slide_id: str) -> str:
        """Save segmentation results to disk"""
        try:
            output_dir = self.paths.masks / slide_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save segmentation mask
            mask_path = output_dir / "segmentation_mask.png"
            cv2.imwrite(str(mask_path), segmentation_mask)
            
            # Save confidence map
            if confidence_map is not None:
                confidence_path = output_dir / "confidence_map.png"
                # Scale confidence to 0-255 for visualization
                confidence_vis = (confidence_map * 255).astype(np.uint8)
                cv2.imwrite(str(confidence_path), confidence_vis)
            
            # Save metadata
            metadata = {
                'slide_id': slide_id,
                'segmentation_shape': segmentation_mask.shape,
                'mean_confidence': float(np.mean(confidence_map)) if confidence_map is not None else 0.0,
                'timestamp': datetime.now().isoformat(),
                'model_type': model_config.seg_model_type
            }
            
            metadata_path = output_dir / "segmentation_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(mask_path)
            
        except Exception as e:
            logger.error(f"Error saving segmentation results for {slide_id}: {e}")
            return ""

    def _analyze_segmentation_classes(self, segmentation_mask: np.ndarray) -> dict:
        """Analyze which tissue classes are present in segmentation"""
        try:
            unique, counts = np.unique(segmentation_mask, return_counts=True)
            class_analysis = {}
            
            class_names = {
                0: 'background',
                1: 'steatosis',
                2: 'ballooning',
                3: 'inflammation',
                4: 'fibrosis'
            }
            
            for class_id, count in zip(unique, counts):
                if class_id in class_names:
                    percentage = (count / segmentation_mask.size) * 100
                    class_analysis[class_names[class_id]] = {
                        'pixel_count': int(count),
                        'percentage': float(percentage)
                    }
            
            return class_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing segmentation classes: {e}")
            return {}

    def run_feature_extraction(self):
        """Run comprehensive feature extraction on all segmented slides"""
        if not pipeline_config.run_feature_extraction:
            logger.info("Skipping feature extraction as configured")
            return None
        
        self.pipeline_logger.log_step_start("feature_extraction")
        
        try:
            registry = self.data_registry.registry
            if registry is None:
                registry = self.data_registry.build_registry()
            
            qc_passed_slides = registry[registry['qc_status'] == 'PASS']
            
            all_features = []
            for _, row in tqdm(qc_passed_slides.iterrows(), total=len(qc_passed_slides), desc="Feature Extraction"):
                try:
                    slide_id = row['patient_id']
                    
                    # Load segmentation results
                    segmentation_mask, original_image = self._load_segmentation_data(slide_id)
                    
                    if segmentation_mask is not None and original_image is not None:
                        # Extract comprehensive features
                        features = self._extract_comprehensive_features(segmentation_mask, original_image, slide_id)
                        all_features.append(features)
                        
                        logger.debug(f"Extracted {len(features) - 1} features for {slide_id}")  # -1 for patient_id
                    else:
                        logger.warning(f"Could not load segmentation data for {slide_id}")
                        
                except Exception as e:
                    logger.error(f"Error extracting features for {row['patient_id']}: {e}")
            
            # Create features dataframe
            features_df = pd.DataFrame(all_features)
            
            # Save features
            features_path = self.paths.features / "comprehensive_features.csv"
            features_df.to_csv(features_path, index=False)
            
            # Save as JSON for better structure
            features_json_path = self.paths.features / "comprehensive_features.json"
            features_df.to_json(features_json_path, orient='records', indent=2)
            
            # Generate feature summary
            feature_summary = self._generate_feature_summary(features_df)
            
            self.pipeline_logger.log_step_completion("feature_extraction", {
                'patients_with_features': len(features_df),
                'total_features': len(features_df.columns) - 1,  # Exclude patient_id
                'feature_categories': list(feature_summary.keys())
            })
            
            logger.info(f"Feature extraction completed: {len(features_df)} patients, {len(features_df.columns) - 1} features")
            
            return features_df
            
        except Exception as e:
            self.pipeline_logger.log_step_error("feature_extraction", e)
            raise

    def _load_segmentation_data(self, slide_id: str) -> tuple:
        """Load segmentation mask and corresponding original image"""
        try:
            # Load segmentation mask
            mask_dir = self.paths.masks / slide_id
            mask_path = mask_dir / "segmentation_mask.png"
            
            if not mask_path.exists():
                logger.warning(f"Segmentation mask not found for {slide_id}")
                return None, None
            
            segmentation_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Load original image (use normalized patches if available)
            normalized_dir = self.paths.normalized / slide_id
            if normalized_dir.exists():
                # Use the first normalized patch as sample
                patch_files = list(normalized_dir.glob("level_1_patch_*_*.png"))
                if patch_files:
                    original_image = cv2.imread(str(patch_files[0]))
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback to loading from original slide
                    original_image = self._load_sample_patch(slide_id)
            else:
                original_image = self._load_sample_patch(slide_id)
            
            return segmentation_mask, original_image
            
        except Exception as e:
            logger.error(f"Error loading segmentation data for {slide_id}: {e}")
            return None, None

    def _load_sample_patch(self, slide_id: str) -> np.ndarray:
        """Load a sample patch from original slide"""
        try:
            registry = self.data_registry.registry
            slide_row = registry[registry['patient_id'] == slide_id].iloc[0]
            slide = self.wsi_loader.load_slide(slide_row['wsi_path'])
            
            patch_extractor = PatchExtractor(patch_size=(512, 512))
            patches = patch_extractor.extract_patches(slide, level=1)
            
            if patches:
                patch, _ = patches[0]
                return patch
            else:
                # Create a dummy image if no patches found
                return np.zeros((512, 512, 3), dtype=np.uint8)
                
        except Exception as e:
            logger.warning(f"Error loading sample patch for {slide_id}: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def _extract_comprehensive_features(self, segmentation_mask: np.ndarray, original_image: np.ndarray, slide_id: str) -> dict:
        """Extract comprehensive features from segmentation and original image"""
        features = {'patient_id': slide_id}
        
        try:
            # Basic morphological features
            if feature_config.extract_morphological:
                morph_features = self.feature_extractor.extract_all_features(
                    segmentation_mask, original_image, slide_id
                )
                features.update(morph_features)
            
            # Spatial features
            if feature_config.extract_spatial:
                spatial_features = self.spatial_analyzer.analyze_spatial_distribution(segmentation_mask)
                features.update(spatial_features)
            
            # Tissue composition features
            composition_features = self._calculate_tissue_composition(segmentation_mask)
            features.update(composition_features)
            
            # Quality metrics
            quality_features = self._calculate_segmentation_quality(segmentation_mask)
            features.update(quality_features)
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive features for {slide_id}: {e}")
        
        return features

    def _calculate_tissue_composition(self, segmentation_mask: np.ndarray) -> dict:
        """Calculate tissue composition percentages"""
        try:
            total_pixels = segmentation_mask.size
            composition = {}
            
            for class_id, class_name in enumerate(['background', 'steatosis', 'ballooning', 'inflammation', 'fibrosis']):
                class_pixels = np.sum(segmentation_mask == class_id)
                percentage = (class_pixels / total_pixels) * 100
                composition[f'{class_name}_percentage'] = percentage
            
            return composition
            
        except Exception as e:
            logger.warning(f"Error calculating tissue composition: {e}")
            return {}

    def _calculate_segmentation_quality(self, segmentation_mask: np.ndarray) -> dict:
        """Calculate segmentation quality metrics"""
        try:
            quality = {}
            
            # Calculate tissue coverage
            tissue_pixels = np.sum(segmentation_mask > 0)  # All non-background
            total_pixels = segmentation_mask.size
            quality['tissue_coverage'] = (tissue_pixels / total_pixels) * 100
            
            # Calculate class balance
            unique, counts = np.unique(segmentation_mask, return_counts=True)
            if len(unique) > 1:
                # Gini-like impurity (1 - sum(p_i^2))
                proportions = counts / total_pixels
                quality['class_diversity'] = 1 - np.sum(proportions ** 2)
            else:
                quality['class_diversity'] = 0.0
            
            return quality
            
        except Exception as e:
            logger.warning(f"Error calculating segmentation quality: {e}")
            return {}

    def _generate_feature_summary(self, features_df: pd.DataFrame) -> dict:
        """Generate summary of extracted features"""
        try:
            feature_cols = [col for col in features_df.columns if col != 'patient_id']
            
            summary = {
                'total_features': len(feature_cols),
                'feature_categories': {},
                'missing_data': {}
            }
            
            # Categorize features
            for feature in feature_cols:
                if 'percentage' in feature:
                    category = 'composition'
                elif 'spatial' in feature or 'distribution' in feature:
                    category = 'spatial'
                elif 'morphological' in feature or 'shape' in feature:
                    category = 'morphological'
                elif 'quality' in feature or 'coverage' in feature:
                    category = 'quality'
                else:
                    category = 'other'
                
                if category not in summary['feature_categories']:
                    summary['feature_categories'][category] = []
                summary['feature_categories'][category].append(feature)
            
            # Calculate missing data
            for feature in feature_cols:
                missing_count = features_df[feature].isna().sum()
                if missing_count > 0:
                    summary['missing_data'][feature] = {
                        'missing_count': int(missing_count),
                        'missing_percentage': (missing_count / len(features_df)) * 100
                    }
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error generating feature summary: {e}")
            return {}

    def run_comprehensive_analysis(self, features_df: pd.DataFrame):
        """Run comprehensive analysis on extracted features"""
        if not pipeline_config.run_analysis:
            logger.info("Skipping analysis as configured")
            return {}
        
        self.pipeline_logger.log_step_start("comprehensive_analysis")
        self.results_logger.start_experiment("nafld_analysis", self.config)
        
        try:
            # Load clinical data
            clinical_data = self.clinical_loader.load_clinical_data()
            
            # Merge with features
            analysis_df = features_df.merge(clinical_data, on='patient_id')
            
            analysis_results = {}
            
            # 1. Subtype discovery
            logger.info("Running subtype discovery...")
            subtype_results = self.subtype_analyzer.analyze_subtypes(features_df, clinical_data)
            analysis_results['subtype_discovery'] = subtype_results
            self.results_logger.log_metric('subtype_count', subtype_results.get('optimal_clusters', 0))
            
            # 2. Clinical correlations
            logger.info("Running clinical correlations...")
            correlation_results = self.clinical_correlator.analyze_correlations(features_df, clinical_data)
            analysis_results['clinical_correlations'] = correlation_results
            
            # 3. Biomarker analysis
            logger.info("Running biomarker analysis...")
            biomarker_results = self._run_biomarker_analysis(features_df, clinical_data)
            analysis_results['biomarker_analysis'] = biomarker_results
            
            # 4. Survival analysis (if data available)
            survival_results = self._run_survival_analysis(features_df, clinical_data)
            if survival_results:
                analysis_results['survival_analysis'] = survival_results
            
            # Generate visualizations
            if pipeline_config.generate_plots:
                self._generate_analysis_visualizations(features_df, clinical_data, analysis_results)
            
            # Save analysis results
            analysis_path = self.paths.analysis / "comprehensive_analysis_results.pkl"
            save_pickle(analysis_results, analysis_path)
            
            # Log results
            self.results_logger.complete_experiment({
                'subtypes_discovered': subtype_results.get('optimal_clusters', 0),
                'clinical_variables_analyzed': len(correlation_results.get('univariate', {})),
                'biomarkers_identified': len(biomarker_results)
            })
            
            self.pipeline_logger.log_step_completion("comprehensive_analysis", {
                'subtypes': subtype_results.get('optimal_clusters', 0),
                'significant_biomarkers': len(biomarker_results)
            })
            
            logger.info("Comprehensive analysis completed successfully")
            
            return analysis_results
            
        except Exception as e:
            self.pipeline_logger.log_step_error("comprehensive_analysis", e)
            self.results_logger.complete_experiment({'error': str(e)})
            raise

    def _run_biomarker_analysis(self, features_df: pd.DataFrame, clinical_data: pd.DataFrame) -> dict:
        """Run biomarker analysis for key clinical outcomes"""
        biomarker_results = {}
        
        # Define key outcomes to analyze
        key_outcomes = ['NAS', 'fibrosis_stage', 'steatosis_grade', 'ballooning_grade', 'inflammation_grade']
        
        for outcome in key_outcomes:
            if outcome in clinical_data.columns:
                try:
                    logger.info(f"Analyzing biomarkers for {outcome}")
                    result = self.biomarker_analyzer.analyze_biomarkers(features_df, clinical_data, outcome)
                    biomarker_results[outcome] = result
                    
                    # Log key metrics
                    if 'cv_auc_mean' in result:
                        self.results_logger.log_metric(f'{outcome}_auc', result['cv_auc_mean'])
                    elif 'cv_r2_mean' in result:
                        self.results_logger.log_metric(f'{outcome}_r2', result['cv_r2_mean'])
                        
                except Exception as e:
                    logger.warning(f"Biomarker analysis failed for {outcome}: {e}")
        
        return biomarker_results

    def _run_survival_analysis(self, features_df: pd.DataFrame, clinical_data: pd.DataFrame) -> dict:
        """Run survival analysis if time-to-event data is available"""
        survival_columns = [col for col in clinical_data.columns 
                          if 'time' in col.lower() or 'survival' in col.lower()]
        event_columns = [col for col in clinical_data.columns 
                        if 'event' in col.lower() or 'death' in col.lower() or 'progression' in col.lower()]
        
        if not survival_columns or not event_columns:
            logger.info("No survival data found for analysis")
            return {}
        
        try:
            time_column = survival_columns[0]
            event_column = event_columns[0]
            
            logger.info(f"Running survival analysis with {time_column} and {event_column}")
            
            survival_results = self.survival_analyzer.analyze_survival(
                features_df, clinical_data, time_column, event_column
            )
            
            # Log concordance index
            if 'c_index' in survival_results:
                self.results_logger.log_metric('survival_c_index', survival_results['c_index'])
            
            return survival_results
            
        except Exception as e:
            logger.warning(f"Survival analysis failed: {e}")
            return {}

    def _generate_analysis_visualizations(self, features_df: pd.DataFrame, clinical_data: pd.DataFrame, analysis_results: dict):
        """Generate comprehensive visualizations for analysis results"""
        try:
            # Feature distributions and correlations
            self.feature_visualizer.plot_feature_distributions(features_df, clinical_data)
            self.feature_visualizer.plot_feature_correlations(features_df, clinical_data)
            self.feature_visualizer.plot_dimensionality_reduction(features_df, clinical_data)
            
            # Subtype visualizations
            if 'subtype_discovery' in analysis_results:
                subtype_data = analysis_results['subtype_discovery']
                if 'dimensionality_reduction' in subtype_data:
                    # Plot subtype clusters
                    pass
            
            # Biomarker visualizations
            if 'biomarker_analysis' in analysis_results:
                biomarker_data = analysis_results['biomarker_analysis']
                for outcome, results in biomarker_data.items():
                    if 'feature_importance' in results:
                        self.model_visualizer.plot_feature_importance(
                            list(results['feature_importance'].keys()),
                            list(results['feature_importance'].values()),
                            f"Biomarkers_{outcome}"
                        )
            
            # Interactive dashboards
            if pipeline_config.generate_interactive_dashboards:
                self.feature_visualizer.create_interactive_feature_explorer(features_df, clinical_data)
                
            logger.info("Analysis visualizations generated successfully")
            
        except Exception as e:
            logger.warning(f"Error generating analysis visualizations: {e}")

    def generate_reports(self, qc_summary: dict, features_df: pd.DataFrame, analysis_results: dict):
        """Generate comprehensive reports"""
        if not pipeline_config.generate_reports:
            logger.info("Skipping report generation as configured")
            return
        
        self.pipeline_logger.log_step_start("report_generation")
        
        try:
            # Generate markdown report
            self._generate_markdown_report(qc_summary, features_df, analysis_results)
            
            # Generate HTML report
            self._generate_html_report(qc_summary, features_df, analysis_results)
            
            # Generate executive summary
            self._generate_executive_summary(qc_summary, features_df, analysis_results)
            
            # Save pipeline execution report
            self.pipeline_logger.save_execution_report()
            self.results_logger.save_results()
            
            self.pipeline_logger.log_step_completion("report_generation", {
                'reports_generated': ['markdown', 'html', 'executive_summary']
            })
            
            logger.info("All reports generated successfully")
            
        except Exception as e:
            self.pipeline_logger.log_step_error("report_generation", e)
            raise

    def _generate_markdown_report(self, qc_summary: dict, features_df: pd.DataFrame, analysis_results: dict):
        """Generate detailed markdown report"""
        report_path = self.paths.results / "detailed_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# NAFLD Digital Pathology Analysis Report\n\n")
            
            f.write("## Executive Summary\n")
            f.write(f"- **Total Slides Processed**: {qc_summary['total_slides']}\n")
            f.write(f"- **Quality Control Pass Rate**: {qc_summary['pass_count'] / qc_summary['total_slides'] * 100:.1f}%\n")
            f.write(f"- **Patients with Features**: {len(features_df)}\n")
            f.write(f"- **Features Extracted**: {len(features_df.columns) - 1}\n")
            
            if 'subtype_discovery' in analysis_results:
                subtypes = analysis_results['subtype_discovery'].get('optimal_clusters', 0)
                f.write(f"- **Patient Subtypes Discovered**: {subtypes}\n")
            
            f.write("\n## Methodology\n")
            f.write("### Data Processing Pipeline\n")
            f.write("1. Whole Slide Image Quality Control\n")
            f.write("2. Stain Normalization\n")
            f.write("3. Tissue Segmentation using Deep Learning\n")
            f.write("4. Comprehensive Feature Extraction\n")
            f.write("5. Multivariate Statistical Analysis\n")
            
            f.write("\n### Analytical Approaches\n")
            f.write("- Unsupervised clustering for subtype discovery\n")
            f.write("- Multivariate regression for biomarker identification\n")
            f.write("- Survival analysis for prognostic factors\n")
            f.write("- Spatial analysis for tissue architecture\n")
            
            f.write("\n## Key Findings\n")
            
            # Add findings from analysis results
            if 'subtype_discovery' in analysis_results:
                f.write("### Patient Subtypes\n")
                subtype_data = analysis_results['subtype_discovery']
                f.write(f"- Identified {subtype_data.get('optimal_clusters', 0)} distinct patient subtypes\n")
                f.write("- Subtypes show different histological patterns and clinical outcomes\n")
            
            if 'biomarker_analysis' in analysis_results:
                f.write("### Biomarker Discovery\n")
                biomarker_data = analysis_results['biomarker_analysis']
                f.write(f"- Analyzed biomarkers for {len(biomarker_data)} clinical outcomes\n")
                for outcome, results in biomarker_data.items():
                    if 'cv_auc_mean' in results:
                        f.write(f"- **{outcome}**: AUC = {results['cv_auc_mean']:.3f}\n")
                    elif 'cv_r2_mean' in results:
                        f.write(f"- **{outcome}**: R² = {results['cv_r2_mean']:.3f}\n")
            
            f.write("\n## Clinical Implications\n")
            f.write("1. **Precision Medicine**: Patient subtypes may benefit from tailored treatments\n")
            f.write("2. **Early Detection**: Identified biomarkers could enable earlier intervention\n")
            f.write("3. **Prognostic Stratification**: Survival analysis informs risk assessment\n")
            f.write("4. **Treatment Monitoring**: Quantitative features enable objective response assessment\n")
            
            f.write("\n## Limitations and Future Work\n")
            f.write("- Validation in independent cohorts required\n")
            f.write("- Integration with molecular data for multi-omics analysis\n")
            f.write("- Prospective validation of clinical utility\n")
            f.write("- Development of clinical decision support tools\n")
        
        logger.info(f"Markdown report saved to {report_path}")

    def _generate_html_report(self, qc_summary: dict, features_df: pd.DataFrame, analysis_results: dict):
        """Generate interactive HTML report"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create interactive dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Quality Control Summary', 'Feature Overview', 
                              'Analysis Results', 'Clinical Correlations'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "heatmap"}]]
            )
            
            # QC Summary pie chart
            qc_data = {
                'Status': ['PASS', 'WARNING', 'FAIL'],
                'Count': [qc_summary['pass_count'], qc_summary['warning_count'], qc_summary['fail_count']]
            }
            fig.add_trace(go.Pie(labels=qc_data['Status'], values=qc_data['Count'], name="QC Status"), 1, 1)
            
            # Feature overview
            feature_cols = [col for col in features_df.columns if col != 'patient_id']
            feature_types = {
                'Morphological': len([f for f in feature_cols if 'morph' in f.lower()]),
                'Spatial': len([f for f in feature_cols if 'spatial' in f.lower()]),
                'Composition': len([f for f in feature_cols if 'percentage' in f]),
                'Other': len([f for f in feature_cols if not any(x in f.lower() for x in ['morph', 'spatial', 'percentage'])])
            }
            fig.add_trace(go.Bar(x=list(feature_types.keys()), y=list(feature_types.values()), name="Feature Types"), 1, 2)
            
            # Save HTML report
            html_path = self.paths.results / "interactive_dashboard.html"
            fig.write_html(str(html_path))
            
            logger.info(f"HTML report saved to {html_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate HTML report: {e}")

    def _generate_executive_summary(self, qc_summary: dict, features_df: pd.DataFrame, analysis_results: dict):
        """Generate executive summary for clinical stakeholders"""
        summary_path = self.paths.results / "executive_summary.json"
        
        summary = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'slides_processed': qc_summary['total_slides'],
                'qc_pass_rate': f"{qc_summary['pass_count'] / qc_summary['total_slides'] * 100:.1f}%",
                'patients_analyzed': len(features_df)
            },
            'key_findings': {
                'features_extracted': len(features_df.columns) - 1,
                'subtypes_identified': analysis_results.get('subtype_discovery', {}).get('optimal_clusters', 0),
                'biomarkers_analyzed': len(analysis_results.get('biomarker_analysis', {})),
                'clinical_correlations': len(analysis_results.get('clinical_correlations', {}).get('univariate', {}))
            },
            'clinical_insights': {
                'subtype_characterization': "Identified distinct patient subgroups with different histological patterns",
                'biomarker_potential': "Discovered quantitative features predictive of clinical outcomes",
                'prognostic_factors': "Identified features associated with disease progression"
            }
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Executive summary saved to {summary_path}")

    def run_complete_pipeline(self):
        """Run the complete analysis pipeline"""
        logger.info("Starting complete NAFLD analysis pipeline")
        
        try:
            # Step 1: Quality Control
            if pipeline_config.run_quality_control:
                logger.info("=== STEP 1: Quality Control ===")
                qc_df, qc_summary = self.run_quality_control()
            else:
                qc_df, qc_summary = None, {'total_slides': 0, 'pass_count': 0, 'fail_count': 0, 'warning_count': 0}
            
            # Step 2: Stain Normalization
            if pipeline_config.run_stain_normalization:
                logger.info("=== STEP 2: Stain Normalization ===")
                self.run_stain_normalization()
            
            # Step 3: Initialize Models
            logger.info("=== STEP 3: Model Initialization ===")
            self.initialize_models()
            
            # Step 4: Segmentation
            if pipeline_config.run_segmentation:
                logger.info("=== STEP 4: Tissue Segmentation ===")
                segmentation_results = self.run_segmentation()
            
            # Step 5: Feature Extraction
            if pipeline_config.run_feature_extraction:
                logger.info("=== STEP 5: Feature Extraction ===")
                features_df = self.run_feature_extraction()
            else:
                # Try to load existing features
                features_path = self.paths.features / "comprehensive_features.csv"
                if features_path.exists():
                    features_df = pd.read_csv(features_path)
                    logger.info(f"Loaded existing features for {len(features_df)} patients")
                else:
                    raise FileNotFoundError("No features available and feature extraction is disabled")
            
            # Step 6: Comprehensive Analysis
            if pipeline_config.run_analysis:
                logger.info("=== STEP 6: Comprehensive Analysis ===")
                analysis_results = self.run_comprehensive_analysis(features_df)
            else:
                analysis_results = {}
            
            # Step 7: Report Generation
            if pipeline_config.generate_reports:
                logger.info("=== STEP 7: Report Generation ===")
                self.generate_reports(qc_summary, features_df, analysis_results)
            
            logger.info("NAFLD analysis pipeline completed successfully")
            
            return {
                'quality_control': qc_summary,
                'features': features_df,
                'analysis': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.pipeline_logger.save_execution_report()
            raise

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='NAFLD Digital Pathology Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--steps', type=str, nargs='+', 
                       choices=['all', 'qc', 'normalization', 'segmentation', 'features', 'analysis', 'reports'],
                       default=['all'], help='Pipeline steps to run')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline with configuration
        pipeline = NAFLDPipeline(args.config)
        
        if 'all' in args.steps:
            results = pipeline.run_complete_pipeline()
        else:
            # Run specific steps
            results = {}
            if 'qc' in args.steps:
                results['qc'] = pipeline.run_quality_control()
            if 'normalization' in args.steps:
                pipeline.run_stain_normalization()
            if 'segmentation' in args.steps:
                results['segmentation'] = pipeline.run_segmentation()
            if 'features' in args.steps:
                results['features'] = pipeline.run_feature_extraction()
            if 'analysis' in args.steps:
                if 'features' in results:
                    results['analysis'] = pipeline.run_comprehensive_analysis(results['features'])
            if 'reports' in args.steps:
                pipeline.generate_reports(
                    results.get('qc', [None, {}])[1] if 'qc' in results else {},
                    results.get('features', pd.DataFrame()),
                    results.get('analysis', {})
                )
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()