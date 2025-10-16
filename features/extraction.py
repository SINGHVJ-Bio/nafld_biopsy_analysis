import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from skimage import measure, morphology, feature, filters
import cv2
from ..config.parameters import feature_params

class MorphologicalFeatureExtractor:
    """Extract morphological features from segmentation masks"""
    
    def __init__(self):
        self.feature_config = feature_params
    
    def extract_steatosis_features(self, mask: np.ndarray, original_image: np.ndarray) -> Dict:
        """Extract features for steatosis droplets"""
        steatosis_mask = (mask == 1)  # Assuming class 1 is steatosis
        labeled_mask = measure.label(steatosis_mask)
        regions = measure.regionprops(labeled_mask, intensity_image=original_image)
        
        features = {}
        
        # Basic morphology
        areas = [r.area for r in regions]
        perimeters = [r.perimeter for r in regions]
        eccentricities = [r.eccentricity for r in regions]
        solidities = [r.solidity for r in regions]
        
        # Size features
        features['steatosis_total_area'] = np.sum(areas)
        features['steatosis_droplet_count'] = len(regions)
        features['steatosis_mean_size'] = np.mean(areas) if areas else 0
        features['steatosis_size_std'] = np.std(areas) if areas else 0
        features['steatosis_size_range'] = np.ptp(areas) if areas else 0
        
        # Shape features
        features['steatosis_mean_circularity'] = self._calculate_circularity(areas, perimeters)
        features['steatosis_mean_eccentricity'] = np.mean(eccentricities) if eccentricities else 0
        features['steatosis_mean_solidity'] = np.mean(solidities) if solidities else 0
        
        # Spatial distribution
        features['steatosis_area_percentage'] = features['steatosis_total_area'] / mask.size
        features['steatosis_droplet_density'] = len(regions) / features['steatosis_total_area'] if features['steatosis_total_area'] > 0 else 0
        
        return features
    
    def extract_ballooning_features(self, mask: np.ndarray, original_image: np.ndarray) -> Dict:
        """Extract features for ballooned hepatocytes"""
        ballooning_mask = (mask == 2)  # Assuming class 2 is ballooning
        labeled_mask = measure.label(ballooning_mask)
        regions = measure.regionprops(labeled_mask, intensity_image=original_image)
        
        features = {}
        
        if not regions:
            return {f'ballooning_{feature}': 0 for feature in self.feature_config.ballooning_features}
        
        # Intensity features
        intensities = [r.mean_intensity for r in regions]
        
        # Morphology
        areas = [r.area for r in regions]
        perimeters = [r.perimeter for r in regions]
        eccentricities = [r.eccentricity for r in regions]
        
        features['ballooning_count'] = len(regions)
        features['ballooning_mean_area'] = np.mean(areas)
        features['ballooning_area_std'] = np.std(areas)
        features['ballooning_mean_perimeter'] = np.mean(perimeters)
        features['ballooning_mean_intensity'] = np.mean(intensities)
        features['ballooning_intensity_std'] = np.std(intensities)
        features['ballooning_mean_eccentricity'] = np.mean(eccentricities)
        
        return features
    
    def extract_inflammation_features(self, mask: np.ndarray) -> Dict:
        """Extract features for inflammatory cells"""
        inflammation_mask = (mask == 3)  # Assuming class 3 is inflammation
        labeled_mask = measure.label(inflammation_mask)
        regions = measure.regionprops(labeled_mask)
        
        features = {}
        
        if not regions:
            return {f'inflammation_{feature}': 0 for feature in self.feature_config.inflammation_features}
        
        areas = [r.area for r in regions]
        
        # Cluster analysis
        features['inflammation_cluster_count'] = len(regions)
        features['inflammation_total_area'] = np.sum(areas)
        features['inflammation_mean_cluster_size'] = np.mean(areas)
        features['inflammation_cluster_size_std'] = np.std(areas)
        features['inflammation_area_percentage'] = np.sum(areas) / mask.size
        features['inflammation_cluster_density'] = len(regions) / np.sum(areas) if np.sum(areas) > 0 else 0
        
        return features
    
    def extract_fibrosis_features(self, mask: np.ndarray) -> Dict:
        """Extract features for fibrosis"""
        fibrosis_mask = (mask == 4)  # Assuming class 4 is fibrosis
        
        features = {}
        
        # Basic area measurement
        fibrosis_area = np.sum(fibrosis_mask)
        features['fibrosis_area_percentage'] = fibrosis_area / mask.size
        
        # Collagen proportionate area (simplified)
        features['collagen_proportionate_area'] = fibrosis_area / mask.size
        
        # Pattern analysis using texture
        if fibrosis_area > 0:
            texture_features = self._analyze_fibrosis_texture(fibrosis_mask)
            features.update(texture_features)
        else:
            features['fibrosis_pattern_entropy'] = 0
            features['fibrosis_pattern_contrast'] = 0
        
        return features
    
    def _calculate_circularity(self, areas: List[float], perimeters: List[float]) -> float:
        """Calculate circularity (4π*area/perimeter²)"""
        circularities = []
        for area, perimeter in zip(areas, perimeters):
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                circularities.append(circularity)
        return np.mean(circularities) if circularities else 0
    
    def _analyze_fibrosis_texture(self, fibrosis_mask: np.ndarray) -> Dict:
        """Analyze fibrosis pattern texture"""
        # Use GLCM for texture analysis
        from skimage.feature import graycomatrix, graycoprops
        
        # Convert to uint8 for GLCM
        fibrosis_uint8 = (fibrosis_mask * 255).astype(np.uint8)
        
        try:
            glcm = graycomatrix(fibrosis_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            return {
                'fibrosis_pattern_contrast': contrast,
                'fibrosis_pattern_dissimilarity': dissimilarity,
                'fibrosis_pattern_homogeneity': homogeneity,
                'fibrosis_pattern_energy': energy,
                'fibrosis_pattern_correlation': correlation
            }
        except:
            return {f'fibrosis_pattern_{prop}': 0 for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}

class ComprehensiveFeatureExtractor:
    """Extract all features from segmented images"""
    
    def __init__(self):
        self.morph_extractor = MorphologicalFeatureExtractor()
    
    def extract_all_features(self, segmentation_mask: np.ndarray, original_image: np.ndarray, patient_id: str) -> Dict:
        """Extract comprehensive feature set"""
        features = {'patient_id': patient_id}
        
        # Extract features for each tissue component
        steatosis_features = self.morph_extractor.extract_steatosis_features(segmentation_mask, original_image)
        ballooning_features = self.morph_extractor.extract_ballooning_features(segmentation_mask, original_image)
        inflammation_features = self.morph_extractor.extract_inflammation_features(segmentation_mask)
        fibrosis_features = self.morph_extractor.extract_fibrosis_features(segmentation_mask)
        
        # Combine all features
        features.update(steatosis_features)
        features.update(ballooning_features)
        features.update(inflammation_features)
        features.update(fibrosis_features)
        
        # Overall tissue composition
        features = self._add_composition_features(features, segmentation_mask)
        
        return features
    
    def _add_composition_features(self, features: Dict, mask: np.ndarray) -> Dict:
        """Add overall tissue composition features"""
        total_pixels = mask.size
        
        for class_idx, class_name in enumerate(['background', 'steatosis', 'ballooning', 'inflammation', 'fibrosis']):
            class_pixels = np.sum(mask == class_idx)
            features[f'{class_name}_percentage'] = class_pixels / total_pixels
        
        return features
    
    def extract_features_batch(self, segmentation_masks: List[np.ndarray], original_images: List[np.ndarray], patient_ids: List[str]) -> pd.DataFrame:
        """Extract features for multiple images"""
        all_features = []
        
        for mask, image, patient_id in zip(segmentation_masks, original_images, patient_ids):
            features = self.extract_all_features(mask, image, patient_id)
            all_features.append(features)
        
        return pd.DataFrame(all_features)