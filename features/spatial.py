"""
Spatial analysis features for tissue architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import spatial, ndimage
from skimage import measure, morphology
import cv2
from ..config.parameters import feature_params
import logging

logger = logging.getLogger(__name__)

class SpatialAnalyzer:
    """Analyze spatial relationships between tissue components"""
    
    def __init__(self):
        self.feature_config = feature_params
    
    def analyze_spatial_distribution(self, mask: np.ndarray) -> Dict:
        """Analyze spatial distribution of all tissue components"""
        spatial_features = {}
        
        for class_id, class_name in enumerate(['steatosis', 'ballooning', 'inflammation', 'fibrosis']):
            class_mask = (mask == class_id + 1)  # +1 because 0 is background
            
            if np.any(class_mask):
                class_features = self._analyze_class_spatial_distribution(class_mask, class_name)
                spatial_features.update(class_features)
            else:
                # Add zero features for missing classes
                spatial_features.update(self._get_zero_spatial_features(class_name))
        
        # Cross-component spatial relationships
        cross_features = self._analyze_cross_component_spatial(mask)
        spatial_features.update(cross_features)
        
        return spatial_features
    
    def _analyze_class_spatial_distribution(self, class_mask: np.ndarray, class_name: str) -> Dict:
        """Analyze spatial distribution for single class"""
        features = {}
        
        # Get coordinates of class pixels
        coords = np.column_stack(np.where(class_mask))
        
        if len(coords) < 2:
            return self._get_zero_spatial_features(class_name)
        
        # Centroid
        centroid = np.mean(coords, axis=0)
        features[f'{class_name}_centroid_x'] = centroid[1]
        features[f'{class_name}_centroid_y'] = centroid[0]
        
        # Spatial dispersion (mean distance to centroid)
        distances_to_centroid = spatial.distance.cdist(coords, [centroid])
        features[f'{class_name}_mean_dispersion'] = np.mean(distances_to_centroid)
        features[f'{class_name}_dispersion_std'] = np.std(distances_to_centroid)
        
        # Nearest neighbor distances
        if len(coords) > 1:
            distance_matrix = spatial.distance_matrix(coords, coords)
            np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
            nearest_neighbor_dists = np.min(distance_matrix, axis=1)
            
            features[f'{class_name}_mean_nearest_neighbor'] = np.mean(nearest_neighbor_dists)
            features[f'{class_name}_nearest_neighbor_std'] = np.std(nearest_neighbor_dists)
        else:
            features[f'{class_name}_mean_nearest_neighbor'] = 0
            features[f'{class_name}_nearest_neighbor_std'] = 0
        
        # Ripley's K function (simplified)
        ripley_k = self._calculate_ripley_k(coords, class_mask.shape)
        features[f'{class_name}_ripley_k'] = ripley_k
        
        # Spatial autocorrelation (Moran's I simplified)
        morans_i = self._calculate_morans_i(class_mask)
        features[f'{class_name}_morans_i'] = morans_i
        
        return features
    
    def _analyze_cross_component_spatial(self, mask: np.ndarray) -> Dict:
        """Analyze spatial relationships between different components"""
        features = {}
        
        component_masks = {}
        for class_id, class_name in enumerate(['steatosis', 'ballooning', 'inflammation', 'fibrosis']):
            component_masks[class_name] = (mask == class_id + 1)
        
        # Analyze spatial proximity between components
        components = list(component_masks.keys())
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                if np.any(component_masks[comp1]) and np.any(component_masks[comp2]):
                    proximity = self._calculate_component_proximity(
                        component_masks[comp1], component_masks[comp2]
                    )
                    features[f'{comp1}_{comp2}_proximity'] = proximity
                else:
                    features[f'{comp1}_{comp2}_proximity'] = 0
        
        return features
    
    def _calculate_ripley_k(self, coords: np.ndarray, image_shape: Tuple[int, int], r_max: float = 50) -> float:
        """Calculate simplified Ripley's K function"""
        if len(coords) < 2:
            return 0.0
        
        area = image_shape[0] * image_shape[1]
        n_points = len(coords)
        
        # Count points within radius r for a sample of points
        r = min(r_max, min(image_shape) / 4)
        sample_size = min(100, n_points)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        
        counts = []
        for idx in sample_indices:
            point = coords[idx]
            distances = spatial.distance.cdist([point], coords)[0]
            within_r = np.sum(distances <= r) - 1  # Exclude self
            counts.append(within_r)
        
        mean_count = np.mean(counts)
        density = n_points / area
        k_value = mean_count / density
        
        return float(k_value)
    
    def _calculate_morans_i(self, binary_mask: np.ndarray) -> float:
        """Calculate simplified Moran's I for spatial autocorrelation"""
        try:
            # Convert to float and normalize
            mask_float = binary_mask.astype(float)
            
            # Calculate mean
            mean_val = np.mean(mask_float)
            
            if mean_val == 0 or mean_val == 1:
                return 0.0
            
            # Calculate numerator and denominator
            n = mask_float.size
            numerator = 0
            denominator = 0
            
            for i in range(1, mask_float.shape[0]-1):
                for j in range(1, mask_float.shape[1]-1):
                    # Simple 4-neighborhood
                    neighbors = [
                        mask_float[i-1, j], mask_float[i+1, j],
                        mask_float[i, j-1], mask_float[i, j+1]
                    ]
                    
                    for neighbor in neighbors:
                        numerator += (mask_float[i, j] - mean_val) * (neighbor - mean_val)
                    
                    denominator += (mask_float[i, j] - mean_val) ** 2
            
            if denominator == 0:
                return 0.0
            
            morans_i = (n / (4 * (n - 1))) * (numerator / denominator)
            return float(morans_i)
        
        except:
            return 0.0
    
    def _calculate_component_proximity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate proximity between two tissue components"""
        # Find centroids
        coords1 = np.column_stack(np.where(mask1))
        coords2 = np.column_stack(np.where(mask2))
        
        if len(coords1) == 0 or len(coords2) == 0:
            return 0.0
        
        centroid1 = np.mean(coords1, axis=0)
        centroid2 = np.mean(coords2, axis=0)
        
        # Distance between centroids
        distance = np.linalg.norm(centroid1 - centroid2)
        
        # Normalize by image diagonal
        image_diagonal = np.linalg.norm(mask1.shape)
        normalized_distance = distance / image_diagonal
        
        return float(1 - normalized_distance)  # Convert to proximity (1 = close, 0 = far)
    
    def _get_zero_spatial_features(self, class_name: str) -> Dict:
        """Return zero values for spatial features when class is absent"""
        return {
            f'{class_name}_centroid_x': 0,
            f'{class_name}_centroid_y': 0,
            f'{class_name}_mean_dispersion': 0,
            f'{class_name}_dispersion_std': 0,
            f'{class_name}_mean_nearest_neighbor': 0,
            f'{class_name}_nearest_neighbor_std': 0,
            f'{class_name}_ripley_k': 0,
            f'{class_name}_morans_i': 0
        }

class ZonalAnalyzer:
    """Analyze zonal distribution patterns in liver tissue"""
    
    def __init__(self):
        self.zones = ['periportal', 'midzonal', 'centrilobular']
    
    def analyze_zonal_distribution(self, mask: np.ndarray, portal_veins: np.ndarray, 
                                 central_veins: np.ndarray) -> Dict:
        """Analyze distribution across liver zones"""
        zonal_features = {}
        
        if not np.any(portal_veins) or not np.any(central_veins):
            logger.warning("Portal or central veins not provided for zonal analysis")
            return zonal_features
        
        # Create distance maps
        portal_distance = ndimage.distance_transform_edt(~portal_veins)
        central_distance = ndimage.distance_transform_edt(~central_veins)
        
        # Define zones based on distances
        max_portal_dist = np.max(portal_distance)
        max_central_dist = np.max(central_distance)
        
        # Simple zone definition (can be improved with anatomical knowledge)
        zone_masks = self._define_zonal_masks(portal_distance, central_distance, 
                                            max_portal_dist, max_central_dist)
        
        # Analyze each tissue component in each zone
        for class_id, class_name in enumerate(['steatosis', 'ballooning', 'inflammation', 'fibrosis']):
            class_mask = (mask == class_id + 1)
            
            for zone_name, zone_mask in zone_masks.items():
                zone_class_mask = class_mask & zone_mask
                zone_area = np.sum(zone_mask)
                
                if zone_area > 0:
                    density = np.sum(zone_class_mask) / zone_area
                else:
                    density = 0
                
                zonal_features[f'{class_name}_{zone_name}_density'] = density
        
        return zonal_features
    
    def _define_zonal_masks(self, portal_distance: np.ndarray, central_distance: np.ndarray,
                          max_portal: float, max_central: float) -> Dict:
        """Define zonal masks based on distance to portal and central veins"""
        zone_masks = {}
        
        # Simple threshold-based zoning
        # This is a simplification - real liver zoning is more complex
        periportal_threshold = max_portal * 0.33
        centrilobular_threshold = max_central * 0.33
        
        zone_masks['periportal'] = portal_distance <= periportal_threshold
        zone_masks['centrilobular'] = central_distance <= centrilobular_threshold
        zone_masks['midzonal'] = ~(zone_masks['periportal'] | zone_masks['centrilobular'])
        
        return zone_masks