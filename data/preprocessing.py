import numpy as np
import cv2
from typing import Dict, List, Tuple
from skimage import exposure, filters, morphology, segmentation
from typing import Tuple, Optional
import torch
from torchvision import transforms
from ..config.parameters import data_params
import logging

logger = logging.getLogger(__name__)

class StainNormalizer:
    """Stain normalization using various methods"""
    
    def __init__(self, method: str = "macenko"):
        self.method = method
        self.reference_matrix = None
        
    def fit(self, reference_image: np.ndarray):
        """Fit normalizer to reference image"""
        if self.method == "macenko":
            self.reference_matrix = self._macenko_fit(reference_image)
        elif self.method == "reinhard":
            self.reference_matrix = self._reinhard_fit(reference_image)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply stain normalization"""
        if self.reference_matrix is None:
            logger.warning("Stain normalizer not fitted. Returning original image.")
            return image
            
        if self.method == "macenko":
            return self._macenko_transform(image)
        elif self.method == "reinhard":
            return self._reinhard_transform(image)
        else:
            return image
    
    def _macenko_fit(self, image: np.ndarray):
        """Macenko method fitting"""
        # Implementation of Macenko stain normalization
        # This is a simplified version - consider using external libraries like stain-tools
        OD = -np.log((image.astype(np.float64) + 1) / 255)
        OD = OD.reshape(-1, 3)
        
        # Remove transparent pixels
        OD = OD[OD.max(axis=1) > 0.15]
        
        # SVD on the OD tuples
        _, _, V = np.linalg.svd(OD, full_matrices=False)
        
        # The stain vectors are the first two principal components
        stain_vectors = V[:2].T
        return stain_vectors
    
    def _macenko_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply Macenko normalization"""
        # Simplified implementation
        # In practice, use a robust implementation from stain-tools
        return image
    
    def _reinhard_fit(self, image: np.ndarray):
        """Reinhard method fitting"""
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        mean = np.mean(lab_image, axis=(0, 1))
        std = np.std(lab_image, axis=(0, 1))
        return {'mean': mean, 'std': std}
    
    def _reinhard_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply Reinhard normalization"""
        if self.reference_matrix is None:
            return image
            
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_image = lab_image.astype(np.float64)
        
        # Normalize
        for i in range(3):
            lab_image[:, :, i] = (lab_image[:, :, i] - np.mean(lab_image[:, :, i])) / np.std(lab_image[:, :, i])
            lab_image[:, :, i] = lab_image[:, :, i] * self.reference_matrix['std'][i] + self.reference_matrix['mean'][i]
        
        lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

class TissueDetector:
    """Detect tissue regions in WSI"""
    
    def __init__(self, min_tissue_area: float = 0.7):
        self.min_tissue_area = min_tissue_area
    
    def detect_tissue_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect tissue regions using multiple methods"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Otsu's thresholding
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive thresholding
        adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(otsu_mask, adaptive_mask)
        
        # Remove small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = morphology.opening(combined_mask, kernel)
        
        return cleaned_mask
    
    def calculate_tissue_percentage(self, mask: np.ndarray) -> float:
        """Calculate percentage of tissue in image"""
        tissue_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        return tissue_pixels / total_pixels

class PatchExtractor:
    """Extract patches from whole slide images"""
    
    def __init__(self, patch_size: Tuple[int, int] = (512, 512), overlap: float = 0.1):
        self.patch_size = patch_size
        self.overlap = overlap
        self.tissue_detector = TissueDetector()
    
    def extract_patches(self, slide, level: int = 0) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Extract patches from slide at specified level"""
        patches = []
        slide_dimensions = slide.level_dimensions[level]
        
        stride = int(self.patch_size[0] * (1 - self.overlap))
        
        for y in range(0, slide_dimensions[1] - self.patch_size[1], stride):
            for x in range(0, slide_dimensions[0] - self.patch_size[0], stride):
                # Read patch
                patch = np.array(slide.read_region((x, y), level, self.patch_size))[:, :, :3]
                
                # Check if patch contains sufficient tissue
                tissue_mask = self.tissue_detector.detect_tissue_regions(patch)
                tissue_percentage = self.tissue_detector.calculate_tissue_percentage(tissue_mask)
                
                if tissue_percentage >= data_params.min_tissue_area:
                    patches.append((patch, (x, y)))
        
        return patches

class ImageAugmentor:
    """Image augmentation for training"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
    
    def augment_batch(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Augment batch of images and optionally masks"""
        if masks is not None:
            # Apply same spatial transformations to images and masks
            stacked = torch.cat([images, masks], dim=1)
            augmented = self.transform(stacked)
            aug_images = augmented[:, :3]
            aug_masks = augmented[:, 3:]
            return aug_images, aug_masks
        else:
            return self.transform(images)