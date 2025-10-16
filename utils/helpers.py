"""
Utility functions and helpers for the pipeline
"""

import logging
import random
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

def create_logger(name: str, log_file: Optional[Path] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_random_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_pickle(obj: Any, filepath: Path):
    """Save object as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: Path) -> Any:
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def calculate_class_weights(labels: np.ndarray) -> np.ndarray:
    """Calculate class weights for imbalanced datasets"""
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts)
    return weights

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        memory_info = []
        
        for i in range(gpu_count):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            memory_info.append({
                'gpu': i,
                'allocated_gb': allocated,
                'reserved_gb': reserved
            })
        
        return memory_info
    else:
        return None

def create_timestamp() -> str:
    """Create timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    if denominator == 0:
        return default
    return numerator / denominator

def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize array using specified method"""
    if method == 'minmax':
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    elif method == 'zscore':
        if arr.std() == 0:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / arr.std()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        return scaler.fit_transform(arr.reshape(-1, 1)).flatten()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_progress_tracker(total: int, description: str = "Processing"):
    """Create a simple progress tracker"""
    from tqdm import tqdm
    return tqdm(total=total, desc=description, unit="items")