"""
NAFLD Digital Pathology Pipeline Configuration Module
"""

from .paths import Paths
from .parameters import data_params, model_params, feature_params, analysis_params

__all__ = ['Paths', 'data_params', 'model_params', 'feature_params', 'analysis_params']