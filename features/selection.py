"""
Feature selection and biomarker identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import (SelectKBest, f_classif, RFE, 
                                     mutual_info_classif, SelectFromModel)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy import stats
from ..config.parameters import analysis_params
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Comprehensive feature selection for NAFLD biomarkers"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_features = {}
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = "random_forest", n_features: int = 20) -> Dict:
        """Select features using multiple methods"""
        # Handle missing values
        X_clean = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        selection_results = {}
        
        if method == "univariate":
            selection_results = self._univariate_selection(X_scaled, y, n_features)
        elif method == "recursive_elimination":
            selection_results = self._recursive_elimination(X_scaled, y, n_features)
        elif method == "random_forest":
            selection_results = self._random_forest_selection(X_scaled, y, n_features)
        elif method == "lasso":
            selection_results = self._lasso_selection(X_scaled, y)
        elif method == "mutual_info":
            selection_results = self._mutual_info_selection(X_scaled, y, n_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        self.selected_features[method] = selection_results
        return selection_results
    
    def _univariate_selection(self, X: np.ndarray, y: pd.Series, n_features: int) -> Dict:
        """Univariate feature selection using ANOVA F-test"""
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get feature scores and p-values
        scores = selector.scores_
        p_values = selector.pvalues_
        
        # Create ranking
        feature_ranking = np.argsort(scores)[::-1]  # Descending order
        selected_indices = feature_ranking[:n_features]
        
        return {
            'selected_features': selected_indices.tolist(),
            'scores': scores.tolist(),
            'p_values': p_values.tolist(),
            'feature_names': [f'feature_{i}' for i in selected_indices],
            'method': 'univariate_anova'
        }
    
    def _recursive_elimination(self, X: np.ndarray, y: pd.Series, n_features: int) -> Dict:
        """Recursive feature elimination"""
        estimator = RandomForestClassifier(n_estimators=100, random_state=analysis_params.random_state)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)
        
        return {
            'selected_features': selector.support_.tolist(),
            'feature_ranking': selector.ranking_.tolist(),
            'feature_scores': selector.estimator_.feature_importances_.tolist(),
            'method': 'recursive_elimination'
        }
    
    def _random_forest_selection(self, X: np.ndarray, y: pd.Series, n_features: int) -> Dict:
        """Feature selection using Random Forest importance"""
        rf = RandomForestClassifier(n_estimators=100, random_state=analysis_params.random_state)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        selected_indices = indices[:n_features]
        
        return {
            'selected_features': selected_indices.tolist(),
            'feature_importances': importances.tolist(),
            'feature_names': [f'feature_{i}' for i in selected_indices],
            'method': 'random_forest'
        }
    
    def _lasso_selection(self, X: np.ndarray, y: pd.Series) -> Dict:
        """Feature selection using Lasso regularization"""
        lasso = LassoCV(cv=5, random_state=analysis_params.random_state)
        lasso.fit(X, y)
        
        # Features with non-zero coefficients are selected
        selected_mask = lasso.coef_ != 0
        selected_indices = np.where(selected_mask)[0]
        
        return {
            'selected_features': selected_indices.tolist(),
            'coefficients': lasso.coef_.tolist(),
            'alpha': lasso.alpha_,
            'method': 'lasso'
        }
    
    def _mutual_info_selection(self, X: np.ndarray, y: pd.Series, n_features: int) -> Dict:
        """Feature selection using mutual information"""
        mi_scores = mutual_info_classif(X, y, random_state=analysis_params.random_state)
        indices = np.argsort(mi_scores)[::-1]
        
        selected_indices = indices[:n_features]
        
        return {
            'selected_features': selected_indices.tolist(),
            'mi_scores': mi_scores.tolist(),
            'feature_names': [f'feature_{i}' for i in selected_indices],
            'method': 'mutual_information'
        }
    
    def consensus_selection(self, X: pd.DataFrame, y: pd.Series, 
                          methods: List[str] = None, n_features: int = 15) -> Dict:
        """Combine multiple selection methods for consensus"""
        if methods is None:
            methods = ['univariate', 'random_forest', 'mutual_info']
        
        all_selected = []
        
        for method in methods:
            result = self.select_features(X, y, method, n_features)
            selected = result['selected_features']
            all_selected.extend(selected)
        
        # Count frequency of selection
        feature_counts = {}
        for feature_idx in all_selected:
            feature_counts[feature_idx] = feature_counts.get(feature_idx, 0) + 1
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        consensus_features = [feat for feat, count in sorted_features[:n_features]]
        
        return {
            'consensus_features': consensus_features,
            'selection_frequency': feature_counts,
            'methods_used': methods,
            'total_methods': len(methods)
        }

class BiomarkerSelector:
    """Identify potential biomarkers for NAFLD progression"""
    
    def __init__(self):
        self.feature_selector = FeatureSelector()
    
    def identify_biomarkers(self, features_df: pd.DataFrame, clinical_target: str,
                          clinical_data: pd.DataFrame) -> Dict:
        """Identify biomarkers for specific clinical target"""
        # Merge features with clinical data
        merged_data = features_df.merge(clinical_data, on='patient_id', how='inner')
        
        if clinical_target not in merged_data.columns:
            raise ValueError(f"Clinical target {clinical_target} not found in data")
        
        # Prepare data
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        X = merged_data[feature_cols]
        y = merged_data[clinical_target]
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        
        # Consensus feature selection
        consensus = self.feature_selector.consensus_selection(X, y)
        
        # Calculate biomarker performance
        biomarker_performance = self._evaluate_biomarkers(X, y, consensus['consensus_features'])
        
        return {
            'clinical_target': clinical_target,
            'selected_biomarkers': consensus['consensus_features'],
            'biomarker_names': [feature_cols[i] for i in consensus['consensus_features']],
            'selection_frequency': consensus['selection_frequency'],
            'performance': biomarker_performance,
            'total_features_considered': len(feature_cols)
        }
    
    def _evaluate_biomarkers(self, X: pd.DataFrame, y: pd.Series, 
                           biomarker_indices: List[int]) -> Dict:
        """Evaluate biomarker performance"""
        if len(biomarker_indices) == 0:
            return {}
        
        # Select biomarker features
        biomarker_data = X.iloc[:, biomarker_indices]
        
        # Calculate correlations with target
        correlations = {}
        p_values = {}
        
        for i, col_idx in enumerate(biomarker_indices):
            col_name = X.columns[col_idx]
            corr, p_val = stats.pearsonr(biomarker_data.iloc[:, i], y)
            correlations[col_name] = corr
            p_values[col_name] = p_val
        
        # Build simple predictive model
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        if len(np.unique(y)) > 2:  # Regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            scoring = 'r2'
        else:  # Classification
            model = LogisticRegression(random_state=analysis_params.random_state)
            scoring = 'accuracy'
        
        cv_scores = cross_val_score(model, biomarker_data, y, cv=5, scoring=scoring)
        
        return {
            'correlations': correlations,
            'p_values': p_values,
            'cv_mean_score': np.mean(cv_scores),
            'cv_std_score': np.std(cv_scores),
            'biomarker_count': len(biomarker_indices)
        }
    
    def generate_biomarker_report(self, biomarker_results: Dict) -> pd.DataFrame:
        """Generate comprehensive biomarker report"""
        report_data = []
        
        for i, biomarker_idx in enumerate(biomarker_results['selected_biomarkers']):
            biomarker_name = biomarker_results['biomarker_names'][i]
            
            report_entry = {
                'biomarker_name': biomarker_name,
                'feature_index': biomarker_idx,
                'selection_frequency': biomarker_results['selection_frequency'].get(biomarker_idx, 0),
                'correlation': biomarker_results['performance']['correlations'].get(biomarker_name, 0),
                'p_value': biomarker_results['performance']['p_values'].get(biomarker_name, 1),
                'clinical_target': biomarker_results['clinical_target']
            }
            report_data.append(report_entry)
        
        return pd.DataFrame(report_data).sort_values('correlation', key=abs, ascending=False)