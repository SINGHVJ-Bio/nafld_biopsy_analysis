"""
Clinical correlation analysis for NAFLD features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from ..config.parameters import analysis_params
import logging

logger = logging.getLogger(__name__)

class ClinicalCorrelator:
    """Analyze correlations between imaging features and clinical data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.correlation_results = {}
    
    def analyze_correlations(self, features_df: pd.DataFrame, 
                           clinical_data: pd.DataFrame) -> Dict:
        """Comprehensive correlation analysis"""
        # Merge data
        merged_data = features_df.merge(clinical_data, on='patient_id', how='inner')
        
        # Separate features and clinical variables
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        clinical_cols = [col for col in clinical_data.columns if col not in ['patient_id', 'group']]
        
        correlation_analysis = {}
        
        # Univariate correlations
        univariate_results = self._univariate_correlations(merged_data, feature_cols, clinical_cols)
        correlation_analysis['univariate'] = univariate_results
        
        # Multivariate analysis
        multivariate_results = self._multivariate_analysis(merged_data, feature_cols, clinical_cols)
        correlation_analysis['multivariate'] = multivariate_results
        
        # Group comparisons
        if 'group' in clinical_data.columns:
            group_results = self._group_comparisons(merged_data, feature_cols)
            correlation_analysis['group_comparisons'] = group_results
        
        self.correlation_results = correlation_analysis
        return correlation_analysis
    
    def _univariate_correlations(self, data: pd.DataFrame, feature_cols: List[str], 
                               clinical_cols: List[str]) -> Dict:
        """Calculate univariate correlations"""
        results = {}
        
        for clinical_var in clinical_cols:
            if clinical_var not in data.columns:
                continue
            
            clinical_series = data[clinical_var]
            
            # Skip if too many missing values
            if clinical_series.isna().sum() > len(clinical_series) * 0.5:
                continue
            
            var_results = {}
            
            for feature in feature_cols:
                if feature not in data.columns:
                    continue
                
                feature_series = data[feature]
                
                # Remove pairs with missing values
                valid_mask = ~clinical_series.isna() & ~feature_series.isna()
                clinical_valid = clinical_series[valid_mask]
                feature_valid = feature_series[valid_mask]
                
                if len(clinical_valid) < 10:  # Minimum sample size
                    continue
                
                # Determine correlation type based on variable type
                if clinical_valid.dtype in ['object', 'category']:
                    # Categorical variable
                    correlation_type = 'categorical'
                    corr_result = self._categorical_correlation(feature_valid, clinical_valid)
                else:
                    # Continuous variable
                    correlation_type = 'continuous'
                    corr_result = self._continuous_correlation(feature_valid, clinical_valid)
                
                var_results[feature] = {
                    'correlation_type': correlation_type,
                    **corr_result
                }
            
            results[clinical_var] = var_results
        
        return results
    
    def _continuous_correlation(self, x: pd.Series, y: pd.Series) -> Dict:
        """Correlation between two continuous variables"""
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(x, y)
        
        # Spearman correlation (non-parametric)
        spearman_corr, spearman_p = stats.spearmanr(x, y)
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'sample_size': len(x)
        }
    
    def _categorical_correlation(self, continuous_var: pd.Series, 
                               categorical_var: pd.Series) -> Dict:
        """Correlation between continuous and categorical variables"""
        # ANOVA for group differences
        groups = categorical_var.unique()
        group_data = [continuous_var[categorical_var == group] for group in groups]
        
        # Remove groups with too few samples
        group_data = [data for data in group_data if len(data) >= 3]
        
        if len(group_data) < 2:
            return {
                'f_statistic': 0,
                'p_value': 1,
                'group_means': {},
                'effect_size': 0
            }
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Effect size (eta squared)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(continuous_var))**2 for group in group_data)
        ss_total = sum((continuous_var - np.mean(continuous_var))**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Group means
        group_means = {group: np.mean(continuous_var[categorical_var == group]) 
                      for group in groups}
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'group_means': group_means,
            'effect_size': eta_squared,
            'sample_size': len(continuous_var)
        }
    
    def _multivariate_analysis(self, data: pd.DataFrame, feature_cols: List[str],
                             clinical_cols: List[str]) -> Dict:
        """Multivariate regression analysis"""
        results = {}
        
        for clinical_var in clinical_cols:
            if clinical_var not in data.columns:
                continue
            
            clinical_series = data[clinical_var].dropna()
            
            if len(clinical_series) < 20:  # Minimum for multivariate
                continue
            
            # Prepare data
            valid_indices = clinical_series.index
            X = data.loc[valid_indices, feature_cols].fillna(0)
            y = clinical_series
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine regression type
            if y.dtype in ['object', 'category'] or len(y.unique()) == 2:
                # Classification
                model = LogisticRegression(random_state=analysis_params.random_state)
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
                score_type = 'auc'
            else:
                # Regression
                model = LinearRegression()
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, 
                                          scoring='neg_mean_squared_error')
                cv_scores = -cv_scores  # Convert back to MSE
                score_type = 'mse'
            
            # Fit final model
            model.fit(X_scaled, y)
            
            # Get feature importance
            if hasattr(model, 'coef_'):
                importances = model.coef_.flatten()
            else:
                importances = np.zeros(len(feature_cols))
            
            results[clinical_var] = {
                'cv_mean_score': np.mean(cv_scores),
                'cv_std_score': np.std(cv_scores),
                'score_type': score_type,
                'feature_importances': dict(zip(feature_cols, importances)),
                'sample_size': len(y)
            }
        
        return results
    
    def _group_comparisons(self, data: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Compare features across patient groups"""
        if 'group' not in data.columns:
            return {}
        
        groups = data['group'].unique()
        results = {}
        
        for feature in feature_cols:
            if feature not in data.columns:
                continue
            
            feature_data = data[['group', feature]].dropna()
            
            if len(feature_data) < 10:
                continue
            
            # ANOVA across groups
            group_lists = [feature_data[feature_data['group'] == group][feature] 
                          for group in groups]
            
            # Remove groups with too few samples
            group_lists = [g for g in group_lists if len(g) >= 3]
            
            if len(group_lists) < 2:
                continue
            
            f_stat, p_value = stats.f_oneway(*group_lists)
            
            # Post-hoc tests (Tukey HSD)
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukey = pairwise_tukeyhsd(feature_data[feature], feature_data['group'])
                posthoc = str(tukey)
            except:
                posthoc = "Post-hoc test failed"
            
            results[feature] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'group_means': {group: np.mean(feature_data[feature_data['group'] == group][feature])
                               for group in groups},
                'group_stds': {group: np.std(feature_data[feature_data['group'] == group][feature])
                              for group in groups},
                'posthoc_results': posthoc
            }
        
        return results

class StatisticalAnalyzer:
    """Comprehensive statistical analysis toolkit"""
    
    def __init__(self):
        self.correlator = ClinicalCorrelator()
    
    def perform_power_analysis(self, effect_size: float, alpha: float = 0.05, 
                             power: float = 0.8) -> int:
        """Calculate required sample size for given effect size"""
        from statsmodels.stats.power import TTestIndPower
        
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size, 
            alpha=alpha, 
            power=power,
            ratio=1.0
        )
        
        return int(np.ceil(sample_size))
    
    def correct_multiple_testing(self, p_values: List[float], 
                               method: str = 'fdr_bh') -> List[float]:
        """Apply multiple testing correction"""
        from statsmodels.stats.multitest import multipletests
        
        _, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method=method)
        return corrected_p.tolist()
    
    def calculate_confidence_intervals(self, data: pd.Series, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals for mean"""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
        return ci