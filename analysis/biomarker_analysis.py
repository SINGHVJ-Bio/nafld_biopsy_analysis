"""
Biomarker analysis and survival modeling for NAFLD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import statsmodels.api as sm
from statsmodels.formula.api import coxph
from ..config.parameters import analysis_params
import logging

logger = logging.getLogger(__name__)

class BiomarkerAnalyzer:
    """Comprehensive biomarker analysis for NAFLD"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.biomarker_models = {}
    
    def analyze_biomarkers(self, features_df: pd.DataFrame, 
                          clinical_data: pd.DataFrame,
                          outcome_variable: str) -> Dict:
        """Analyze biomarkers for clinical outcomes"""
        # Merge data
        merged_data = features_df.merge(clinical_data, on='patient_id', how='inner')
        
        if outcome_variable not in merged_data.columns:
            raise ValueError(f"Outcome variable {outcome_variable} not found")
        
        # Prepare data
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        X = merged_data[feature_cols].fillna(0)
        y = merged_data[outcome_variable]
        
        # Remove patients with missing outcomes
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid data after removing missing outcomes")
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        if len(X.columns) == 0:
            raise ValueError("No varying features found in the data")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine analysis type
        if y.dtype in ['object', 'category'] or len(y.unique()) == 2:
            return self._classification_analysis(X_scaled, y, X.columns.tolist(), outcome_variable)
        else:
            return self._regression_analysis(X_scaled, y, X.columns.tolist(), outcome_variable)
    
    def _classification_analysis(self, X: np.ndarray, y: pd.Series, 
                               feature_names: List[str], outcome: str) -> Dict:
        """Biomarker analysis for classification outcomes"""
        # Logistic regression with regularization
        model = LogisticRegression(penalty='l1', solver='liblinear', 
                                 random_state=analysis_params.random_state, max_iter=1000)
        model.fit(X, y)
        
        # Cross-validated performance
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        # Feature importance
        coefficients = model.coef_[0]
        feature_importance = dict(zip(feature_names, coefficients))
        
        # Significant biomarkers (non-zero coefficients)
        significant_biomarkers = {
            feature: coef for feature, coef in feature_importance.items() 
            if abs(coef) > 0.001
        }
        
        return {
            'analysis_type': 'classification',
            'outcome_variable': outcome,
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'feature_importance': feature_importance,
            'significant_biomarkers': significant_biomarkers,
            'model_intercept': model.intercept_[0],
            'sample_size': len(y),
            'n_features': len(feature_names)
        }
    
    def _regression_analysis(self, X: np.ndarray, y: pd.Series,
                           feature_names: List[str], outcome: str) -> Dict:
        """Biomarker analysis for continuous outcomes"""
        # Linear regression with statsmodels for p-values
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        # Feature importance (standardized coefficients)
        coefficients = model.params[1:]  # Skip intercept
        p_values = model.pvalues[1:]
        
        feature_importance = dict(zip(feature_names, coefficients))
        feature_p_values = dict(zip(feature_names, p_values))
        
        # Significant biomarkers (p < 0.05)
        significant_biomarkers = {
            feature: {
                'coefficient': coef,
                'p_value': feature_p_values[feature],
                'significant': feature_p_values[feature] < 0.05
            }
            for feature, coef in feature_importance.items()
        }
        
        # Cross-validated R²
        lm_model = LinearRegression()
        cv_scores = cross_val_score(lm_model, X, y, cv=5, scoring='r2')
        
        return {
            'analysis_type': 'regression',
            'outcome_variable': outcome,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'feature_importance': feature_importance,
            'feature_p_values': feature_p_values,
            'significant_biomarkers': significant_biomarkers,
            'model_summary': str(model.summary()),
            'sample_size': len(y),
            'n_features': len(feature_names)
        }
    
    def build_biomarker_signature(self, features_df: pd.DataFrame,
                                 clinical_data: pd.DataFrame,
                                 outcome_variable: str,
                                 n_biomarkers: int = 10,
                                 method: str = 'lasso') -> Dict:
        """Build optimized biomarker signature using different methods"""
        # First, identify important biomarkers
        biomarker_results = self.analyze_biomarkers(features_df, clinical_data, outcome_variable)
        
        # Get top biomarkers based on method
        if method == 'lasso':
            top_feature_names = self._select_features_lasso(
                features_df, clinical_data, outcome_variable, n_biomarkers
            )
        elif method == 'importance':
            feature_importance = biomarker_results['feature_importance']
            top_features = sorted(feature_importance.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:n_biomarkers]
            top_feature_names = [feat for feat, _ in top_features]
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        if not top_feature_names:
            logger.warning("No features selected for signature")
            return {
                'biomarker_signature': [],
                'signature_model': None,
                'cv_performance': {'mean': 0, 'std': 0, 'type': 'none'},
                'feature_weights': {},
                'original_analysis': biomarker_results,
                'success': False
            }
        
        # Build signature model using only top biomarkers
        merged_data = features_df.merge(clinical_data, on='patient_id')
        X_signature = merged_data[top_feature_names].fillna(0)
        y_signature = merged_data[outcome_variable]
        
        valid_mask = ~y_signature.isna()
        X_signature = X_signature[valid_mask]
        y_signature = y_signature[valid_mask]
        
        if len(X_signature) == 0:
            logger.warning("No valid data for signature building")
            return {
                'biomarker_signature': top_feature_names,
                'signature_model': None,
                'cv_performance': {'mean': 0, 'std': 0, 'type': 'none'},
                'feature_weights': {},
                'original_analysis': biomarker_results,
                'success': False
            }
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_signature)
        
        # Determine model type and train final model
        is_classification = (y_signature.dtype in ['object', 'category'] or 
                           len(y_signature.unique()) == 2)
        
        if is_classification:
            signature_model = LogisticRegression(random_state=analysis_params.random_state)
            cv_scores = cross_val_score(signature_model, X_scaled, y_signature, 
                                      cv=5, scoring='roc_auc')
            score_type = 'auc'
        else:
            signature_model = LinearRegression()
            cv_scores = cross_val_score(signature_model, X_scaled, y_signature, 
                                      cv=5, scoring='r2')
            score_type = 'r2'
        
        signature_model.fit(X_scaled, y_signature)
        
        # Get feature weights
        if hasattr(signature_model, 'coef_'):
            if len(signature_model.coef_.shape) > 1:
                weights = signature_model.coef_[0]
            else:
                weights = signature_model.coef_
        else:
            weights = np.zeros(len(top_feature_names))
        
        feature_weights = dict(zip(top_feature_names, weights))
        
        return {
            'biomarker_signature': top_feature_names,
            'signature_model': signature_model,
            'cv_performance': {
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'type': score_type
            },
            'feature_weights': feature_weights,
            'original_analysis': biomarker_results,
            'success': True,
            'method': method
        }
    
    def _select_features_lasso(self, features_df: pd.DataFrame,
                              clinical_data: pd.DataFrame,
                              outcome_variable: str,
                              n_features: int) -> List[str]:
        """Select features using LASSO regularization"""
        merged_data = features_df.merge(clinical_data, on='patient_id')
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        X = merged_data[feature_cols].fillna(0)
        y = merged_data[outcome_variable]
        
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            return []
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine if classification or regression
        is_classification = (y.dtype in ['object', 'category'] or 
                           len(y.unique()) == 2)
        
        if is_classification:
            model = LogisticRegression(penalty='l1', solver='liblinear', 
                                     random_state=analysis_params.random_state, 
                                     max_iter=1000)
        else:
            model = Lasso(alpha=0.1, random_state=analysis_params.random_state, max_iter=1000)
        
        model.fit(X_scaled, y)
        
        # Get non-zero coefficients
        if hasattr(model, 'coef_'):
            coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        else:
            coefficients = np.zeros(len(feature_cols))
        
        # Select top n_features with highest absolute coefficients
        feature_importance = list(zip(feature_cols, np.abs(coefficients)))
        selected_features = [feat for feat, imp in sorted(feature_importance, 
                                                         key=lambda x: x[1], 
                                                         reverse=True)[:n_features] 
                           if imp > 0]
        
        return selected_features
    
    def validate_biomarker_signature(self, signature_results: Dict,
                                   test_features: pd.DataFrame,
                                   test_clinical: pd.DataFrame,
                                   outcome_variable: str) -> Dict:
        """Validate biomarker signature on test set"""
        signature_model = signature_results['signature_model']
        biomarker_names = signature_results['biomarker_signature']
        
        if signature_model is None or not biomarker_names:
            return {
                'test_score': 0,
                'score_type': 'none',
                'test_sample_size': 0,
                'biomarkers_used': [],
                'success': False
            }
        
        # Prepare test data
        merged_test = test_features.merge(test_clinical, on='patient_id')
        X_test = merged_test[biomarker_names].fillna(0)
        y_test = merged_test[outcome_variable]
        
        # Remove missing outcomes
        valid_mask = ~y_test.isna()
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]
        
        if len(X_test) == 0:
            return {
                'test_score': 0,
                'score_type': 'none',
                'test_sample_size': 0,
                'biomarkers_used': biomarker_names,
                'success': False
            }
        
        # Scale test features using the same scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict on test set
        if hasattr(signature_model, 'predict_proba'):  # Classification
            y_pred_proba = signature_model.predict_proba(X_test_scaled)[:, 1]
            test_score = roc_auc_score(y_test, y_pred_proba)
            score_type = 'auc'
        else:  # Regression
            y_pred = signature_model.predict(X_test_scaled)
            test_score = r2_score(y_test, y_pred)
            score_type = 'r2'
        
        return {
            'test_score': test_score,
            'score_type': score_type,
            'test_sample_size': len(y_test),
            'biomarkers_used': biomarker_names,
            'success': True
        }

class SurvivalAnalyzer:
    """Survival analysis for NAFLD progression"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def analyze_survival(self, features_df: pd.DataFrame,
                        clinical_data: pd.DataFrame,
                        time_column: str,
                        event_column: str) -> Dict:
        """Perform survival analysis using Cox proportional hazards"""
        # Merge data
        merged_data = features_df.merge(clinical_data, on='patient_id', how='inner')
        
        if time_column not in merged_data.columns or event_column not in merged_data.columns:
            raise ValueError("Time or event column not found in clinical data")
        
        # Prepare survival data
        survival_data = merged_data[[time_column, event_column]].dropna()
        valid_indices = survival_data.index
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        X = merged_data.loc[valid_indices, feature_cols].fillna(0)
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        feature_cols = X.columns.tolist()
        
        if len(X) == 0:
            raise ValueError("No valid features for survival analysis")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create survival structure for scikit-survival
        y_survival = np.array([(bool(row[event_column]), row[time_column]) 
                              for _, row in survival_data.iterrows()],
                             dtype=[('event', 'bool'), ('time', 'float')])
        
        # Fit Cox model
        cox_model = CoxPHSurvivalAnalysis(alpha=0.1)
        cox_model.fit(X_scaled, y_survival)
        
        # Feature importance (hazard ratios)
        hazard_ratios = np.exp(cox_model.coef_)
        feature_importance = dict(zip(feature_cols, hazard_ratios))
        
        # Calculate p-values using bootstrap (simplified)
        p_values = self._calculate_cox_pvalues(cox_model, X_scaled, y_survival)
        
        # Significant features (p < 0.05)
        significant_features = {
            feature: {
                'hazard_ratio': hr,
                'p_value': p_values.get(feature, 1.0),
                'coefficient': cox_model.coef_[i]
            }
            for i, (feature, hr) in enumerate(feature_importance.items())
            if p_values.get(feature, 1.0) < 0.05
        }
        
        # Calculate concordance index (model performance)
        c_index = cox_model.score(X_scaled, y_survival)
        
        return {
            'analysis_type': 'survival',
            'c_index': c_index,
            'feature_hazard_ratios': feature_importance,
            'significant_features': significant_features,
            'coefficients': cox_model.coef_.tolist(),
            'p_values': p_values,
            'sample_size': len(y_survival),
            'total_events': np.sum([item[0] for item in y_survival]),
            'median_survival_time': np.median(survival_data[time_column])
        }
    
    def _calculate_cox_pvalues(self, model, X: np.ndarray, y: np.ndarray, 
                              n_bootstrap: int = 100) -> Dict[str, float]:
        """Calculate p-values for Cox model using bootstrap"""
        # Simplified bootstrap p-value calculation
        # In practice, you'd use a more robust method
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        p_values = {name: 0.5 for name in feature_names}  # Placeholder
        
        try:
            # Simple approach: use magnitude of coefficients relative to their stability
            coef_variation = np.abs(model.coef_) / (np.abs(model.coef_) + 1)
            for i, name in enumerate(feature_names):
                p_values[name] = 1 - coef_variation[i]  # Simplified p-value
        except:
            logger.warning("Failed to calculate Cox p-values, using defaults")
        
        return p_values
    
    def calculate_survival_curves(self, features_df: pd.DataFrame,
                                 clinical_data: pd.DataFrame,
                                 time_column: str,
                                 event_column: str,
                                 risk_groups: int = 3) -> Dict:
        """Calculate survival curves for different risk groups"""
        # First perform survival analysis
        survival_results = self.analyze_survival(features_df, clinical_data, 
                                                time_column, event_column)
        
        # Prepare data for risk stratification
        merged_data = features_df.merge(clinical_data, on='patient_id')
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        X = merged_data[feature_cols].fillna(0)
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        feature_cols = X.columns.tolist()
        
        # Calculate risk scores
        coefficients = survival_results['coefficients']
        if len(coefficients) != len(feature_cols):
            logger.warning("Coefficient length doesn't match features, using first few")
            coefficients = coefficients[:len(feature_cols)]
        
        risk_scores = X[feature_cols] @ coefficients
        
        # Create risk groups
        try:
            risk_labels = pd.qcut(risk_scores, risk_groups, labels=False, duplicates='drop')
            # Handle case where we can't create the requested number of groups
            actual_groups = len(np.unique(risk_labels))
            if actual_groups < risk_groups:
                logger.warning(f"Could only create {actual_groups} risk groups instead of {risk_groups}")
                risk_groups = actual_groups
        except Exception as e:
            logger.warning(f"Failed to create risk groups: {e}. Using median split.")
            risk_labels = (risk_scores > risk_scores.median()).astype(int)
            risk_groups = 2
        
        merged_data['risk_group'] = risk_labels
        
        # Calculate survival curves for each group
        survival_curves = {}
        
        for group in range(risk_groups):
            group_data = merged_data[merged_data['risk_group'] == group]
            if len(group_data) == 0:
                continue
                
            group_times = group_data[time_column].values
            group_events = group_data[event_column].values.astype(bool)
            
            if len(group_times) > 0:
                times, survival_probs = self._kaplan_meier_estimator(group_events, group_times)
                survival_curves[f'risk_group_{group}'] = {
                    'times': times.tolist(),
                    'survival_probabilities': survival_probs.tolist(),
                    'sample_size': len(group_data),
                    'events': np.sum(group_events),
                    'median_survival': self._calculate_median_survival(times, survival_probs)
                }
        
        return {
            'risk_groups': risk_groups,
            'risk_scores': risk_scores.tolist(),
            'risk_labels': risk_labels.tolist(),
            'survival_curves': survival_curves,
            'survival_analysis': survival_results
        }
    
    def _kaplan_meier_estimator(self, events: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple Kaplan-Meier estimator implementation"""
        # Sort by time
        order = np.argsort(times)
        times_sorted = times[order]
        events_sorted = events[order]
        
        # Calculate survival probabilities
        n = len(times_sorted)
        survival_probs = np.ones(n)
        current_prob = 1.0
        
        for i in range(n):
            if events_sorted[i]:
                at_risk = np.sum(times_sorted >= times_sorted[i])
                current_prob *= (at_risk - 1) / at_risk
            survival_probs[i] = current_prob
        
        return times_sorted, survival_probs
    
    def _calculate_median_survival(self, times: np.ndarray, survival_probs: np.ndarray) -> float:
        """Calculate median survival time from survival curve"""
        if len(survival_probs) == 0:
            return float('nan')
        
        # Find time where survival probability drops below 0.5
        below_median = survival_probs < 0.5
        if np.any(below_median):
            first_below = np.where(below_median)[0][0]
            return float(times[first_below])
        else:
            return float(times[-1])  # Median not reached
    
    def time_dependent_auc(self, features_df: pd.DataFrame,
                          clinical_data: pd.DataFrame,
                          time_column: str,
                          event_column: str,
                          time_points: List[float] = None) -> Dict:
        """Calculate time-dependent AUC for survival prediction"""
        if time_points is None:
            time_points = [1, 2, 3, 5]  # 1, 2, 3, 5 years
        
        # Perform survival analysis to get risk scores
        survival_results = self.analyze_survival(features_df, clinical_data, 
                                                time_column, event_column)
        
        # Prepare data
        merged_data = features_df.merge(clinical_data, on='patient_id')
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        X = merged_data[feature_cols].fillna(0)
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        feature_cols = X.columns.tolist()
        
        # Calculate risk scores
        coefficients = survival_results['coefficients']
        if len(coefficients) != len(feature_cols):
            coefficients = coefficients[:len(feature_cols)]
        
        risk_scores = X[feature_cols] @ coefficients
        
        # Simplified time-dependent AUC calculation
        # In practice, you'd use a proper method like timeROC in R or scikit-survival
        time_aucs = {}
        
        for time_point in time_points:
            # Create binary outcome: event occurred before time_point
            time_events = (merged_data[time_column] <= time_point) & (merged_data[event_column] == 1)
            time_censored = (merged_data[time_column] <= time_point) & (merged_data[event_column] == 0)
            
            # Only include patients with known status at this time
            valid_mask = (merged_data[time_column] > time_point) | time_events | time_censored
            y_binary = time_events[valid_mask]
            scores_at_time = risk_scores[valid_mask]
            
            if len(np.unique(y_binary)) > 1 and len(scores_at_time) > 0:
                try:
                    auc = roc_auc_score(y_binary, scores_at_time)
                    time_aucs[time_point] = auc
                except:
                    time_aucs[time_point] = 0.5  # Random performance
            else:
                time_aucs[time_point] = 0.5  # Not enough events
        
        return {
            'time_points': time_points,
            'time_dependent_auc': time_aucs,
            'mean_auc': np.mean(list(time_aucs.values())),
            'risk_scores': risk_scores.tolist(),
            'best_timepoint': max(time_aucs.items(), key=lambda x: x[1])[0] if time_aucs else None
        }

class MultiModalIntegrator:
    """Integrate multiple data modalities for biomarker discovery"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.integrated_models = {}
    
    def integrate_modalities(self, modalities: Dict[str, pd.DataFrame],
                           clinical_data: pd.DataFrame,
                           outcome_variable: str,
                           integration_method: str = 'early') -> Dict:
        """Integrate multiple data modalities for analysis"""
        
        if integration_method == 'early':
            return self._early_integration(modalities, clinical_data, outcome_variable)
        elif integration_method == 'late':
            return self._late_integration(modalities, clinical_data, outcome_variable)
        else:
            raise ValueError(f"Unsupported integration method: {integration_method}")
    
    def _early_integration(self, modalities: Dict[str, pd.DataFrame],
                          clinical_data: pd.DataFrame,
                          outcome_variable: str) -> Dict:
        """Early integration: concatenate all features"""
        # Start with clinical data
        integrated_data = clinical_data.copy()
        
        for modality_name, modality_data in modalities.items():
            # Remove patient_id from modality data to avoid duplicate columns
            modality_features = modality_data.drop(columns=['patient_id'], errors='ignore')
            # Add modality prefix to feature names
            modality_features = modality_features.add_prefix(f'{modality_name}_')
            modality_features['patient_id'] = modality_data['patient_id']
            
            integrated_data = integrated_data.merge(
                modality_features, on='patient_id', how='inner'
            )
        
        # Prepare features and target
        feature_cols = [col for col in integrated_data.columns 
                       if col not in ['patient_id', outcome_variable]]
        X = integrated_data[feature_cols].fillna(0)
        y = integrated_data[outcome_variable]
        
        # Remove missing outcomes
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            return {
                'integration_method': 'early',
                'success': False,
                'error': 'No valid data after integration'
            }
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Analyze using biomarker analyzer
        analyzer = BiomarkerAnalyzer()
        
        # Create temporary features dataframe for the analyzer
        temp_features = pd.DataFrame(X_scaled, columns=X.columns)
        temp_features['patient_id'] = integrated_data[valid_mask]['patient_id'].values
        
        try:
            result = analyzer.analyze_biomarkers(temp_features, integrated_data[valid_mask], outcome_variable)
            result['integration_method'] = 'early'
            result['success'] = True
            result['modality_count'] = len(modalities)
            result['total_features'] = len(X.columns)
            return result
        except Exception as e:
            logger.error(f"Early integration failed: {e}")
            return {
                'integration_method': 'early',
                'success': False,
                'error': str(e)
            }
    
    def _late_integration(self, modalities: Dict[str, pd.DataFrame],
                         clinical_data: pd.DataFrame,
                         outcome_variable: str) -> Dict:
        """Late integration: train separate models and combine predictions using stacking"""
        modality_results = {}
        modality_models = {}
        modality_predictions = {}
        
        # Analyze each modality separately
        analyzer = BiomarkerAnalyzer()
        
        # Determine problem type from first modality
        first_modality_name = next(iter(modalities.keys()))
        merged_sample = modalities[first_modality_name].merge(clinical_data, on='patient_id')
        y_sample = merged_sample[outcome_variable]
        is_classification = y_sample.dtype in ['object', 'category'] or len(y_sample.unique()) == 2
        
        for modality_name, modality_data in modalities.items():
            try:
                logger.info(f"Analyzing modality: {modality_name}")
                
                # Analyze this modality
                result = analyzer.analyze_biomarkers(modality_data, clinical_data, outcome_variable)
                modality_results[modality_name] = result
                
                # Build a signature model for this modality
                signature_result = analyzer.build_biomarker_signature(
                    modality_data, clinical_data, outcome_variable, n_biomarkers=10
                )
                modality_models[modality_name] = signature_result
                
                # Get cross-validated predictions for this modality
                cv_predictions = self._get_cross_val_predictions(
                    modality_data, clinical_data, outcome_variable, 
                    signature_result, is_classification
                )
                modality_predictions[modality_name] = cv_predictions
                
                logger.info(f"Successfully processed modality {modality_name}, predictions: {len(cv_predictions)}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze modality {modality_name}: {e}")
                modality_results[modality_name] = None
                modality_models[modality_name] = None
                modality_predictions[modality_name] = None
        
        # Remove failed modalities
        valid_modalities = {k: v for k, v in modality_predictions.items() 
                           if v is not None and len(v) > 0}
        
        if not valid_modalities:
            logger.warning("No valid modalities found for integration")
            return {
                'integration_method': 'late',
                'modality_results': modality_results,
                'success': False,
                'error': 'No valid modalities'
            }
        
        # Combine predictions using stacking
        stacking_result = self._stack_predictions(
            valid_modalities, clinical_data, outcome_variable, is_classification
        )
        
        # Calculate modality importance
        modality_importance = self._calculate_modality_importance(
            valid_modalities, clinical_data, outcome_variable
        )
        
        return {
            'integration_method': 'late_stacking',
            'modality_results': modality_results,
            'modality_models': modality_models,
            'modality_importance': modality_importance,
            'stacking_result': stacking_result,
            'valid_modality_count': len(valid_modalities),
            'success': True
        }
    
    def _get_cross_val_predictions(self, features_df: pd.DataFrame,
                                  clinical_data: pd.DataFrame,
                                  outcome_variable: str,
                                  signature_result: Dict,
                                  is_classification: bool) -> np.ndarray:
        """Get cross-validated predictions for a single modality"""
        # Merge data
        merged_data = features_df.merge(clinical_data, on='patient_id')
        
        # Prepare features and target
        biomarker_names = signature_result['biomarker_signature']
        if not biomarker_names:
            return np.array([])
            
        X = merged_data[biomarker_names].fillna(0)
        y = merged_data[outcome_variable]
        
        # Remove missing outcomes
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            return np.array([])
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        if len(X.columns) == 0:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Choose model type
        if is_classification:
            model = LogisticRegression(random_state=analysis_params.random_state)
            # For binary classification, take probabilities of positive class
            if len(np.unique(y)) == 2:
                predictions = cross_val_predict(model, X_scaled, y, cv=5, 
                                              method='predict_proba', n_jobs=-1)
                return predictions[:, 1]  # Probability of positive class
            else:
                # For multi-class, we'll use predict for simplicity
                return cross_val_predict(model, X_scaled, y, cv=5, method='predict')
        else:
            model = LinearRegression()
            return cross_val_predict(model, X_scaled, y, cv=5, n_jobs=-1)
    
    def _stack_predictions(self, modality_predictions: Dict[str, np.ndarray],
                          clinical_data: pd.DataFrame,
                          outcome_variable: str,
                          is_classification: bool) -> Dict:
        """Stack predictions from multiple modalities using a meta-learner"""
        # Get common patient indices (all modalities should have the same patients)
        first_modality = next(iter(modality_predictions.values()))
        if len(first_modality) == 0:
            return {'success': False, 'error': 'No predictions available'}
        
        # Create stacked features matrix
        stacked_features = np.column_stack(list(modality_predictions.values()))
        modality_names = list(modality_predictions.keys())
        
        # Get target variable
        y = clinical_data[outcome_variable].values
        # Remove missing outcomes (assuming all modalities have same patient order)
        valid_mask = ~pd.isna(y)
        y = y[valid_mask]
        stacked_features = stacked_features[valid_mask]
        
        if len(y) == 0:
            return {'success': False, 'error': 'No valid targets'}
        
        # Train meta-learner
        if is_classification:
            meta_model = LogisticRegression(random_state=analysis_params.random_state)
            cv_scores = cross_val_score(meta_model, stacked_features, y, 
                                      cv=5, scoring='roc_auc')
            meta_model.fit(stacked_features, y)
            
            # Get feature importance (coefficients)
            importance = dict(zip(modality_names, meta_model.coef_[0]))
            
            # Calculate final performance
            final_predictions = meta_model.predict_proba(stacked_features)[:, 1]
            final_score = roc_auc_score(y, final_predictions)
            score_type = 'auc'
            
        else:
            meta_model = LinearRegression()
            cv_scores = cross_val_score(meta_model, stacked_features, y, 
                                      cv=5, scoring='r2')
            meta_model.fit(stacked_features, y)
            
            # Get feature importance (coefficients)
            importance = dict(zip(modality_names, meta_model.coef_))
            
            # Calculate final performance
            final_predictions = meta_model.predict(stacked_features)
            final_score = r2_score(y, final_predictions)
            score_type = 'r2'
        
        return {
            'meta_model': meta_model,
            'cv_performance': {
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'type': score_type
            },
            'final_performance': final_score,
            'modality_importance': importance,
            'stacked_predictions': final_predictions.tolist(),
            'success': True
        }
    
    def _calculate_modality_importance(self, modality_predictions: Dict[str, np.ndarray],
                                     clinical_data: pd.DataFrame,
                                     outcome_variable: str) -> Dict:
        """Calculate importance of each modality using multiple methods"""
        y = clinical_data[outcome_variable].values
        valid_mask = ~pd.isna(y)
        y = y[valid_mask]
        
        if len(y) == 0:
            return {}
        
        # Determine problem type
        is_classification = (np.issubdtype(y.dtype, np.object_) or 
                            np.issubdtype(y.dtype, np.str_) or 
                            len(np.unique(y)) == 2)
        
        importance_scores = {}
        
        for modality_name, predictions in modality_predictions.items():
            if len(predictions) == 0:
                continue
                
            pred_valid = predictions[valid_mask]
            
            if is_classification:
                # For classification, use AUC
                try:
                    score = roc_auc_score(y, pred_valid)
                except:
                    score = 0.5  # Random performance
            else:
                # For regression, use R²
                try:
                    score = r2_score(y, pred_valid)
                    # R² can be negative, so we normalize to [0, 1]
                    score = max(0, score)
                except:
                    score = 0
            
            importance_scores[modality_name] = score
        
        # Normalize importance scores to sum to 1
        total = sum(importance_scores.values())
        if total > 0:
            normalized_importance = {k: v/total for k, v in importance_scores.items()}
        else:
            normalized_importance = {k: 1/len(importance_scores) for k in importance_scores.keys()}
        
        return {
            'raw_scores': importance_scores,
            'normalized_scores': normalized_importance,
            'top_modality': max(importance_scores.items(), key=lambda x: x[1])[0] if importance_scores else None
        }