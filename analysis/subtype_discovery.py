"""
Subtype discovery and clustering analysis for NAFLD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from ..config.parameters import analysis_params
import logging

logger = logging.getLogger(__name__)

class SubtypeAnalyzer:
    """Discover patient subtypes using unsupervised learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_results = {}
        self.optimal_clusters = {}
    
    def analyze_subtypes(self, features_df: pd.DataFrame, 
                        clinical_data: pd.DataFrame = None,
                        n_clusters_range: Tuple[int, int] = (2, 8)) -> Dict:
        """Comprehensive subtype discovery analysis"""
        # Prepare feature matrix
        feature_cols = [col for col in features_df.columns if col != 'patient_id']
        X = features_df[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Dimensionality reduction for visualization
        reduction_results = self._perform_dimensionality_reduction(X_scaled)
        
        # Find optimal number of clusters
        optimal_k = self._find_optimal_clusters(X_scaled, n_clusters_range)
        
        # Apply clustering with optimal k
        clustering_methods = ['kmeans', 'hierarchical', 'gmm']
        cluster_results = {}
        
        for method in clustering_methods:
            clusters = self._apply_clustering(X_scaled, method, optimal_k)
            cluster_results[method] = clusters
        
        # Consensus clustering
        consensus_clusters = self._consensus_clustering(X_scaled, cluster_results)
        
        # Characterize subtypes
        if clinical_data is not None:
            subtype_characterization = self._characterize_subtypes(
                features_df, clinical_data, consensus_clusters
            )
        else:
            subtype_characterization = {}
        
        # Compile results
        self.clustering_results = {
            'optimal_clusters': optimal_k,
            'cluster_assignments': consensus_clusters,
            'dimensionality_reduction': reduction_results,
            'individual_clusterings': cluster_results,
            'subtype_characterization': subtype_characterization,
            'feature_matrix': X_scaled,
            'patient_ids': features_df['patient_id'].tolist()
        }
        
        return self.clustering_results
    
    def _perform_dimensionality_reduction(self, X: np.ndarray) -> Dict:
        """Perform multiple dimensionality reduction techniques"""
        results = {}
        
        # PCA
        pca = PCA(n_components=2, random_state=analysis_params.random_state)
        X_pca = pca.fit_transform(X)
        results['pca'] = {
            'components': X_pca,
            'explained_variance': pca.explained_variance_ratio_.tolist()
        }
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=analysis_params.random_state, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        results['tsne'] = {'components': X_tsne}
        
        # UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=analysis_params.random_state)
        X_umap = umap_reducer.fit_transform(X)
        results['umap'] = {'components': X_umap}
        
        return results
    
    def _find_optimal_clusters(self, X: np.ndarray, k_range: Tuple[int, int]) -> int:
        """Find optimal number of clusters using multiple metrics"""
        k_min, k_max = k_range
        k_values = range(k_min, k_max + 1)
        
        metrics = {
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for k in k_values:
            if k == 1:
                # Skip k=1 for metrics that require multiple clusters
                metrics['silhouette'].append(0)
                metrics['calinski_harabasz'].append(0)
                metrics['davies_bouldin'].append(float('inf'))
                continue
            
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=analysis_params.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                metrics['silhouette'].append(silhouette_score(X, labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
            else:
                metrics['silhouette'].append(0)
                metrics['calinski_harabasz'].append(0)
                metrics['davies_bouldin'].append(float('inf'))
        
        # Find optimal k (maximize silhouette and calinski, minimize davies-bouldin)
        silhouette_optimal = k_values[np.argmax(metrics['silhouette'])]
        calinski_optimal = k_values[np.argmax(metrics['calinski_harabasz'])]
        davies_optimal = k_values[np.argmin(metrics['davies_bouldin'])]
        
        # Consensus optimal k (simple majority)
        optimal_k = max(set([silhouette_optimal, calinski_optimal, davies_optimal]), 
                       key=[silhouette_optimal, calinski_optimal, davies_optimal].count)
        
        self.optimal_clusters = {
            'k_values': list(k_values),
            'silhouette_scores': metrics['silhouette'],
            'calinski_scores': metrics['calinski_harabasz'],
            'davies_scores': metrics['davies_bouldin'],
            'optimal_k': optimal_k
        }
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def _apply_clustering(self, X: np.ndarray, method: str, n_clusters: int) -> np.ndarray:
        """Apply specific clustering method"""
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=analysis_params.random_state, n_init=10)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=analysis_params.random_state)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        if method == 'gmm':
            labels = model.fit_predict(X)
        else:
            labels = model.fit_predict(X)
        
        return labels
    
    def _consensus_clustering(self, X: np.ndarray, 
                            individual_clusterings: Dict[str, np.ndarray]) -> np.ndarray:
        """Create consensus clusters from multiple methods"""
        # Simple majority voting for consensus
        all_labels = np.array(list(individual_clusterings.values()))
        
        # For each sample, assign to most frequent cluster across methods
        consensus_labels = []
        for i in range(X.shape[0]):
            sample_labels = all_labels[:, i]
            unique, counts = np.unique(sample_labels, return_counts=True)
            consensus_label = unique[np.argmax(counts)]
            consensus_labels.append(consensus_label)
        
        return np.array(consensus_labels)
    
    def _characterize_subtypes(self, features_df: pd.DataFrame, 
                             clinical_data: pd.DataFrame, 
                             cluster_labels: np.ndarray) -> Dict:
        """Characterize discovered subtypes"""
        # Merge data
        data_with_clusters = features_df.copy()
        data_with_clusters['subtype'] = cluster_labels
        
        if clinical_data is not None:
            data_with_clusters = data_with_clusters.merge(clinical_data, on='patient_id')
        
        characterization = {}
        subtypes = np.unique(cluster_labels)
        
        for subtype in subtypes:
            subtype_data = data_with_clusters[data_with_clusters['subtype'] == subtype]
            
            # Feature characteristics
            feature_cols = [col for col in features_df.columns if col != 'patient_id']
            feature_means = subtype_data[feature_cols].mean().to_dict()
            feature_stds = subtype_data[feature_cols].std().to_dict()
            
            # Clinical characteristics (if available)
            clinical_chars = {}
            if clinical_data is not None:
                clinical_cols = [col for col in clinical_data.columns 
                               if col not in ['patient_id', 'group']]
                for col in clinical_cols:
                    if col in subtype_data.columns:
                        clinical_chars[col] = {
                            'mean': subtype_data[col].mean(),
                            'std': subtype_data[col].std(),
                            'count': len(subtype_data[col].dropna())
                        }
            
            characterization[f'subtype_{subtype}'] = {
                'size': len(subtype_data),
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'clinical_characteristics': clinical_chars,
                'patient_ids': subtype_data['patient_id'].tolist()
            }
        
        return characterization
    
    def generate_subtype_report(self) -> pd.DataFrame:
        """Generate comprehensive subtype report"""
        if not self.clustering_results:
            raise ValueError("No clustering results available. Run analyze_subtypes first.")
        
        report_data = []
        
        for subtype, char_data in self.clustering_results['subtype_characterization'].items():
            report_entry = {
                'subtype': subtype,
                'patient_count': char_data['size'],
                **char_data['feature_means']  # Add feature means
            }
            report_data.append(report_entry)
        
        return pd.DataFrame(report_data)

class ClusterValidator:
    """Validate clustering results and stability"""
    
    def __init__(self):
        self.stability_scores = {}
    
    def assess_cluster_stability(self, X: np.ndarray, labels: np.ndarray, 
                               n_iterations: int = 10) -> Dict:
        """Assess clustering stability using bootstrap"""
        from sklearn.utils import resample
        
        stability_scores = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            X_bootstrap, labels_bootstrap = resample(X, labels, 
                                                   random_state=analysis_params.random_state + i)
            
            # Recluster bootstrap sample
            k = len(np.unique(labels))
            kmeans = KMeans(n_clusters=k, random_state=analysis_params.random_state)
            new_labels = kmeans.fit_predict(X_bootstrap)
            
            # Calculate agreement with original labels
            agreement = self._calculate_label_agreement(labels_bootstrap, new_labels)
            stability_scores.append(agreement)
        
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'stability_scores': stability_scores
        }
    
    def _calculate_label_agreement(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate agreement between two label sets (adjusted rand index)"""
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(labels1, labels2)