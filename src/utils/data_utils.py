"""
Data Handling Utilities for StreamSplit Framework
Implements Distribution-Aware Sampling, Optimal Transport, and other data utilities
Based on Appendix E, Section 3.2.2, and related algorithms from the paper
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import random
import math
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import time

# Export main classes for import
__all__ = ['DistributionAwareSampling', 'OptimalTransport', 'KMeansOnline', 
           'EmbeddingStore', 'DataLoader', 'BatchSampler']

class DistributionAwareSampling:
    """
    Distribution-Aware Sampling (DAS) for negative selection
    Maintains GMM and provides sampling probabilities (Appendix E)
    Implements Equations 2-3 from the paper
    """
    
    def __init__(self, embedding_dim: int, n_components: int = 5, device: str = 'cpu'):
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize GMM parameters
        self.means = torch.randn(n_components, embedding_dim, device=device)
        self.covariances = torch.stack([
            torch.eye(embedding_dim, device=device) for _ in range(n_components)
        ])
        self.weights = torch.ones(n_components, device=device) / n_components
        
        # Learning parameters
        self.learning_rate = 0.01
        self.epsilon = 1e-6
        self.alpha = 1.0  # Temperature parameter for inverse sampling
        
        # Tracking variables
        self.update_count = 0
        self.last_update_time = time.time()
        
        self.logger.info(f"DistributionAwareSampling initialized with {n_components} components")
    
    def _gaussian_pdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Compute multivariate Gaussian PDF
        Used in GMM computation for DAS
        """
        diff = x - mean.unsqueeze(0)  # (N, D)
        
        # Add small epsilon for numerical stability
        cov = cov + self.epsilon * torch.eye(self.embedding_dim, device=self.device)
        
        # Compute inverse covariance
        try:
            inv_cov = torch.inverse(cov)
            det_cov = torch.det(cov)
        except:
            # Fallback to regularized version
            reg_cov = cov + 0.01 * torch.eye(self.embedding_dim, device=self.device)
            inv_cov = torch.inverse(reg_cov)
            det_cov = torch.det(reg_cov)
        
        # Compute quadratic form: (x-μ)^T Σ^(-1) (x-μ)
        quad_form = torch.sum(diff * torch.matmul(diff, inv_cov), dim=1)
        
        # Compute normalization constant
        normalization = torch.sqrt((2 * math.pi) ** self.embedding_dim * det_cov)
        
        # Compute PDF
        pdf = torch.exp(-0.5 * quad_form) / normalization
        
        return pdf
    
    def _compute_responsibilities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute responsibilities (E-step of EM algorithm)
        γ_{n,c} = π_c * N(x_n | μ_c, Σ_c) / Σ_k π_k * N(x_n | μ_k, Σ_k)
        """
        n_samples = embeddings.size(0)
        responsibilities = torch.zeros(n_samples, self.n_components, device=self.device)
        
        # Compute weighted likelihood for each component
        for c in range(self.n_components):
            pdf = self._gaussian_pdf(embeddings, self.means[c], self.covariances[c])
            responsibilities[:, c] = self.weights[c] * pdf
        
        # Normalize responsibilities
        total_resp = torch.sum(responsibilities, dim=1, keepdim=True) + self.epsilon
        responsibilities = responsibilities / total_resp
        
        return responsibilities
    
    def update(self, embeddings: torch.Tensor):
        """
        Update GMM parameters with new embeddings (online EM)
        Implements the online learning for DAS (Appendix E.2)
        """
        if embeddings.size(0) == 0:
            return
        
        embeddings = embeddings.to(self.device)
        batch_size = embeddings.size(0)
        
        # E-step: compute responsibilities
        responsibilities = self._compute_responsibilities(embeddings)
        
        # M-step: update parameters with learning rate decay
        lr = self.learning_rate / (1 + 0.001 * self.update_count)
        
        for c in range(self.n_components):
            resp_c = responsibilities[:, c]
            n_c = torch.sum(resp_c) + self.epsilon
            
            # Update mean
            weighted_sum = torch.sum(resp_c.unsqueeze(1) * embeddings, dim=0)
            new_mean = weighted_sum / n_c
            self.means[c] = (1 - lr) * self.means[c] + lr * new_mean
            
            # Update covariance
            diff = embeddings - self.means[c]
            weighted_cov = torch.matmul((resp_c.unsqueeze(1) * diff).T, diff) / n_c
            self.covariances[c] = (1 - lr) * self.covariances[c] + lr * weighted_cov
            
            # Update weight
            new_weight = n_c / batch_size
            self.weights[c] = (1 - lr) * self.weights[c] + lr * new_weight
        
        # Normalize weights
        self.weights = self.weights / torch.sum(self.weights)
        
        self.update_count += 1
    
    def get_pdf(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get probability density for embeddings under the GMM
        p_θ(e) = Σ_c π_c * N(e; μ_c, Σ_c) (Equation 2)
        """
        embeddings = embeddings.to(self.device)
        pdf = torch.zeros(embeddings.size(0), device=self.device)
        
        for c in range(self.n_components):
            component_pdf = self.weights[c] * self._gaussian_pdf(
                embeddings, self.means[c], self.covariances[c]
            )
            pdf += component_pdf
        
        return pdf
    
    def get_sampling_probabilities(self, embeddings: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """
        Get sampling probabilities using inverse density sampling
        p_sample(e_i) ∝ (p_θ(e_i) + ε)^{-α} (Equation 3)
        """
        if alpha is None:
            alpha = self.alpha
            
        # Get density estimates
        pdf = self.get_pdf(embeddings)
        
        # Inverse density sampling with temperature
        sampling_probs = 1.0 / (pdf + self.epsilon) ** alpha
        
        # Normalize probabilities
        sampling_probs = sampling_probs / torch.sum(sampling_probs)
        
        return sampling_probs
    
    def sample_negatives(self, anchor: torch.Tensor, embeddings: torch.Tensor, 
                        n_negatives: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negatives using distribution-aware sampling
        Returns sampled embeddings and their sampling probabilities
        """
        if embeddings.size(0) <= n_negatives:
            # Return all available embeddings
            probs = self.get_sampling_probabilities(embeddings)
            return embeddings, probs
        
        # Get sampling probabilities
        sampling_probs = self.get_sampling_probabilities(embeddings)
        
        # Sample indices according to probabilities
        indices = torch.multinomial(sampling_probs, n_negatives, replacement=False)
        
        # Return sampled embeddings and their probabilities
        sampled_embeddings = embeddings[indices]
        sampled_probs = sampling_probs[indices]
        
        return sampled_embeddings, sampled_probs

class OptimalTransport:
    """
    Optimal Transport utilities for distribution alignment
    Used in server-side aggregation (Section 3.2.2)
    """
    
    def __init__(self, reg: float = 0.1, max_iter: int = 100, device: str = 'cpu'):
        self.reg = reg  # Regularization parameter for Sinkhorn
        self.max_iter = max_iter
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def _sinkhorn_knopp(self, cost_matrix: torch.Tensor, 
                       source_weights: torch.Tensor,
                       target_weights: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm for regularized optimal transport
        Solves the entropic regularized optimal transport problem
        """
        # Initialize
        n_source, n_target = cost_matrix.shape
        
        # Compute kernel K = exp(-cost_matrix / reg)
        K = torch.exp(-cost_matrix / self.reg)
        
        # Initialize scaling factors
        u = torch.ones(n_source, device=self.device) / n_source
        v = torch.ones(n_target, device=self.device) / n_target
        
        # Sinkhorn iterations
        for iteration in range(self.max_iter):
            u_prev = u.clone()
            
            # Update v: v = target_weights / (K^T @ u)
            Ktu = torch.matmul(K.T, u)
            v = target_weights / (Ktu + 1e-10)
            
            # Update u: u = source_weights / (K @ v)
            Kv = torch.matmul(K, v)
            u = source_weights / (Kv + 1e-10)
            
            # Check convergence
            if torch.norm(u - u_prev) < 1e-6:
                break
        
        # Compute transport plan: T = diag(u) @ K @ diag(v)
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        
        return transport_plan
    
    def wasserstein_distance(self, source_embeddings: torch.Tensor,
                           target_embeddings: torch.Tensor,
                           source_weights: Optional[torch.Tensor] = None,
                           target_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Wasserstein distance between two sets of embeddings
        Uses Sinkhorn algorithm for efficient computation
        """
        source_embeddings = source_embeddings.to(self.device)
        target_embeddings = target_embeddings.to(self.device)
        
        # Compute cost matrix (squared Euclidean distance)
        cost_matrix = torch.cdist(source_embeddings, target_embeddings, p=2) ** 2
        
        # Default to uniform weights if not provided
        n_source, n_target = cost_matrix.shape
        if source_weights is None:
            source_weights = torch.ones(n_source, device=self.device) / n_source
        if target_weights is None:
            target_weights = torch.ones(n_target, device=self.device) / n_target
        
        # Compute optimal transport plan
        transport_plan = self._sinkhorn_knopp(cost_matrix, source_weights, target_weights)
        
        # Compute Wasserstein distance
        wasserstein_dist = torch.sum(transport_plan * cost_matrix)
        
        return wasserstein_dist
    
    def align_distributions(self, source_embeddings: torch.Tensor,
                          target_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align source distribution to target distribution using optimal transport
        Returns transport plan and aligned embeddings
        """
        # Compute transport plan
        transport_plan = self._sinkhorn_knopp(
            torch.cdist(source_embeddings, target_embeddings, p=2) ** 2,
            torch.ones(source_embeddings.size(0), device=self.device) / source_embeddings.size(0),
            torch.ones(target_embeddings.size(0), device=self.device) / target_embeddings.size(0)
        )
        
        # Apply transport plan to align embeddings
        aligned_embeddings = torch.matmul(transport_plan, target_embeddings)
        
        return transport_plan, aligned_embeddings

class KMeansOnline:
    """
    Online K-means clustering for prototype maintenance
    Used for uncertainty calculation and prototype centers
    """
    
    def __init__(self, n_clusters: int, embedding_dim: int, device: str = 'cpu'):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize cluster centers randomly
        self.centers = torch.randn(n_clusters, embedding_dim, device=device)
        self.counts = torch.zeros(n_clusters, device=device)
        self.learning_rate = 0.01
        
        self.logger = logging.getLogger(__name__)
    
    def _find_closest_cluster(self, embedding: torch.Tensor) -> int:
        """Find the closest cluster center to the embedding"""
        distances = torch.norm(self.centers - embedding, dim=1)
        return torch.argmin(distances).item()
    
    def update(self, embeddings: torch.Tensor):
        """Update cluster centers with new embeddings (online k-means)"""
        embeddings = embeddings.to(self.device)
        
        for embedding in embeddings:
            # Find closest cluster
            closest_idx = self._find_closest_cluster(embedding)
            
            # Update cluster center with decaying learning rate
            self.counts[closest_idx] += 1
            adaptive_lr = self.learning_rate / (1 + 0.001 * self.counts[closest_idx])
            
            self.centers[closest_idx] = (
                (1 - adaptive_lr) * self.centers[closest_idx] + 
                adaptive_lr * embedding
            )
    
    def get_closest_distance(self, embedding: torch.Tensor) -> float:
        """Get distance to closest cluster center (for uncertainty calculation)"""
        distances = torch.norm(self.centers - embedding, dim=1)
        return torch.min(distances).item()
    
    def get_centers(self) -> torch.Tensor:
        """Get current cluster centers"""
        return self.centers.clone()

class EmbeddingStore:
    """
    Efficient storage and retrieval of embeddings with metadata
    Supports temporal indexing and device-based organization
    """
    
    def __init__(self, max_size: int = 10000, embedding_dim: int = 128):
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        
        # Storage buffers
        self.embeddings = torch.zeros(max_size, embedding_dim)
        self.timestamps = torch.zeros(max_size)
        self.device_ids = [''] * max_size
        self.metadata = [{}] * max_size
        
        # Index management
        self.current_size = 0
        self.write_ptr = 0
        
        # Device-based indexing
        self.device_indices = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def add(self, embedding: torch.Tensor, timestamp: float, 
            device_id: str = '', metadata: Dict[str, Any] = None):
        """Add embedding with metadata to the store"""
        if metadata is None:
            metadata = {}
        
        # Store embedding and metadata
        idx = self.write_ptr
        self.embeddings[idx] = embedding.detach().cpu()
        self.timestamps[idx] = timestamp
        self.device_ids[idx] = device_id
        self.metadata[idx] = metadata.copy()
        
        # Update device index
        self.device_indices[device_id].append(idx)
        
        # Update pointers
        if self.current_size < self.max_size:
            self.current_size += 1
        else:
            # Remove old device index entry when overwriting
            old_device_id = self.device_ids[idx]
            if old_device_id in self.device_indices:
                try:
                    self.device_indices[old_device_id].remove(idx)
                except ValueError:
                    pass
        
        self.write_ptr = (self.write_ptr + 1) % self.max_size
    
    def get_by_device(self, device_id: str, max_count: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings and timestamps for a specific device"""
        indices = self.device_indices.get(device_id, [])
        
        if max_count is not None and len(indices) > max_count:
            # Get most recent embeddings
            recent_indices = sorted(indices, key=lambda i: self.timestamps[i])[-max_count:]
            indices = recent_indices
        
        if not indices:
            return torch.empty(0, self.embedding_dim), torch.empty(0)
        
        embeddings = torch.stack([self.embeddings[i] for i in indices])
        timestamps = torch.tensor([self.timestamps[i] for i in indices])
        
        return embeddings, timestamps
    
    def get_recent(self, max_age: float, max_count: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings from the last max_age seconds"""
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        # Find indices of recent embeddings
        recent_indices = []
        for i in range(self.current_size):
            if self.timestamps[i] >= cutoff_time:
                recent_indices.append(i)
        
        # Sort by timestamp and limit count
        recent_indices.sort(key=lambda i: self.timestamps[i], reverse=True)
        if max_count is not None and len(recent_indices) > max_count:
            recent_indices = recent_indices[:max_count]
        
        if not recent_indices:
            return torch.empty(0, self.embedding_dim), torch.empty(0)
        
        embeddings = torch.stack([self.embeddings[i] for i in recent_indices])
        timestamps = torch.tensor([self.timestamps[i] for i in recent_indices])
        
        return embeddings, timestamps
    
    def size(self) -> int:
        """Get current number of stored embeddings"""
        return self.current_size
    
    def get_all_devices(self) -> List[str]:
        """Get list of all device IDs"""
        return list(self.device_indices.keys())

class BatchSampler:
    """
    Smart batch sampler for contrastive learning
    Ensures diversity in batches while maintaining efficiency
    """
    
    def __init__(self, batch_size: int = 32, ensure_diversity: bool = True):
        self.batch_size = batch_size
        self.ensure_diversity = ensure_diversity
        self.logger = logging.getLogger(__name__)
    
    def sample_batch(self, embeddings: torch.Tensor, 
                    metadata: List[Dict[str, Any]] = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Sample a diverse batch of embeddings
        Returns embeddings and their indices
        """
        n_embeddings = embeddings.size(0)
        
        if n_embeddings <= self.batch_size:
            # Return all embeddings if we have fewer than batch size
            indices = list(range(n_embeddings))
            return embeddings, indices
        
        if not self.ensure_diversity or metadata is None:
            # Random sampling
            indices = random.sample(range(n_embeddings), self.batch_size)
            return embeddings[indices], indices
        
        # Diverse sampling based on metadata
        indices = self._diverse_sampling(embeddings, metadata)
        return embeddings[indices], indices
    
    def _diverse_sampling(self, embeddings: torch.Tensor, 
                         metadata: List[Dict[str, Any]]) -> List[int]:
        """Sample diverse batch based on embedding similarity and metadata"""
        n_embeddings = embeddings.size(0)
        
        # Start with random seed
        selected_indices = [random.randint(0, n_embeddings - 1)]
        remaining_indices = list(range(n_embeddings))
        remaining_indices.remove(selected_indices[0])
        
        # Greedily select diverse embeddings
        while len(selected_indices) < self.batch_size and remaining_indices:
            best_idx = None
            best_diversity = -1
            
            for idx in remaining_indices:
                # Calculate diversity score
                diversity_score = self._calculate_diversity(
                    idx, selected_indices, embeddings, metadata
                )
                
                if diversity_score > best_diversity:
                    best_diversity = diversity_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _calculate_diversity(self, candidate_idx: int, selected_indices: List[int],
                           embeddings: torch.Tensor, metadata: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for a candidate embedding"""
        if not selected_indices:
            return 1.0
        
        candidate_emb = embeddings[candidate_idx]
        selected_embs = embeddings[selected_indices]
        
        # Calculate minimum distance to selected embeddings
        distances = torch.norm(selected_embs - candidate_emb, dim=1)
        min_distance = torch.min(distances).item()
        
        # Add metadata-based diversity if available
        metadata_diversity = 1.0
        if metadata:
            candidate_meta = metadata[candidate_idx]
            # Simple diversity based on device_id if available
            if 'device_id' in candidate_meta:
                selected_devices = {metadata[i].get('device_id', '') for i in selected_indices}
                if candidate_meta['device_id'] not in selected_devices:
                    metadata_diversity = 2.0
        
        return min_distance * metadata_diversity

class DataLoader:
    """
    Efficient data loader for streaming embeddings
    Supports both batch and streaming modes
    """
    
    def __init__(self, embedding_store: EmbeddingStore, 
                 batch_size: int = 32, shuffle: bool = True):
        self.embedding_store = embedding_store
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_sampler = BatchSampler(batch_size)
        
        self.logger = logging.getLogger(__name__)
    
    def __iter__(self):
        """Iterator for batch processing"""
        # Get all current embeddings
        if self.embedding_store.current_size == 0:
            return
        
        # Create index list
        indices = list(range(self.embedding_store.current_size))
        
        if self.shuffle:
            random.shuffle(indices)
        
        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Get embeddings and metadata for batch
            batch_embeddings = torch.stack([
                self.embedding_store.embeddings[j] for j in batch_indices
            ])
            batch_timestamps = torch.tensor([
                self.embedding_store.timestamps[j] for j in batch_indices
            ])
            batch_metadata = [
                self.embedding_store.metadata[j] for j in batch_indices
            ]
            
            yield {
                'embeddings': batch_embeddings,
                'timestamps': batch_timestamps,
                'metadata': batch_metadata,
                'indices': batch_indices
            }
    
    def get_streaming_batch(self, device_id: str = None, 
                          max_age: float = None) -> Dict[str, Any]:
        """Get a batch for streaming processing"""
        if device_id is not None:
            embeddings, timestamps = self.embedding_store.get_by_device(
                device_id, max_count=self.batch_size
            )
        elif max_age is not None:
            embeddings, timestamps = self.embedding_store.get_recent(
                max_age, max_count=self.batch_size
            )
        else:
            # Get most recent embeddings
            embeddings, timestamps = self.embedding_store.get_recent(
                float('inf'), max_count=self.batch_size
            )
        
        if embeddings.size(0) == 0:
            return None
        
        return {
            'embeddings': embeddings,
            'timestamps': timestamps,
            'batch_size': embeddings.size(0)
        }

# Utility functions

def create_das_sampler(embedding_dim: int, n_components: int = 5, 
                      device: str = 'cpu') -> DistributionAwareSampling:
    """Create Distribution-Aware Sampling instance"""
    return DistributionAwareSampling(embedding_dim, n_components, device)

def create_optimal_transport(reg: float = 0.1, max_iter: int = 100, 
                           device: str = 'cpu') -> OptimalTransport:
    """Create Optimal Transport instance"""
    return OptimalTransport(reg, max_iter, device)

def create_embedding_store(max_size: int = 10000, 
                         embedding_dim: int = 128) -> EmbeddingStore:
    """Create embedding store with specified capacity"""
    return EmbeddingStore(max_size, embedding_dim)

# Testing and validation functions

def test_distribution_aware_sampling():
    """Test DAS functionality"""
    print("Testing Distribution-Aware Sampling...")
    
    # Create test embeddings
    embedding_dim = 128
    n_samples = 1000
    embeddings = torch.randn(n_samples, embedding_dim)
    
    # Initialize DAS
    das = DistributionAwareSampling(embedding_dim, n_components=3)
    
    # Update with embeddings
    das.update(embeddings)
    
    # Test sampling
    anchor = torch.randn(embedding_dim)
    negatives, probs = das.sample_negatives(anchor, embeddings, n_negatives=64)
    
    print(f"✓ Sampled {negatives.size(0)} negatives")
    print(f"✓ Probability sum: {probs.sum():.4f}")
    print("Distribution-Aware Sampling test completed!")

def test_optimal_transport():
    """Test Optimal Transport functionality"""
    print("\nTesting Optimal Transport...")
    
    # Create test distributions
    source = torch.randn(100, 64)
    target = torch.randn(150, 64)
    
    # Initialize OT
    ot = OptimalTransport(reg=0.1)
    
    # Compute Wasserstein distance
    distance = ot.wasserstein_distance(source, target)
    print(f"✓ Wasserstein distance: {distance:.4f}")
    
    # Test alignment
    transport_plan, aligned = ot.align_distributions(source, target)
    print(f"✓ Transport plan shape: {transport_plan.shape}")
    print(f"✓ Aligned embeddings shape: {aligned.shape}")
    print("Optimal Transport test completed!")

def test_embedding_store():
    """Test EmbeddingStore functionality"""
    print("\nTesting EmbeddingStore...")
    
    # Create store
    store = create_embedding_store(max_size=1000, embedding_dim=128)
    
    # Add test embeddings
    for i in range(150):
        embedding = torch.randn(128)
        timestamp = time.time() + i
        device_id = f"device_{i % 3}"
        metadata = {'index': i, 'type': 'test'}
        
        store.add(embedding, timestamp, device_id, metadata)
    
    print(f"✓ Stored {store.size()} embeddings")
    
    # Test device-based retrieval
    device_embeddings, device_timestamps = store.get_by_device('device_0')
    print(f"✓ Device 0 has {device_embeddings.size(0)} embeddings")
    
    # Test recent retrieval
    recent_embeddings, recent_timestamps = store.get_recent(max_age=100)
    print(f"✓ Found {recent_embeddings.size(0)} recent embeddings")
    
    print("EmbeddingStore test completed!")

if __name__ == "__main__":
    # Run tests
    test_distribution_aware_sampling()
    test_optimal_transport()
    test_embedding_store()
    print("\nAll data utilities tests completed successfully!")