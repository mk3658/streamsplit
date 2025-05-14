"""
Loss Functions for StreamSplit Framework
Implements Sliced-Wasserstein + Laplacian hybrid loss and other contrastive losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.stats import wasserstein_distance
import math

class LocalContrastiveLoss(nn.Module):
    """
    Local contrastive loss for edge devices with weighted negatives
    Equation 4 from the paper
    """
    
    def __init__(self, temperature: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negatives: torch.Tensor, negative_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embedding (D,)
            positive: Positive embedding (D,)
            negatives: Negative embeddings (N, D)
            negative_weights: Weights for negatives (N,)
        """
        # Ensure tensors are on the same device
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negatives = negatives.to(self.device)
        
        if negative_weights is None:
            negative_weights = torch.ones(negatives.size(0), device=self.device)
        else:
            negative_weights = negative_weights.to(self.device)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # Calculate similarities
        pos_sim = torch.dot(anchor, positive) / self.temperature
        neg_sims = torch.matmul(negatives, anchor) / self.temperature
        
        # Weight negatives and compute denominator
        weighted_neg_exp = torch.sum(negative_weights * torch.exp(neg_sims))
        
        # Contrastive loss
        loss = -pos_sim + torch.log(torch.exp(pos_sim) + weighted_neg_exp)
        
        return loss

class SlicedWassersteinLoss(nn.Module):
    """
    Sliced-Wasserstein distance loss for distribution alignment
    Equation 14 from the paper
    """
    
    def __init__(self, num_projections: int = 100, device: str = 'cpu'):
        super().__init__()
        self.num_projections = num_projections
        self.device = device
        
    def _random_projections(self, embedding_dim: int, num_projections: int) -> torch.Tensor:
        """Generate random unit vectors for projections"""
        # Generate random vectors from standard normal distribution
        projections = torch.randn(embedding_dim, num_projections, device=self.device)
        # Normalize to unit vectors
        projections = F.normalize(projections, dim=0)
        return projections
    
    def _compute_1d_wasserstein(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        """Compute 1D Wasserstein distance between projected distributions"""
        # Sort the projections
        x_sorted, _ = torch.sort(x_proj)
        y_sorted, _ = torch.sort(y_proj)
        
        # Handle different sizes by interpolation
        if x_sorted.size(0) != y_sorted.size(0):
            # Interpolate to the same number of points
            n_points = min(x_sorted.size(0), y_sorted.size(0))
            x_indices = torch.linspace(0, x_sorted.size(0) - 1, n_points, device=self.device)
            y_indices = torch.linspace(0, y_sorted.size(0) - 1, n_points, device=self.device)
            
            # Interpolate
            x_sorted = torch.index_select(x_sorted, 0, x_indices.long())
            y_sorted = torch.index_select(y_sorted, 0, y_indices.long())
        
        # Compute L2 Wasserstein distance
        wasserstein_dist = torch.mean((x_sorted - y_sorted) ** 2)
        return wasserstein_dist
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute sliced Wasserstein distance between two sets of embeddings
        
        Args:
            x: First set of embeddings (N1, D)
            y: Second set of embeddings (N2, D)
        """
        # Ensure tensors are on the same device
        x = x.to(self.device)
        y = y.to(self.device)
        
        embedding_dim = x.size(1)
        
        # Generate random projections
        projections = self._random_projections(embedding_dim, self.num_projections)
        
        # Project embeddings onto random directions
        x_projected = torch.matmul(x, projections)  # (N1, num_projections)
        y_projected = torch.matmul(y, projections)  # (N2, num_projections)
        
        # Compute 1D Wasserstein distance for each projection
        sw_distances = []
        for i in range(self.num_projections):
            wd = self._compute_1d_wasserstein(x_projected[:, i], y_projected[:, i])
            sw_distances.append(wd)
        
        # Average over all projections
        sliced_wasserstein_dist = torch.mean(torch.stack(sw_distances))
        
        # Take square root as in equation 14
        return torch.sqrt(sliced_wasserstein_dist)

class LaplacianRegularizationLoss(nn.Module):
    """
    Laplacian regularization loss for preserving local structure
    Equation 15 from the paper
    """
    
    def __init__(self, k_neighbors: int = 5, sigma: float = 1.0, device: str = 'cpu'):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.device = device
    
    def _compute_adjacency_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute k-NN adjacency matrix with Gaussian weights"""
        n = embeddings.size(0)
        
        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings, p=2)
        
        # Find k-nearest neighbors
        _, indices = torch.topk(dists, k=self.k_neighbors + 1, largest=False)
        indices = indices[:, 1:]  # Exclude self
        
        # Create adjacency matrix
        adjacency = torch.zeros(n, n, device=self.device)
        
        for i in range(n):
            for j in indices[i]:
                # Symmetric k-NN graph
                dist_ij = dists[i, j]
                weight = torch.exp(-(dist_ij ** 2) / (2 * self.sigma ** 2))
                adjacency[i, j] = weight
                adjacency[j, i] = weight
        
        return adjacency
    
    def _compute_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute graph Laplacian from adjacency matrix"""
        # Compute degree matrix
        degree = torch.sum(adjacency, dim=1)
        degree_matrix = torch.diag(degree)
        
        # Laplacian = D - W
        laplacian = degree_matrix - adjacency
        
        return laplacian
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian regularization loss
        
        Args:
            embeddings: Input embeddings (N, D)
        """
        embeddings = embeddings.to(self.device)
        
        # Compute adjacency matrix
        adjacency = self._compute_adjacency_matrix(embeddings)
        
        # Compute Laplacian
        laplacian = self._compute_laplacian(adjacency)
        
        # Compute regularization term: tr(E^T L E)
        # This is equivalent to sum_{i,j} W_{ij} ||e_i - e_j||^2
        laplacian_loss = torch.trace(torch.matmul(torch.matmul(embeddings.T, laplacian), embeddings))
        
        # Normalize by number of embeddings
        laplacian_loss = laplacian_loss / (embeddings.size(0) ** 2)
        
        return laplacian_loss

class HybridSWLaplacianLoss(nn.Module):
    """
    Hybrid loss combining Sliced-Wasserstein and Laplacian regularization
    L_server = L_SW + λ * L_Lap (Section 3.2.3)
    """
    
    def __init__(self, 
                 num_projections: int = 100,
                 k_neighbors: int = 5,
                 lambda_laplacian: float = 0.5,
                 sigma_laplacian: float = 1.0,
                 device: str = 'cpu'):
        super().__init__()
        
        self.sw_loss = SlicedWassersteinLoss(num_projections, device)
        self.laplacian_loss = LaplacianRegularizationLoss(k_neighbors, sigma_laplacian, device)
        self.lambda_laplacian = lambda_laplacian
        self.device = device
    
    def forward(self, device_embeddings: torch.Tensor, 
                global_embeddings: torch.Tensor,
                all_embeddings: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hybrid loss
        
        Args:
            device_embeddings: Embeddings from specific device (N1, D)
            global_embeddings: Global distribution samples (N2, D)
            all_embeddings: All embeddings for Laplacian regularization (N, D)
        """
        # Compute Sliced-Wasserstein loss for distribution alignment
        sw_loss = self.sw_loss(device_embeddings, global_embeddings)
        
        # Compute Laplacian regularization
        if all_embeddings is not None:
            laplacian_loss = self.laplacian_loss(all_embeddings)
        else:
            # Use device embeddings if all embeddings not available
            laplacian_loss = self.laplacian_loss(device_embeddings)
        
        # Combine losses
        total_loss = sw_loss + self.lambda_laplacian * laplacian_loss
        
        # Return detailed loss breakdown
        loss_breakdown = {
            'sw_loss': sw_loss,
            'laplacian_loss': laplacian_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_breakdown

class DistributionAwareSampling:
    """
    Distribution-Aware Sampling for negative selection
    Maintains GMM and provides sampling probabilities (Appendix E)
    """
    
    def __init__(self, embedding_dim: int, n_components: int = 5, device: str = 'cpu'):
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.device = device
        
        # Initialize GMM parameters
        self.means = torch.randn(n_components, embedding_dim, device=device)
        self.covariances = torch.stack([torch.eye(embedding_dim, device=device) for _ in range(n_components)])
        self.weights = torch.ones(n_components, device=device) / n_components
        
        # Learning parameters
        self.lr = 0.01
        self.epsilon = 1e-6
        
    def _gaussian_pdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Compute multivariate Gaussian PDF"""
        diff = x - mean
        inv_cov = torch.inverse(cov + self.epsilon * torch.eye(self.embedding_dim, device=self.device))
        
        # Compute density
        exp_term = -0.5 * torch.sum(diff * torch.matmul(diff, inv_cov), dim=1)
        normalization = torch.sqrt((2 * math.pi) ** self.embedding_dim * torch.det(cov))
        
        return torch.exp(exp_term) / normalization
    
    def _compute_responsibilities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute responsibilities (E-step)"""
        n_samples = embeddings.size(0)
        responsibilities = torch.zeros(n_samples, self.n_components, device=self.device)
        
        for c in range(self.n_components):
            responsibilities[:, c] = self.weights[c] * self._gaussian_pdf(
                embeddings, self.means[c], self.covariances[c]
            )
        
        # Normalize responsibilities
        responsibilities = responsibilities / (torch.sum(responsibilities, dim=1, keepdim=True) + self.epsilon)
        
        return responsibilities
    
    def update(self, embeddings: torch.Tensor):
        """Update GMM parameters with new embeddings (online EM)"""
        if embeddings.size(0) == 0:
            return
        
        embeddings = embeddings.to(self.device)
        
        # E-step: compute responsibilities
        responsibilities = self._compute_responsibilities(embeddings)
        
        # M-step: update parameters
        for c in range(self.n_components):
            resp_c = responsibilities[:, c]
            n_c = torch.sum(resp_c)
            
            if n_c > 0:
                # Update mean
                new_mean = torch.sum(resp_c.unsqueeze(1) * embeddings, dim=0) / n_c
                self.means[c] = (1 - self.lr) * self.means[c] + self.lr * new_mean
                
                # Update covariance
                diff = embeddings - self.means[c]
                new_cov = torch.matmul((resp_c.unsqueeze(1) * diff).T, diff) / n_c
                self.covariances[c] = (1 - self.lr) * self.covariances[c] + self.lr * new_cov
                
                # Update weight
                self.weights[c] = (1 - self.lr) * self.weights[c] + self.lr * (n_c / embeddings.size(0))
        
        # Normalize weights
        self.weights = self.weights / torch.sum(self.weights)
    
    def get_pdf(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get probability density for embeddings"""
        embeddings = embeddings.to(self.device)
        
        pdf = torch.zeros(embeddings.size(0), device=self.device)
        
        for c in range(self.n_components):
            component_pdf = self.weights[c] * self._gaussian_pdf(
                embeddings, self.means[c], self.covariances[c]
            )
            pdf += component_pdf
        
        return pdf
    
    def get_sampling_probabilities(self, embeddings: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Get sampling probabilities (inverse density sampling)"""
        # Get density estimates
        pdf = self.get_pdf(embeddings)
        
        # Inverse density sampling with temperature
        sampling_probs = 1.0 / (pdf + self.epsilon) ** alpha
        
        # Normalize
        sampling_probs = sampling_probs / torch.sum(sampling_probs)
        
        return sampling_probs

class OptimalTransport:
    """
    Optimal Transport utilities for distribution alignment
    Used in server-side aggregation
    """
    
    def __init__(self, reg: float = 0.1, max_iter: int = 100, device: str = 'cpu'):
        self.reg = reg  # Regularization parameter
        self.max_iter = max_iter
        self.device = device
    
    def _sinkhorn_knopp(self, cost_matrix: torch.Tensor, 
                       source_weights: torch.Tensor,
                       target_weights: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-Knopp algorithm for regularized optimal transport"""
        # Initialize
        n_source, n_target = cost_matrix.shape
        
        # Compute kernel K = exp(-cost_matrix / reg)
        K = torch.exp(-cost_matrix / self.reg)
        
        # Initialize scaling factors
        u = torch.ones(n_source, device=self.device) / n_source
        v = torch.ones(n_target, device=self.device) / n_target
        
        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u_prev = u.clone()
            
            # Update v
            v = target_weights / (K.T @ u)
            
            # Update u
            u = source_weights / (K @ v)
            
            # Check convergence
            if torch.norm(u - u_prev) < 1e-6:
                break
        
        # Compute transport plan
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        
        return transport_plan
    
    def wasserstein_distance(self, source_embeddings: torch.Tensor,
                           target_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute Wasserstein distance between two sets of embeddings"""
        source_embeddings = source_embeddings.to(self.device)
        target_embeddings = target_embeddings.to(self.device)
        
        # Compute cost matrix (squared Euclidean distance)
        cost_matrix = torch.cdist(source_embeddings, target_embeddings, p=2) ** 2
        
        # Uniform weights
        n_source, n_target = cost_matrix.shape
        source_weights = torch.ones(n_source, device=self.device) / n_source
        target_weights = torch.ones(n_target, device=self.device) / n_target
        
        # Compute optimal transport plan
        transport_plan = self._sinkhorn_knopp(cost_matrix, source_weights, target_weights)
        
        # Compute Wasserstein distance
        wasserstein_dist = torch.sum(transport_plan * cost_matrix)
        
        return wasserstein_dist

# Additional loss functions for comparison

class KLDivergenceLoss(nn.Module):
    """KL Divergence loss for distribution comparison"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        # Convert to probability distributions
        logits1 = F.log_softmax(embeddings1 / self.temperature, dim=1)
        logits2 = F.softmax(embeddings2 / self.temperature, dim=1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(logits1, logits2, reduction='batchmean')
        
        return kl_loss

class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss"""
    
    def __init__(self, kernel_bandwidth: float = 1.0):
        super().__init__()
        self.bandwidth = kernel_bandwidth
    
    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF kernel"""
        distances = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-distances / (2 * self.bandwidth ** 2))
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        # Compute kernel matrices
        k_xx = self._gaussian_kernel(embeddings1, embeddings1)
        k_yy = self._gaussian_kernel(embeddings2, embeddings2)
        k_xy = self._gaussian_kernel(embeddings1, embeddings2)
        
        # Compute MMD
        m, n = embeddings1.size(0), embeddings2.size(0)
        mmd = (torch.sum(k_xx) / (m * m) + 
               torch.sum(k_yy) / (n * n) - 
               2 * torch.sum(k_xy) / (m * n))
        
        return mmd

# Factory functions for creating loss functions

def create_local_contrastive_loss(temperature: float = 0.1, device: str = 'cpu') -> LocalContrastiveLoss:
    """Create local contrastive loss with specified parameters"""
    return LocalContrastiveLoss(temperature=temperature, device=device)

def create_hybrid_loss(num_projections: int = 100,
                      k_neighbors: int = 5,
                      lambda_laplacian: float = 0.5,
                      device: str = 'cpu') -> HybridSWLaplacianLoss:
    """Create hybrid SW+Laplacian loss with specified parameters"""
    return HybridSWLaplacianLoss(
        num_projections=num_projections,
        k_neighbors=k_neighbors,
        lambda_laplacian=lambda_laplacian,
        device=device
    )