"""
Edge Module for StreamSplit Framework
Implements streaming contrastive learning with dynamic feature extraction and memory bank
"""

import asyncio
import time
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import psutil
import threading
from dataclasses import dataclass

from ..models.encoders import MobileNetV3EdgeEncoder
from ..models.losses import LocalContrastiveLoss
from ..utils.audio_processing import AudioAugmentations
from ..utils.data_utils import DistributionAwareSampling

@dataclass
class EdgeResourceState:
    """Current resource state of edge device"""
    cpu_utilization: float
    memory_usage: float
    available_memory: float
    temperature: float = 0.0
    battery_level: float = 100.0

class MemoryBank:
    """Adaptive memory bank for negative sampling with Distribution-Aware Sampling"""
    
    def __init__(self, min_size: int = 64, max_size: int = 512, 
                 embedding_dim: int = 128, device: str = 'cpu'):
        self.min_size = min_size
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Dynamic memory bank
        self.embeddings = torch.zeros((max_size, embedding_dim), device=device)
        self.timestamps = torch.zeros(max_size, device=device)
        self.current_size = 0
        self.write_ptr = 0
        
        # Distribution-Aware Sampling (GMM)
        self.das = DistributionAwareSampling(embedding_dim, n_components=5)
        
        # Age decay factor for temporal weighting
        self.age_decay = 0.99
        
    def add(self, embedding: torch.Tensor, timestamp: float):
        """Add new embedding to memory bank"""
        if self.current_size < self.max_size:
            self.embeddings[self.current_size] = embedding.detach()
            self.timestamps[self.current_size] = timestamp
            self.current_size += 1
        else:
            # Circular buffer
            self.embeddings[self.write_ptr] = embedding.detach()
            self.timestamps[self.write_ptr] = timestamp
            self.write_ptr = (self.write_ptr + 1) % self.max_size
        
        # Update DAS with new embedding
        self.das.update(embedding.unsqueeze(0))
    
    def sample_negatives(self, anchor: torch.Tensor, n_negatives: int, 
                        current_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample negatives using Distribution-Aware Sampling with age weighting"""
        if self.current_size < n_negatives:
            # Not enough negatives, return all available
            negatives = self.embeddings[:self.current_size]
            weights = self._compute_age_weights(self.timestamps[:self.current_size], current_time)
            return negatives, weights
        
        # Get sampling probabilities from DAS
        available_embeddings = self.embeddings[:self.current_size]
        sampling_probs = self.das.get_sampling_probabilities(available_embeddings)
        
        # Adjust probabilities with age weighting
        age_weights = self._compute_age_weights(self.timestamps[:self.current_size], current_time)
        combined_probs = sampling_probs * age_weights
        combined_probs = combined_probs / combined_probs.sum()
        
        # Sample indices
        indices = torch.multinomial(combined_probs, n_negatives, replacement=False)
        
        return available_embeddings[indices], age_weights[indices]
    
    def _compute_age_weights(self, timestamps: torch.Tensor, current_time: float) -> torch.Tensor:
        """Compute age-based weights for temporal coherence"""
        ages = current_time - timestamps
        weights = torch.pow(self.age_decay, ages)
        return weights
    
    def resize(self, new_size: int):
        """Dynamically resize memory bank based on available memory"""
        new_size = max(self.min_size, min(new_size, self.max_size))
        
        if new_size < self.current_size:
            # Keep most recent embeddings
            keep_indices = torch.argsort(self.timestamps[:self.current_size], descending=True)[:new_size]
            self.embeddings[:new_size] = self.embeddings[keep_indices]
            self.timestamps[:new_size] = self.timestamps[keep_indices]
            self.current_size = new_size
            self.write_ptr = self.current_size % new_size

class UncertaintyEstimator:
    """Estimate embedding uncertainty for transmission decisions"""
    
    def __init__(self, n_prototypes: int = 10, embedding_dim: int = 128):
        self.n_prototypes = n_prototypes
        self.prototypes = None
        self.embedding_dim = embedding_dim
        
    def update_prototypes(self, embeddings: torch.Tensor):
        """Update prototype centers using k-means-like approach"""
        if self.prototypes is None:
            # Initialize prototypes randomly
            self.prototypes = embeddings[:self.n_prototypes].clone()
        else:
            # Simple online update (could be improved with full k-means)
            for embedding in embeddings:
                # Find closest prototype
                distances = torch.norm(self.prototypes - embedding, dim=1)
                closest_idx = torch.argmin(distances)
                
                # Update closest prototype with learning rate
                lr = 0.01
                self.prototypes[closest_idx] = (1 - lr) * self.prototypes[closest_idx] + lr * embedding
    
    def calculate_uncertainty(self, embedding: torch.Tensor, 
                            augmented_embedding: torch.Tensor) -> Dict[str, float]:
        """Calculate multiple uncertainty metrics"""
        uncertainties = {}
        
        # 1. Consistency uncertainty (between original and augmented)
        consistency_uncertainty = F.mse_loss(embedding, augmented_embedding).item()
        uncertainties['consistency_uncertainty'] = consistency_uncertainty
        
        # 2. Prototype uncertainty (distance to nearest prototype)
        if self.prototypes is not None:
            distances = torch.norm(self.prototypes - embedding, dim=1)
            prototype_uncertainty = torch.min(distances).item()
            uncertainties['prototype_uncertainty'] = prototype_uncertainty
        
        # 3. Entropy uncertainty (simplified as norm of embedding)
        entropy_uncertainty = -torch.norm(embedding).item()
        uncertainties['entropy_uncertainty'] = entropy_uncertainty
        
        return uncertainties

class EdgeModule:
    """
    Edge module implementing streaming contrastive learning with adaptive processing
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoder
        self.encoder = MobileNetV3EdgeEncoder(
            input_dim=config.n_mels,
            embedding_dim=128,
            width_multiplier=0.75
        ).to(self.device)
        
        # Momentum encoder for consistency loss
        self.encoder_momentum = MobileNetV3EdgeEncoder(
            input_dim=config.n_mels,
            embedding_dim=128,
            width_multiplier=0.75
        ).to(self.device)
        
        # Copy weights to momentum encoder
        self._update_momentum_encoder(momentum=0.0)
        
        # Initialize memory bank
        self.memory_bank = MemoryBank(
            min_size=config.memory_bank_size_min,
            max_size=config.memory_bank_size_max,
            device=str(self.device)
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Audio augmentations
        self.augmentations = AudioAugmentations(config.sample_rate)
        
        # Loss function
        self.contrastive_loss = LocalContrastiveLoss(
            temperature=config.temperature,
            device=self.device
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.learning_rate
        )
        
        # Adaptive feature extraction policy
        self.resource_threshold = config.cpu_threshold
        self.current_resolution = 'full'
        
        # Resource monitoring
        self.resource_monitor = threading.Timer(1.0, self._monitor_resources)
        self.resource_state = EdgeResourceState(0.0, 0.0, 0.0)
        
        # Performance metrics
        self.metrics = {
            'processing_latency': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'embeddings_processed': 0,
            'transmissions_sent': 0
        }
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 4
        self.accumulated_gradients = 0
        
        self.logger.info("EdgeModule initialized")
        
    async def start(self):
        """Start the edge module"""
        self.is_running = True
        self.resource_monitor.start()
        self.logger.info("EdgeModule started")
        
    async def stop(self):
        """Stop the edge module"""
        self.is_running = False
        if self.resource_monitor.is_alive():
            self.resource_monitor.cancel()
        self.logger.info("EdgeModule stopped")
    
    async def process(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process audio spectrogram with full edge processing
        
        Args:
            spectrogram: Input audio spectrogram tensor
            
        Returns:
            Tuple of (embedding, metadata)
        """
        start_time = time.time()
        
        # Adaptive feature extraction based on resources
        if self.resource_state.cpu_utilization > self.resource_threshold:
            spectrogram = self._reduce_resolution(spectrogram)
            self.current_resolution = 'reduced'
        else:
            self.current_resolution = 'full'
        
        # Move to device
        spectrogram = spectrogram.to(self.device)
        
        # Generate positive pair through augmentation
        augmented_spec = self.augmentations.apply(spectrogram)
        
        # Forward pass
        with torch.no_grad():
            embedding = self.encoder(spectrogram.unsqueeze(0)).squeeze(0)
            embedding_aug = self.encoder(augmented_spec.unsqueeze(0)).squeeze(0)
            embedding_momentum = self.encoder_momentum(spectrogram.unsqueeze(0)).squeeze(0)
        
        # Calculate uncertainties
        uncertainties = self.uncertainty_estimator.calculate_uncertainty(
            embedding, embedding_aug
        )
        
        # Update memory bank with current embedding
        current_time = time.time()
        self.memory_bank.add(embedding, current_time)
        
        # Sample negatives for training
        n_negatives = min(64, self.memory_bank.current_size)
        negatives, neg_weights = self.memory_bank.sample_negatives(
            embedding, n_negatives, current_time
        )
        
        # Compute contrastive loss for training
        if negatives.size(0) > 0:
            loss = self._compute_training_loss(
                embedding, embedding_aug, embedding_momentum, negatives, neg_weights
            )
            
            # Backward pass with gradient accumulation
            loss.backward()
            self.accumulated_gradients += 1
            
            if self.accumulated_gradients >= self.gradient_accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulated_gradients = 0
                
                # Update momentum encoder
                self._update_momentum_encoder(momentum=self.config.momentum)
        
        # Update uncertainty estimator
        with torch.no_grad():
            self.uncertainty_estimator.update_prototypes(embedding.unsqueeze(0))
        
        # Prepare metadata
        metadata = {
            'timestamp': current_time,
            'resolution': self.current_resolution,
            'uncertainty': uncertainties,
            'resource_state': self.resource_state.__dict__,
            'memory_bank_size': self.memory_bank.current_size
        }
        
        # Update metrics
        processing_time = (time.time() - start_time) * 1000  # ms
        self.metrics['processing_latency'].append(processing_time)
        self.metrics['embeddings_processed'] += 1
        
        return embedding.detach(), metadata
    
    async def process_partial(self, spectrogram: torch.Tensor, 
                            split_point: int) -> Dict[str, Any]:
        """
        Process spectrogram up to a specific split point for dynamic splitting
        
        Args:
            spectrogram: Input spectrogram
            split_point: Layer index to split at
            
        Returns:
            Dictionary containing partial results
        """
        start_time = time.time()
        
        # Adaptive resolution based on resources
        if self.resource_state.cpu_utilization > self.resource_threshold:
            spectrogram = self._reduce_resolution(spectrogram)
        
        spectrogram = spectrogram.to(self.device)
        
        # Augmented version for positive pair
        augmented_spec = self.augmentations.apply(spectrogram)
        
        # Partial forward pass
        with torch.no_grad():
            partial_features = self.encoder.forward_partial(
                spectrogram.unsqueeze(0), split_point
            ).squeeze(0)
            partial_features_aug = self.encoder.forward_partial(
                augmented_spec.unsqueeze(0), split_point
            ).squeeze(0)
        
        # Calculate uncertainty at intermediate layer
        # (simplified version - in practice would need proper uncertainty estimation)
        uncertainty = F.mse_loss(partial_features, partial_features_aug).item()
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'partial_features': partial_features.detach(),
            'partial_features_aug': partial_features_aug.detach(),
            'split_point': split_point,
            'uncertainty': {'consistency_uncertainty': uncertainty},
            'processing_time': processing_time,
            'timestamp': time.time(),
            'metadata': {
                'resolution': self.current_resolution,
                'resource_state': self.resource_state.__dict__
            }
        }
    
    async def process_complete(self, partial_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete processing from partial features when server unavailable
        
        Args:
            partial_output: Output from process_partial
            
        Returns:
            Complete processing results
        """
        start_time = time.time()
        
        # Continue forward pass from partial features
        partial_features = partial_output['partial_features']
        split_point = partial_output['split_point']
        
        with torch.no_grad():
            embedding = self.encoder.forward_from_partial(
                partial_features.unsqueeze(0), split_point
            ).squeeze(0)
        
        # Add to memory bank
        current_time = time.time()
        self.memory_bank.add(embedding, current_time)
        
        # Calculate full uncertainties
        if 'partial_features_aug' in partial_output:
            embedding_aug = self.encoder.forward_from_partial(
                partial_output['partial_features_aug'].unsqueeze(0), split_point
            ).squeeze(0)
            uncertainties = self.uncertainty_estimator.calculate_uncertainty(
                embedding, embedding_aug
            )
        else:
            uncertainties = partial_output['uncertainty']
        
        processing_time = (time.time() - start_time) * 1000
        total_time = processing_time + partial_output['processing_time']
        
        return {
            'embedding': embedding.detach(),
            'metadata': {
                'timestamp': current_time,
                'total_processing_time': total_time,
                'uncertainty': uncertainties,
                'split_completed_locally': True,
                **partial_output['metadata']
            }
        }
    
    def _compute_training_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                              momentum_anchor: torch.Tensor, negatives: torch.Tensor,
                              negative_weights: torch.Tensor) -> torch.Tensor:
        """Compute local contrastive loss with consistency regularization"""
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(
            anchor, positive, negatives, negative_weights
        )
        
        # Consistency loss
        consistency_loss = F.mse_loss(anchor, momentum_anchor)
        
        # Combined loss
        total_loss = contrastive_loss + 0.1 * consistency_loss
        
        return total_loss
    
    def _update_momentum_encoder(self, momentum: float = 0.999):
        """Update momentum encoder weights"""
        for param_q, param_k in zip(self.encoder.parameters(), 
                                   self.encoder_momentum.parameters()):
            param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data
    
    def _reduce_resolution(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Reduce spectrogram resolution for resource-constrained processing"""
        # Reduce frequency resolution by averaging adjacent bins
        if spectrogram.dim() == 2:
            freq_dim, time_dim = spectrogram.shape
            new_freq_dim = freq_dim // 2
            
            # Reshape and average adjacent frequency bins
            reduced_spec = spectrogram[:new_freq_dim*2].view(new_freq_dim, 2, time_dim).mean(dim=1)
            
            # Increase stride in time dimension
            reduced_spec = reduced_spec[:, ::2]
            
            return reduced_spec
        return spectrogram
    
    def _monitor_resources(self):
        """Monitor system resources"""
        if not hasattr(self, 'is_running') or not self.is_running:
            return
            
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_memory = memory.available / (1024**3)  # GB
        
        # Update resource state
        self.resource_state = EdgeResourceState(
            cpu_utilization=cpu_percent / 100.0,
            memory_usage=memory_percent / 100.0,
            available_memory=available_memory
        )
        
        # Adjust memory bank size based on available memory
        if available_memory < 1.0:  # Less than 1GB
            new_size = self.config.memory_bank_size_min
        elif available_memory < 2.0:  # Less than 2GB
            new_size = (self.config.memory_bank_size_min + self.config.memory_bank_size_max) // 2
        else:
            new_size = self.config.memory_bank_size_max
        
        self.memory_bank.resize(new_size)
        
        # Update metrics
        self.metrics['memory_usage'].append(memory_percent)
        
        # Schedule next monitoring
        if self.is_running:
            self.resource_monitor = threading.Timer(1.0, self._monitor_resources)
            self.resource_monitor.start()
    
    def get_resource_state(self) -> Dict[str, float]:
        """Get current resource state"""
        return self.resource_state.__dict__
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resource metrics"""
        return {
            'cpu': self.resource_state.cpu_utilization * 100,
            'memory': self.resource_state.memory_usage * 100,
            'available_memory_gb': self.resource_state.available_memory,
            'current_resolution': self.current_resolution,
            'memory_bank_size': self.memory_bank.current_size,
            'embeddings_processed': self.metrics['embeddings_processed']
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        metrics = {}
        
        if self.metrics['processing_latency']:
            metrics['avg_latency_ms'] = np.mean(self.metrics['processing_latency'])
            metrics['std_latency_ms'] = np.std(self.metrics['processing_latency'])
        
        metrics['total_embeddings'] = self.metrics['embeddings_processed']
        metrics['memory_bank_utilization'] = (
            self.memory_bank.current_size / self.memory_bank.max_size
        )
        
        return metrics
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing"""
        return {
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_momentum_state_dict': self.encoder_momentum.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory_bank_embeddings': self.memory_bank.embeddings[:self.memory_bank.current_size],
            'memory_bank_timestamps': self.memory_bank.timestamps[:self.memory_bank.current_size],
            'metrics': dict(self.metrics),
            'resource_state': self.resource_state.__dict__
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint"""
        self.encoder.load_state_dict(state['encoder_state_dict'])
        self.encoder_momentum.load_state_dict(state['encoder_momentum_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Restore memory bank
        current_size = state['memory_bank_embeddings'].size(0)
        self.memory_bank.embeddings[:current_size] = state['memory_bank_embeddings']
        self.memory_bank.timestamps[:current_size] = state['memory_bank_timestamps']
        self.memory_bank.current_size = current_size
        
        # Restore metrics
        for key, value in state['metrics'].items():
            self.metrics[key] = value
        
        self.logger.info("EdgeModule state loaded from checkpoint")
