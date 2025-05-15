"""
Server Trainer for StreamSplit Framework
Implements server-side training with hybrid SW+Laplacian loss and distribution aggregation
Based on Section 3.2 of the StreamSplit paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
import asyncio

from ..models.encoders import MobileNetV3ServerEncoder
from ..models.losses import HybridSWLaplacianLoss, SlicedWassersteinLoss, LaplacianRegularizationLoss
from ..utils.data_utils import OptimalTransport, KMeansOnline, EmbeddingStore
from ..utils.metrics import MetricsCollector


@dataclass
class ServerTrainingConfig:
    """Configuration for server trainer"""
    learning_rate: float = 5e-4
    batch_size: int = 256
    update_frequency: int = 10  # Updates per aggregation cycle
    temporal_window: float = 30.0  # seconds
    
    # Hybrid loss parameters
    sw_projections: int = 100
    sw_weight: float = 1.0
    laplacian_k_neighbors: int = 5
    laplacian_weight: float = 0.5
    laplacian_sigma: float = 1.0
    
    # Aggregation parameters
    min_devices_per_update: int = 2
    max_age_threshold: float = 60.0  # seconds
    prototype_update_rate: float = 0.01
    
    # Optimization parameters
    scheduler_type: str = "cosine"
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 1.0
    
    # Splitting agent parameters
    splitting_reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.splitting_reward_weights is None:
            self.splitting_reward_weights = {
                'accuracy': 0.3,
                'resource': 0.2,
                'latency': 0.2,
                'privacy': 0.3
            }


class DistributionEstimator:
    """Estimates device distributions using hierarchical aggregation (Section 3.2.2)"""
    
    def __init__(self, embedding_dim: int = 128, bandwidth_sigma: float = 1.0):
        self.embedding_dim = embedding_dim
        self.bandwidth_sigma = bandwidth_sigma
        self.device_distributions = {}
        self.global_distribution = None
        self.logger = logging.getLogger(__name__)
        
    def update_device_distribution(self, device_id: str, embeddings: torch.Tensor,
                                 timestamps: torch.Tensor):
        """Update distribution estimate for a specific device (Equation 13)"""
        if embeddings.size(0) == 0:
            return
        
        # Create Gaussian kernel density estimate
        # K(e, e') = exp(-||e - e'||^2 / 2σ^2)
        n_embeddings = embeddings.size(0)
        
        # Store representative embeddings for efficient computation
        if n_embeddings > 500:  # Subsample if too many embeddings
            indices = torch.randperm(n_embeddings)[:500]
            embeddings = embeddings[indices]
            timestamps = timestamps[indices]
        
        # Weight by recency (more recent embeddings get higher weight)
        current_time = time.time()
        ages = current_time - timestamps
        weights = torch.exp(-ages / 30.0)  # 30 second decay
        weights = weights / weights.sum()
        
        self.device_distributions[device_id] = {
            'embeddings': embeddings.detach(),
            'weights': weights.detach(),
            'timestamp': current_time,
            'count': n_embeddings
        }
        
        self.logger.debug(f"Updated distribution for device {device_id} with {n_embeddings} embeddings")
    
    def get_device_distribution(self, device_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get distribution estimate for a device"""
        return self.device_distributions.get(device_id)
    
    def estimate_global_distribution(self, n_samples: int = 1000) -> torch.Tensor:
        """Estimate global distribution by combining device distributions"""
        if not self.device_distributions:
            return torch.empty(0, self.embedding_dim)
        
        # Collect embeddings from all devices
        all_embeddings = []
        all_weights = []
        
        for device_id, dist_data in self.device_distributions.items():
            embeddings = dist_data['embeddings']
            weights = dist_data['weights']
            
            # Weight by device contribution and recency
            device_weight = 1.0 / len(self.device_distributions)
            scaled_weights = weights * device_weight
            
            all_embeddings.append(embeddings)
            all_weights.append(scaled_weights)
        
        if not all_embeddings:
            return torch.empty(0, self.embedding_dim)
        
        # Combine embeddings and weights
        combined_embeddings = torch.cat(all_embeddings, dim=0)
        combined_weights = torch.cat(all_weights, dim=0)
        
        # Sample according to weights
        if combined_embeddings.size(0) > n_samples:
            indices = torch.multinomial(combined_weights, n_samples, replacement=False)
            sampled_embeddings = combined_embeddings[indices]
        else:
            sampled_embeddings = combined_embeddings
        
        self.global_distribution = sampled_embeddings
        return sampled_embeddings
    
    def clean_old_distributions(self, max_age: float = 300.0):
        """Remove old device distributions"""
        current_time = time.time()
        devices_to_remove = []
        
        for device_id, dist_data in self.device_distributions.items():
            if current_time - dist_data['timestamp'] > max_age:
                devices_to_remove.append(device_id)
        
        for device_id in devices_to_remove:
            del self.device_distributions[device_id]
            self.logger.info(f"Removed old distribution for device {device_id}")


class PrototypeMaintenance:
    """Maintains prototype centers for uncertainty calculation (Appendix I)"""
    
    def __init__(self, n_prototypes: int = 10, embedding_dim: int = 128,
                 learning_rate: float = 0.01):
        self.n_prototypes = n_prototypes
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Initialize prototypes randomly
        self.prototypes = torch.randn(n_prototypes, embedding_dim)
        self.prototype_counts = torch.zeros(n_prototypes)
        self.last_update = time.time()
        self.logger = logging.getLogger(__name__)
    
    def update_prototypes(self, embeddings: torch.Tensor):
        """Update prototype centers using online k-means (Equations 29-30)"""
        if embeddings.size(0) == 0:
            return
        
        for embedding in embeddings:
            # Find closest prototype (Equation 29)
            distances = torch.norm(self.prototypes - embedding, dim=1)
            closest_idx = torch.argmin(distances)
            
            # Update prototype (Equation 30)
            self.prototype_counts[closest_idx] += 1
            
            # Adaptive learning rate with decay
            adaptive_lr = self.learning_rate / (1 + 0.001 * self.prototype_counts[closest_idx])
            
            self.prototypes[closest_idx] = (
                (1 - adaptive_lr) * self.prototypes[closest_idx] + 
                adaptive_lr * embedding
            )
        
        # Reinitialize rarely used prototypes
        min_count_threshold = 10
        for i in range(self.n_prototypes):
            if self.prototype_counts[i] < min_count_threshold:
                if len(embeddings) > 0:
                    # Reinitialize with a random embedding plus noise
                    idx = torch.randint(0, embeddings.size(0), (1,))
                    self.prototypes[i] = embeddings[idx] + 0.1 * torch.randn_like(embeddings[idx])
                    self.prototype_counts[i] = 1
        
        self.last_update = time.time()
    
    def get_prototype_uncertainty(self, embedding: torch.Tensor) -> float:
        """Calculate prototype-based uncertainty (Equation 11)"""
        distances = torch.norm(self.prototypes - embedding, dim=1)
        min_distance = torch.min(distances)
        return min_distance.item()
    
    def get_prototypes(self) -> torch.Tensor:
        """Get current prototype centers"""
        return self.prototypes.clone()


class ServerTrainer:
    """
    Server trainer implementing hierarchical aggregation and hybrid loss refinement
    Based on Section 3.2 of the StreamSplit paper
    """
    
    def __init__(self, config: ServerTrainingConfig, model: MobileNetV3ServerEncoder, 
                 device: str = 'cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize hybrid loss function
        self.hybrid_loss = HybridSWLaplacianLoss(
            num_projections=config.sw_projections,
            k_neighbors=config.laplacian_k_neighbors,
            lambda_laplacian=config.laplacian_weight,
            sigma_laplacian=config.laplacian_sigma,
            device=device
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        if config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.9
            )
        else:
            self.scheduler = None
        
        # Distribution estimation and prototype maintenance
        self.distribution_estimator = DistributionEstimator(
            embedding_dim=model.embedding_dim
        )
        self.prototype_maintenance = PrototypeMaintenance(
            n_prototypes=10,
            embedding_dim=model.embedding_dim,
            learning_rate=config.prototype_update_rate
        )
        
        # Embedding storage for batch processing
        self.embedding_store = EmbeddingStore(
            max_size=10000,
            embedding_dim=model.embedding_dim
        )
        
        # Training state
        self.global_step = 0
        self.last_aggregation_time = time.time()
        self.device_last_seen = {}
        
        # Performance tracking
        self.training_metrics = {
            'total_updates': 0,
            'total_embeddings_processed': 0,
            'average_loss': 0.0,
            'sw_loss': 0.0,
            'laplacian_loss': 0.0,
            'devices_contributing': 0,
            'learning_rate': config.learning_rate
        }
        
        # Asynchronous processing
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.batch_processing_task = None
        self.is_running = False
        
        self.logger.info(f"ServerTrainer initialized with hybrid loss (SW weight: {config.sw_weight}, "
                        f"Laplacian weight: {config.laplacian_weight})")
    
    async def start_training(self):
        """Start asynchronous training process"""
        self.is_running = True
        self.batch_processing_task = asyncio.create_task(self._batch_processing_loop())
        self.logger.info("Server training started")
    
    async def stop_training(self):
        """Stop training process"""
        self.is_running = False
        if self.batch_processing_task:
            self.batch_processing_task.cancel()
            try:
                await self.batch_processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Server training stopped")
    
    async def receive_embeddings(self, device_id: str, embeddings: Dict[str, Any]):
        """
        Receive embeddings from edge device for processing
        
        Args:
            device_id: ID of the edge device
            embeddings: Dictionary containing embeddings and metadata
        """
        try:
            # Add to processing queue
            await self.processing_queue.put({
                'device_id': device_id,
                'embeddings': embeddings['embeddings'],
                'timestamps': embeddings.get('timestamps', [time.time()]),
                'metadata': embeddings.get('metadata', {}),
                'split_point': embeddings.get('split_point', 0)
            })
            
            # Update device last seen
            self.device_last_seen[device_id] = time.time()
            
            self.training_metrics['total_embeddings_processed'] += len(embeddings['embeddings'])
            
        except Exception as e:
            self.logger.error(f"Error receiving embeddings from {device_id}: {e}")
    
    async def _batch_processing_loop(self):
        """Main loop for batch processing of embeddings"""
        batch_embeddings = []
        batch_metadata = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Collect embeddings for batch
                try:
                    # Wait for embeddings with timeout
                    embedding_data = await asyncio.wait_for(
                        self.processing_queue.get(), timeout=1.0
                    )
                    
                    batch_embeddings.append(embedding_data)
                    batch_metadata.append(embedding_data['metadata'])
                    
                    # Collect more embeddings up to batch size
                    while (len(batch_embeddings) < self.config.batch_size and 
                           not self.processing_queue.empty()):
                        embedding_data = await asyncio.wait_for(
                            self.processing_queue.get(), timeout=0.1
                        )
                        batch_embeddings.append(embedding_data)
                        batch_metadata.append(embedding_data['metadata'])
                        
                except asyncio.TimeoutError:
                    # Process what we have if timeout occurs
                    pass
                
                # Process batch if we have enough embeddings or timeout
                current_time = time.time()
                should_process = (
                    len(batch_embeddings) >= self.config.batch_size or
                    (batch_embeddings and 
                     current_time - last_batch_time > self.config.temporal_window)
                )
                
                if should_process and batch_embeddings:
                    await self._process_batch(batch_embeddings, batch_metadata)
                    batch_embeddings = []
                    batch_metadata = []
                    last_batch_time = current_time
                
                # Periodic cleanup
                if current_time - self.last_aggregation_time > 60.0:
                    self._cleanup_old_data()
                    self.last_aggregation_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, batch_embeddings: List[Dict], 
                            batch_metadata: List[Dict]):
        """Process a batch of embeddings with hybrid loss"""
        if not batch_embeddings:
            return
        
        # Group embeddings by device
        device_groups = defaultdict(list)
        for embedding_data in batch_embeddings:
            device_id = embedding_data['device_id']
            device_groups[device_id].append(embedding_data)
        
        # Update device distributions
        for device_id, device_embeddings in device_groups.items():
            embeddings = torch.stack([
                data['embeddings'] for data in device_embeddings
            ])
            timestamps = torch.tensor([
                data['timestamps'][0] if data['timestamps'] else time.time()
                for data in device_embeddings
            ])
            
            self.distribution_estimator.update_device_distribution(
                device_id, embeddings, timestamps
            )
            
            # Store embeddings for later use
            for data in device_embeddings:
                self.embedding_store.add(
                    data['embeddings'],
                    data['timestamps'][0] if data['timestamps'] else time.time(),
                    device_id,
                    data['metadata']
                )
        
        # Check if we have enough devices for update
        if len(device_groups) < self.config.min_devices_per_update:
            self.logger.debug(f"Not enough devices for update: {len(device_groups)}")
            return
        
        # Perform training update
        await self._training_update(device_groups)
    
    async def _training_update(self, device_groups: Dict[str, List[Dict]]):
        """Perform training update with hybrid loss"""
        # Estimate global distribution
        global_dist = self.distribution_estimator.estimate_global_distribution()
        
        if global_dist.size(0) < 10:
            self.logger.warning("Insufficient global distribution samples")
            return
        
        total_loss = 0.0
        sw_loss_total = 0.0
        laplacian_loss_total = 0.0
        
        self.optimizer.zero_grad()
        
        # Process each device's embeddings
        for device_id, device_embeddings in device_groups.items():
            # Get device distribution
            device_dist = self.distribution_estimator.get_device_distribution(device_id)
            
            if device_dist is None:
                continue
            
            device_embs = device_dist['embeddings'].to(self.device)
            
            # Subsample for efficiency
            if device_embs.size(0) > 100:
                indices = torch.randperm(device_embs.size(0))[:100]
                device_embs = device_embs[indices]
            
            # Compute hybrid loss
            batch_loss, loss_breakdown = self.hybrid_loss(
                device_embs,
                global_dist.to(self.device),
                all_embeddings=global_dist.to(self.device)
            )
            
            # Accumulate losses
            total_loss += batch_loss
            sw_loss_total += loss_breakdown['sw_loss']
            laplacian_loss_total += loss_breakdown['laplacian_loss']
            
            # Update prototypes with device embeddings
            self.prototype_maintenance.update_prototypes(device_embs.cpu())
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
        
        # Update metrics
        self.global_step += 1
        self.training_metrics['total_updates'] += 1
        
        if total_loss > 0:
            self.training_metrics['average_loss'] = (
                0.9 * self.training_metrics['average_loss'] + 
                0.1 * total_loss.item()
            )
            self.training_metrics['sw_loss'] = sw_loss_total.item()
            self.training_metrics['laplacian_loss'] = laplacian_loss_total.item()
        
        self.training_metrics['devices_contributing'] = len(device_groups)
        self.training_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # Log progress
        if self.global_step % 10 == 0:
            self.logger.info(
                f"Step {self.global_step}: Loss={total_loss:.4f}, "
                f"SW={sw_loss_total:.4f}, Lap={laplacian_loss_total:.4f}, "
                f"Devices={len(device_groups)}, LR={self.training_metrics['learning_rate']:.6f}"
            )
    
    def _cleanup_old_data(self):
        """Clean up old data and distributions"""
        current_time = time.time()
        
        # Clean old device distributions
        self.distribution_estimator.clean_old_distributions(
            max_age=self.config.max_age_threshold
        )
        
        # Clean devices that haven't been seen recently
        devices_to_remove = []
        for device_id, last_seen in self.device_last_seen.items():
            if current_time - last_seen > self.config.max_age_threshold:
                devices_to_remove.append(device_id)
        
        for device_id in devices_to_remove:
            del self.device_last_seen[device_id]
            self.logger.info(f"Removed inactive device {device_id}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        return {
            **self.training_metrics,
            'global_step': self.global_step,
            'active_devices': len(self.device_last_seen),
            'global_distribution_size': (
                self.distribution_estimator.global_distribution.size(0)
                if self.distribution_estimator.global_distribution is not None else 0
            ),
            'prototype_centers': self.prototype_maintenance.get_prototypes(),
            'embedding_store_size': self.embedding_store.size()
        }
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """Get statistics for all devices"""
        device_stats = {}
        current_time = time.time()
        
        for device_id, dist_data in self.distribution_estimator.device_distributions.items():
            last_seen = self.device_last_seen.get(device_id, 0)
            device_stats[device_id] = {
                'embeddings_count': dist_data.get('count', 0),
                'last_update': dist_data.get('timestamp', 0),
                'last_seen': last_seen,
                'age_minutes': (current_time - last_seen) / 60.0,
                'is_active': (current_time - last_seen) < 300  # 5 minutes
            }
        
        return device_stats
    
    def evaluate_transmission_uncertainty(self, embedding_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate whether embeddings should be transmitted based on uncertainty
        Implements the multi-component uncertainty calculation (Equations 8-11)
        """
        embedding = embedding_data['embedding']
        metadata = embedding_data.get('metadata', {})
        
        # Calculate different uncertainty components
        uncertainties = {}
        
        # 1. Consistency uncertainty (if available)
        if 'augmented_embedding' in embedding_data:
            consistency_uncertainty = F.mse_loss(
                embedding,
                embedding_data['augmented_embedding']
            ).item()
            uncertainties['consistency'] = consistency_uncertainty
        
        # 2. Entropy uncertainty (simplified as embedding variance)
        entropy_uncertainty = torch.var(embedding).item()
        uncertainties['entropy'] = entropy_uncertainty
        
        # 3. Prototype uncertainty
        prototype_uncertainty = self.prototype_maintenance.get_prototype_uncertainty(embedding)
        uncertainties['prototype'] = prototype_uncertainty
        
        # 4. Combined uncertainty (Equation 8)
        weights = {'consistency': 0.4, 'entropy': 0.3, 'prototype': 0.3}
        
        total_uncertainty = 0.0
        total_weight = 0.0
        
        for uncertainty_type, weight in weights.items():
            if uncertainty_type in uncertainties:
                total_uncertainty += weight * uncertainties[uncertainty_type]
                total_weight += weight
        
        if total_weight > 0:
            total_uncertainty /= total_weight
        
        # Get device ID and determine transmission decision
        device_id = embedding_data.get('device_id', 'unknown')
        
        # Adaptive threshold based on network conditions
        base_threshold = 0.5
        network_factor = 1.0  # Could be adjusted based on network state
        adaptive_threshold = base_threshold * network_factor
        
        should_transmit = total_uncertainty > adaptive_threshold
        
        return {
            'should_transmit': should_transmit,
            'total_uncertainty': total_uncertainty,
            'uncertainty_breakdown': uncertainties,
            'threshold_used': adaptive_threshold,
            'device_id': device_id
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for checkpointing"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'training_metrics': self.training_metrics,
            'prototypes': self.prototype_maintenance.get_prototypes(),
            'device_last_seen': self.device_last_seen
        }
    
    def load_model_state(self, state: Dict[str, Any]):
        """Load model state from checkpoint"""
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if state['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        self.global_step = state['global_step']
        self.training_metrics = state['training_metrics']
        self.device_last_seen = state['device_last_seen']
        
        # Restore prototypes
        if 'prototypes' in state:
            self.prototype_maintenance.prototypes = state['prototypes']
        
        self.logger.info(f"Model state loaded from checkpoint at step {self.global_step}")
    
    def save_checkpoint(self, filepath: str):
        """Save complete training checkpoint"""
        checkpoint = {
            'config': self.config,
            'model_state': self.get_model_state(),
            'distribution_estimator_state': {
                'device_distributions': self.distribution_estimator.device_distributions,
                'global_distribution': self.distribution_estimator.global_distribution
            },
            'embedding_store_state': {
                'embeddings': self.embedding_store.embeddings[:self.embedding_store.current_size],
                'timestamps': self.embedding_store.timestamps[:self.embedding_store.current_size],
                'device_ids': self.embedding_store.device_ids[:self.embedding_store.current_size],
                'current_size': self.embedding_store.current_size
            }
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Server checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load complete training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.load_model_state(checkpoint['model_state'])
        
        # Restore distribution estimator
        if 'distribution_estimator_state' in checkpoint:
            dist_state = checkpoint['distribution_estimator_state']
            self.distribution_estimator.device_distributions = dist_state['device_distributions']
            self.distribution_estimator.global_distribution = dist_state['global_distribution']
        
        # Restore embedding store
        if 'embedding_store_state' in checkpoint:
            store_state = checkpoint['embedding_store_state']
            size = store_state['current_size']
            self.embedding_store.embeddings[:size] = store_state['embeddings']
            self.embedding_store.timestamps[:size] = store_state['timestamps']
            self.embedding_store.device_ids[:size] = store_state['device_ids']
            self.embedding_store.current_size = size
        
        self.logger.info(f"Server checkpoint loaded from {filepath}")


# Utility functions for creating server trainer instances

def create_server_trainer(config_dict: Dict[str, Any],
                         model: MobileNetV3ServerEncoder,
                         device: str = 'cuda') -> ServerTrainer:
    """Factory function to create ServerTrainer with configuration"""
    config = ServerTrainingConfig(**config_dict)
    return ServerTrainer(config, model, device)


def create_trainer_from_checkpoint(checkpoint_path: str,
                                  model: MobileNetV3ServerEncoder,
                                  device: str = 'cuda') -> ServerTrainer:
    """Create server trainer and load from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', ServerTrainingConfig())
    
    trainer = ServerTrainer(config, model, device)
    trainer.load_checkpoint(checkpoint_path)
    return trainer


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..models.encoders import MobileNetV3ServerEncoder
    
    # Create server model
    model = MobileNetV3ServerEncoder(
        intermediate_dim=256,
        embedding_dim=128,
        num_layers=4
    )
    
    # Create trainer configuration
    config = ServerTrainingConfig(
        learning_rate=5e-4,
        batch_size=256,
        sw_projections=100,
        laplacian_k_neighbors=5,
        laplacian_weight=0.5
    )
    
    # Create trainer
    trainer = ServerTrainer(config, model, device='cpu')
    
    async def test_server_training():
        # Start training
        await trainer.start_training()
        
        # Simulate receiving embeddings from edge devices
        for i in range(10):
            device_id = f"device_{i % 3}"
            embeddings = {
                'embeddings': torch.randn(128),
                'timestamps': [time.time()],
                'metadata': {'device_type': 'raspberry_pi', 'version': '1.0'}
            }
            
            await trainer.receive_embeddings(device_id, embeddings)
            await asyncio.sleep(0.1)
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Get metrics
        metrics = trainer.get_training_metrics()
        print("Training Metrics:")
        for key, value in metrics.items():
            if not isinstance(value, torch.Tensor):
                print(f"  {key}: {value}")
        