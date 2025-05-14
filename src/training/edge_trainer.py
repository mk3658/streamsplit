"""
Edge Trainer for StreamSplit Framework
Implements streaming contrastive learning with local memory bank and momentum updates
Based on Section 3.1 of the StreamSplit paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import psutil
import threading
from dataclasses import dataclass

from ..models.encoders import MobileNetV3EdgeEncoder
from ..models.losses import LocalContrastiveLoss
from ..utils.audio_processing import AudioAugmentations
from ..utils.data_utils import DistributionAwareSampling


@dataclass
class TrainingConfig:
    """Configuration for edge trainer"""
    learning_rate: float = 1e-4
    momentum: float = 0.999
    temperature: float = 0.1
    consistency_weight: float = 0.1
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    memory_bank_min_size: int = 64
    memory_bank_max_size: int = 512
    age_decay_factor: float = 0.99
    resource_threshold: float = 0.7
    
    # DAS parameters
    das_components: int = 5
    das_alpha: float = 1.0
    das_learning_rate: float = 0.01


class EdgeMemoryBank:
    """Memory bank for negative sampling with temporal weighting"""
    
    def __init__(self, min_size: int, max_size: int, embedding_dim: int, device: str):
        self.min_size = min_size
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Circular buffer for embeddings
        self.embeddings = torch.zeros((max_size, embedding_dim), device=device)
        self.timestamps = torch.zeros(max_size, device=device)
        self.current_size = 0
        self.write_ptr = 0
        
        # Distribution-Aware Sampling
        self.das = DistributionAwareSampling(embedding_dim, device=device)
        
        # Age weighting parameter
        self.age_decay = 0.99
        
    def add(self, embedding: torch.Tensor, timestamp: float):
        """Add new embedding to memory bank"""
        embedding = embedding.detach()
        
        # Store embedding
        self.embeddings[self.write_ptr] = embedding
        self.timestamps[self.write_ptr] = timestamp
        
        # Update DAS with new embedding
        self.das.update(embedding.unsqueeze(0))
        
        # Update pointers
        if self.current_size < self.max_size:
            self.current_size += 1
        self.write_ptr = (self.write_ptr + 1) % self.max_size
    
    def sample_negatives(self, anchor: torch.Tensor, n_negatives: int, 
                        current_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample negatives using DAS with age weighting"""
        if self.current_size == 0:
            return torch.empty(0, self.embedding_dim, device=self.device), torch.empty(0, device=self.device)
        
        # Get available embeddings
        available_embeddings = self.embeddings[:self.current_size]
        available_timestamps = self.timestamps[:self.current_size]
        
        if self.current_size <= n_negatives:
            # Use all available embeddings
            weights = self._compute_age_weights(available_timestamps, current_time)
            return available_embeddings, weights
        
        # Sample using DAS
        das_probs = self.das.get_sampling_probabilities(available_embeddings)
        
        # Apply age weighting
        age_weights = self._compute_age_weights(available_timestamps, current_time)
        combined_probs = das_probs * age_weights
        combined_probs = combined_probs / combined_probs.sum()
        
        # Sample indices
        indices = torch.multinomial(combined_probs, n_negatives, replacement=False)
        
        return available_embeddings[indices], age_weights[indices]
    
    def _compute_age_weights(self, timestamps: torch.Tensor, current_time: float) -> torch.Tensor:
        """Compute age-based weights for temporal coherence"""
        ages = current_time - timestamps
        weights = torch.pow(self.age_decay, ages)
        return weights / (weights.sum() + 1e-8)
    
    def resize(self, new_size: int):
        """Dynamically resize memory bank based on available memory"""
        new_size = max(self.min_size, min(new_size, self.max_size))
        
        if new_size < self.current_size:
            # Keep most recent embeddings
            if new_size > 0:
                # Sort by timestamp to keep most recent
                sorted_indices = torch.argsort(self.timestamps[:self.current_size], descending=True)
                keep_indices = sorted_indices[:new_size]
                
                # Rearrange embeddings and timestamps
                self.embeddings[:new_size] = self.embeddings[keep_indices]
                self.timestamps[:new_size] = self.timestamps[keep_indices]
                
                self.current_size = new_size
                self.write_ptr = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        return {
            'current_size': self.current_size,
            'max_size': self.max_size,
            'utilization': self.current_size / self.max_size,
            'oldest_timestamp': float(self.timestamps[:self.current_size].min()) if self.current_size > 0 else 0.0,
            'newest_timestamp': float(self.timestamps[:self.current_size].max()) if self.current_size > 0 else 0.0
        }


class EdgeTrainer:
    """
    Edge trainer implementing streaming contrastive learning
    Based on Section 3.1 of the StreamSplit paper
    """
    
    def __init__(self, config: TrainingConfig, model: MobileNetV3EdgeEncoder, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.model = model.to(device)
        
        # Momentum model for consistency loss (Eq. 5)
        self.momentum_model = MobileNetV3EdgeEncoder(
            input_dim=model.embedding_dim,
            embedding_dim=model.embedding_dim
        ).to(device)
        self._copy_weights(self.model, self.momentum_model)
        
        # Loss functions
        self.contrastive_loss = LocalContrastiveLoss(
            temperature=config.temperature,
            device=device
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate scheduler (adaptive based on resources)
        self.base_lr = config.learning_rate
        
        # Memory bank
        self.memory_bank = EdgeMemoryBank(
            min_size=config.memory_bank_min_size,
            max_size=config.memory_bank_max_size,
            embedding_dim=self.model.embedding_dim,
            device=device
        )
        
        # Audio augmentations
        self.augmentations = AudioAugmentations(compute_on_device=True)
        
        # Training state
        self.global_step = 0
        self.gradient_accumulation_counter = 0
        self.epoch = 0
        
        # Performance tracking
        self.loss_history = deque(maxlen=1000)
        self.training_metrics = {
            'total_batches': 0,
            'total_samples': 0,
            'mean_loss': 0.0,
            'std_loss': 0.0,
            'learning_rate': config.learning_rate,
            'memory_bank_utilization': 0.0
        }
        
        # Resource monitoring
        self.resource_monitor_active = True
        self.current_resources = {'cpu': 0.0, 'memory': 0.0}
        self.resource_thread = threading.Thread(target=self._monitor_resources)
        self.resource_thread.daemon = True
        self.resource_thread.start()
        
        self.logger.info(f"EdgeTrainer initialized with batch_size={config.batch_size}")
    
    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target model"""
        target.load_state_dict(source.state_dict())
    
    def _update_momentum_model(self):
        """Update momentum model using EMA (momentum updates)"""
        with torch.no_grad():
            for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()):
                param_k.data = self.config.momentum * param_k.data + (1 - self.config.momentum) * param_q.data
    
    def _monitor_resources(self):
        """Monitor system resources in background thread"""
        while self.resource_monitor_active:
            try:
                # CPU utilization
                cpu_percent = psutil.cpu_percent(interval=1.0)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                self.current_resources = {
                    'cpu': cpu_percent / 100.0,
                    'memory': memory_percent / 100.0,
                    'available_memory_gb': memory.available / (1024**3)
                }
                
                # Adapt memory bank size based on available memory
                self._adapt_memory_bank_size()
                
                # Adapt learning rate based on resources
                self._adapt_learning_rate()
                
            except Exception as e:
                self.logger.warning(f"Resource monitoring error: {e}")
                time.sleep(5.0)
    
    def _adapt_memory_bank_size(self):
        """Adapt memory bank size based on available memory"""
        available_memory = self.current_resources.get('available_memory_gb', 4.0)
        
        # Heuristic: adjust memory bank size based on available memory
        if available_memory < 0.5:  # Less than 500MB
            new_size = self.config.memory_bank_min_size
        elif available_memory < 1.0:  # Less than 1GB
            ratio = 0.5
            new_size = int(self.config.memory_bank_min_size + 
                          ratio * (self.config.memory_bank_max_size - self.config.memory_bank_min_size))
        else:
            new_size = self.config.memory_bank_max_size
        
        self.memory_bank.resize(new_size)
    
    def _adapt_learning_rate(self):
        """Adapt learning rate based on resource utilization (Appendix H)"""
        cpu_utilization = self.current_resources.get('cpu', 0.0)
        
        # Linear scaling strategy from Eq. 28
        scale_factor = 1.0 - cpu_utilization
        scale_factor = max(0.1, scale_factor)  # Minimum 10% of base LR
        
        adaptive_lr = self.base_lr * scale_factor
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adaptive_lr
        
        self.training_metrics['learning_rate'] = adaptive_lr
    
    def _create_positive_pair(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create positive pair using audio augmentations"""
        original = spectrogram
        augmented = self.augmentations.apply(spectrogram)
        return original, augmented
    
    def _extract_embeddings(self, spectrograms: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings using both query and momentum models"""
        batch = torch.stack(spectrograms).to(self.device)
        
        # Query embeddings (trainable)
        self.model.train()
        embeddings = self.model(batch)
        
        # Momentum embeddings (no gradients)
        with torch.no_grad():
            self.momentum_model.eval()
            momentum_embeddings = self.momentum_model(batch)
        
        return embeddings, momentum_embeddings
    
    def _compute_local_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                           momentum_anchor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute local contrastive loss with consistency regularization"""
        current_time = time.time()
        
        # Sample negatives from memory bank
        negatives, negative_weights = self.memory_bank.sample_negatives(
            anchor, n_negatives=min(64, self.memory_bank.current_size), 
            current_time=current_time
        )
        
        # Compute contrastive loss (Eq. 4)
        contrastive_loss = 0.0
        if negatives.size(0) > 0:
            contrastive_loss = self.contrastive_loss(anchor, positive, negatives, negative_weights)
        else:
            # Fallback: use simple contrastive loss without negatives
            temp = self.config.temperature
            contrastive_loss = -F.log_softmax(torch.stack([
                F.cosine_similarity(anchor, positive, dim=0) / temp,
                torch.tensor(0.0, device=self.device)
            ]), dim=0)[0]
        
        # Consistency loss (Eq. 5)
        consistency_loss = F.mse_loss(anchor, momentum_anchor)
        
        # Combined loss (Eq. 6)
        total_loss = contrastive_loss + self.config.consistency_weight * consistency_loss
        
        loss_breakdown = {
            'contrastive_loss': contrastive_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item(),
            'negatives_used': negatives.size(0),
            'memory_bank_size': self.memory_bank.current_size
        }
        
        return total_loss, loss_breakdown
    
    def train_step(self, spectrogram: torch.Tensor) -> Dict[str, Any]:
        """Single training step with a spectrogram"""
        # Create positive pair
        original, augmented = self._create_positive_pair(spectrogram)
        
        # Extract embeddings
        embeddings, momentum_embeddings = self._extract_embeddings([original, augmented])
        anchor, positive = embeddings[0], embeddings[1]
        momentum_anchor = momentum_embeddings[0]
        
        # Compute loss
        loss, loss_breakdown = self._compute_local_loss(anchor, positive, momentum_anchor)
        
        # Gradient accumulation (Appendix G)
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        self.gradient_accumulation_counter += 1
        
        # Update weights when accumulation is complete
        if self.gradient_accumulation_counter >= self.config.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update momentum model
            self._update_momentum_model()
            
            # Reset accumulation counter
            self.gradient_accumulation_counter = 0
            self.global_step += 1
        
        # Add anchor to memory bank
        current_time = time.time()
        self.memory_bank.add(anchor.detach(), current_time)
        
        # Update metrics
        self.loss_history.append(loss_breakdown['total_loss'])
        self.training_metrics['total_batches'] += 1
        self.training_metrics['total_samples'] += 1
        
        if len(self.loss_history) > 0:
            self.training_metrics['mean_loss'] = np.mean(self.loss_history)
            self.training_metrics['std_loss'] = np.std(self.loss_history)
        
        # Update memory bank utilization
        self.training_metrics['memory_bank_utilization'] = self.memory_bank.current_size / self.memory_bank.max_size
        
        return {
            'loss': loss_breakdown,
            'embeddings': {
                'anchor': anchor.detach(),
                'positive': positive.detach(),
                'momentum_anchor': momentum_anchor.detach()
            },
            'memory_bank_stats': self.memory_bank.get_stats(),
            'resources': self.current_resources.copy(),
            'global_step': self.global_step
        }
    
    def train_batch(self, spectrograms: List[torch.Tensor]) -> Dict[str, Any]:
        """Train on a batch of spectrograms"""
        batch_results = []
        total_loss = 0.0
        
        for spectrogram in spectrograms:
            result = self.train_step(spectrogram)
            batch_results.append(result)
            total_loss += result['loss']['total_loss']
        
        # Aggregate batch statistics
        batch_stats = {
            'batch_size': len(spectrograms),
            'total_loss': total_loss,
            'average_loss': total_loss / len(spectrograms),
            'memory_bank_utilization': self.memory_bank.current_size / self.memory_bank.max_size,
            'learning_rate': self.training_metrics['learning_rate'],
            'global_step': self.global_step
        }
        
        # Resource adaptation check
        cpu_util = self.current_resources.get('cpu', 0.0)
        if cpu_util > self.config.resource_threshold:
            batch_stats['resource_warning'] = f"High CPU utilization: {cpu_util:.1%}"
        
        return {
            'batch_stats': batch_stats,
            'individual_results': batch_results
        }
    
    def process_streaming_audio(self, spectrogram: torch.Tensor) -> Dict[str, Any]:
        """Process streaming audio with optional resolution adaptation"""
        # Check if we should use reduced resolution
        cpu_util = self.current_resources.get('cpu', 0.0)
        use_reduced = cpu_util > self.config.resource_threshold
        
        if use_reduced:
            # Apply reduced resolution processing if needed
            # This could involve downsampling the spectrogram
            pass
        
        # Perform training step
        result = self.train_step(spectrogram)
        
        # Add processing metadata
        result['processing_metadata'] = {
            'reduced_resolution': use_reduced,
            'cpu_utilization': cpu_util,
            'timestamp': time.time()
        }
        
        return result
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        return {
            **self.training_metrics,
            'memory_bank_stats': self.memory_bank.get_stats(),
            'current_resources': self.current_resources.copy(),
            'global_step': self.global_step,
            'gradient_accumulation_counter': self.gradient_accumulation_counter,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'momentum_model_state_dict': self.momentum_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'loss_history': list(self.loss_history),
            'memory_bank_embeddings': self.memory_bank.embeddings[:self.memory_bank.current_size],
            'memory_bank_timestamps': self.memory_bank.timestamps[:self.memory_bank.current_size],
            'memory_bank_size': self.memory_bank.current_size
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.momentum_model.load_state_dict(checkpoint['momentum_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.training_metrics = checkpoint['training_metrics']
        self.loss_history = deque(checkpoint['loss_history'], maxlen=1000)
        
        # Restore memory bank
        if 'memory_bank_embeddings' in checkpoint:
            size = checkpoint['memory_bank_size']
            self.memory_bank.embeddings[:size] = checkpoint['memory_bank_embeddings']
            self.memory_bank.timestamps[:size] = checkpoint['memory_bank_timestamps']
            self.memory_bank.current_size = size
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def set_training_mode(self, mode: bool):
        """Set training mode for models"""
        self.model.train(mode)
        # Momentum model is always in eval mode
        self.momentum_model.eval()
    
    def eval_mode(self):
        """Set models to evaluation mode"""
        self.set_training_mode(False)
    
    def train_mode(self):
        """Set models to training mode"""
        self.set_training_mode(True)
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def shutdown(self):
        """Clean shutdown of trainer"""
        self.resource_monitor_active = False
        if self.resource_thread.is_alive():
            self.resource_thread.join(timeout=2.0)
        self.logger.info("EdgeTrainer shutdown complete")
    
    def __del__(self):
        """Destructor to ensure clean shutdown"""
        try:
            self.shutdown()
        except:
            pass


# Utility functions for creating trainer instances

def create_edge_trainer(config_dict: Dict[str, Any], 
                       model: MobileNetV3EdgeEncoder, 
                       device: str = 'cpu') -> EdgeTrainer:
    """Factory function to create EdgeTrainer with configuration"""
    config = TrainingConfig(**config_dict)
    return EdgeTrainer(config, model, device)


def create_trainer_from_checkpoint(checkpoint_path: str, 
                                 model: MobileNetV3EdgeEncoder,
                                 device: str = 'cpu') -> EdgeTrainer:
    """Create trainer and load from checkpoint"""
    # Create trainer with default config (will be overridden by checkpoint)
    trainer = EdgeTrainer(TrainingConfig(), model, device)
    trainer.load_checkpoint(checkpoint_path)
    return trainer


# Example usage
if __name__ == "__main__":
    from ..models.encoders import MobileNetV3EdgeEncoder
    
    # Create model
    model = MobileNetV3EdgeEncoder(input_dim=128, embedding_dim=128)
    
    # Create trainer
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        memory_bank_max_size=512
    )
    trainer = EdgeTrainer(config, model, device='cpu')
    
    # Example training step
    spectrogram = torch.randn(128, 64)  # Example spectrogram
    result = trainer.train_step(spectrogram)
    
    print("Training step completed!")
    print(f"Loss: {result['loss']['total_loss']:.4f}")
    print(f"Memory bank size: {result['memory_bank_stats']['current_size']}")
    
    trainer.shutdown()