"""
Server Module for StreamSplit Framework
Implements server-side aggregation and refinement with hybrid SW+Laplacian loss
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from ..models.encoders import MobileNetV3ServerEncoder
from ..models.losses import SlicedWassersteinLoss, LaplacianRegularizationLoss
from ..utils.data_utils import OptimalTransport, KMeansOnline

@dataclass
class PrototypeCenter:
    """Prototype center for uncertainty calculation"""
    center: torch.Tensor
    count: int
    last_updated: float

class HierarchicalAggregator:
    """Hierarchical aggregation for sparse, asynchronous embeddings"""
    
    def __init__(self, embedding_dim: int = 128, kernel_bandwidth: float = 0.1):
        self.embedding_dim = embedding_dim
        self.kernel_bandwidth = kernel_bandwidth
        
        # Store embeddings by device
        self.device_embeddings = defaultdict(list)
        self.device_timestamps = defaultdict(list)
        
        # Global embedding pool for cross-device alignment
        self.global_pool = deque(maxlen=10000)
        self.global_timestamps = deque(maxlen=10000)
        
    def add_device_embedding(self, device_id: str, embedding: torch.Tensor, 
                           timestamp: float):
        """Add embedding from specific device"""
        self.device_embeddings[device_id].append(embedding.detach().cpu())
        self.device_timestamps[device_id].append(timestamp)
        
        # Also add to global pool
        self.global_pool.append(embedding.detach().cpu())
        self.global_timestamps.append(timestamp)
        
        # Limit per-device storage
        max_per_device = 1000
        if len(self.device_embeddings[device_id]) > max_per_device:
            self.device_embeddings[device_id] = self.device_embeddings[device_id][-max_per_device:]
            self.device_timestamps[device_id] = self.device_timestamps[device_id][-max_per_device:]
    
    def estimate_device_distribution(self, device_id: str, 
                                   recent_window: int = 100) -> torch.Tensor:
        """Estimate distribution for a specific device using Gaussian kernel"""
        if device_id not in self.device_embeddings:
            return None
            
        embeddings = self.device_embeddings[device_id][-recent_window:]
        if len(embeddings) < 5:
            return None
            
        # Convert to tensor
        embeddings = torch.stack(embeddings)
        
        # Create kernel density estimate
        # For efficiency, we'll use a simplified approach with representative points
        n_representatives = min(50, len(embeddings))
        indices = torch.linspace(0, len(embeddings)-1, n_representatives).long()
        representatives = embeddings[indices]
        
        return representatives
    
    def get_global_distribution(self, n_samples: int = 500) -> torch.Tensor:
        """Get representative samples from global distribution"""
        if len(self.global_pool) < n_samples:
            return torch.stack(list(self.global_pool))
        
        # Sample uniformly from recent global pool
        indices = torch.randperm(len(self.global_pool))[:n_samples]
        samples = [self.global_pool[i] for i in indices]
        
        return torch.stack(samples)

class SelectiveTransmissionPolicy:
    """Manages selective transmission based on uncertainty and network conditions"""
    
    def __init__(self, config):
        self.config = config
        self.uncertainty_weights = {
            'consistency': 0.4,
            'entropy': 0.3,
            'prototype': 0.3
        }
        
        # Prototype centers for uncertainty calculation
        self.prototype_centers = {}
        self.n_prototypes = 10
        
        # Adaptive threshold parameters
        self.base_threshold = config.uncertainty_threshold_base
        self.threshold_adaptation_rate = 0.01
        
        # Network condition tracking
        self.network_history = deque(maxlen=100)
        
    def update_prototypes(self, embeddings: torch.Tensor, device_id: str):
        """Update prototype centers using online k-means"""
        if device_id not in self.prototype_centers:
            # Initialize prototypes
            self.prototype_centers[device_id] = KMeansOnline(
                self.n_prototypes, embeddings.shape[1]
            )
        
        self.prototype_centers[device_id].update(embeddings)
    
    def calculate_uncertainty(self, embedding_data: Dict[str, Any]) -> float:
        """Calculate weighted uncertainty score"""
        uncertainties = embedding_data.get('uncertainty', {})
        
        total_uncertainty = 0.0
        total_weight = 0.0
        
        # Weight different uncertainty components
        for uncertainty_type, weight in self.uncertainty_weights.items():
            key = f"{uncertainty_type}_uncertainty"
            if key in uncertainties:
                total_uncertainty += weight * uncertainties[key]
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_uncertainty /= total_weight
        
        return total_uncertainty
    
    def should_transmit(self, embedding_data: Dict[str, Any], 
                       network_state: Dict[str, float]) -> bool:
        """Determine if embedding should be transmitted to server"""
        uncertainty = self.calculate_uncertainty(embedding_data)
        
        # Adaptive threshold based on network conditions
        bandwidth_factor = min(1.0, network_state.get('bandwidth', 1.0) / 2.0)  # Normalize to 2Mbps
        latency_factor = max(0.1, 1.0 - network_state.get('latency', 100) / 500)  # Normalize to 500ms
        
        adaptive_threshold = self.base_threshold * (2.0 - bandwidth_factor * latency_factor)
        
        # Update network history for adaptive learning
        self.network_history.append({
            'uncertainty': uncertainty,
            'transmitted': uncertainty > adaptive_threshold,
            'bandwidth': network_state.get('bandwidth', 0),
            'latency': network_state.get('latency', 0)
        })
        
        return uncertainty > adaptive_threshold
    
    def get_transmission_decision(self, embedding_data: Dict[str, Any],
                                network_state: Dict[str, float]) -> Dict[str, Any]:
        """Get detailed transmission decision"""
        uncertainty = self.calculate_uncertainty(embedding_data)
        should_transmit = self.should_transmit(embedding_data, network_state)
        
        decision = {
            'should_transmit': should_transmit,
            'uncertainty_score': uncertainty,
            'threshold_used': self.base_threshold,
            'network_state': network_state,
            'reasoning': {
                'uncertainty_breakdown': embedding_data.get('uncertainty', {}),
                'network_quality': network_state,
                'decision_factors': {
                    'bandwidth': network_state.get('bandwidth', 0),
                    'latency': network_state.get('latency', 0),
                    'uncertainty': uncertainty
                }
            }
        }
        
        return decision

class ServerModule:
    """
    Server module implementing hierarchical aggregation and hybrid loss refinement
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize server encoder (completes edge encoder)
        self.encoder = MobileNetV3ServerEncoder(
            intermediate_dim=256,  # From edge encoder
            embedding_dim=128,
            num_layers=4
        ).to(self.device)
        
        # Hierarchical aggregator
        self.aggregator = HierarchicalAggregator(embedding_dim=128)
        
        # Selective transmission policy
        self.transmission_policy = SelectiveTransmissionPolicy(config)
        
        # Hybrid loss components
        self.sw_loss = SlicedWassersteinLoss(
            num_projections=100,
            device=self.device
        )
        self.laplacian_loss = LaplacianRegularizationLoss(
            k_neighbors=5,
            device=self.device
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.learning_rate * 5,  # Higher LR for server
            weight_decay=1e-5
        )
        
        # Prototype maintenance
        self.prototype_centers = {}
        self.prototype_update_rate = 0.01
        
        # Performance tracking
        self.metrics = {
            'embeddings_received': 0,
            'embeddings_processed': 0,
            'loss_values': deque(maxlen=1000),
            'sw_loss_values': deque(maxlen=1000),
            'laplacian_loss_values': deque(maxlen=1000),
            'device_transmissions': defaultdict(int),
            'processing_latency': deque(maxlen=100)
        }
        
        # Batch processing
        self.batch_size = 256
        self.pending_embeddings = []
        self.pending_metadata = []
        self.last_batch_time = time.time()
        self.batch_timeout = 5.0  # seconds
        
        # Asynchronous processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_queue = asyncio.Queue(maxsize=1000)
        
        self.logger.info("ServerModule initialized")
    
    async def start(self):
        """Start the server module"""
        self.is_running = True
        
        # Start background processing tasks
        asyncio.create_task(self._batch_processing_loop())
        asyncio.create_task(self._refinement_loop())
        
        self.logger.info("ServerModule started")
    
    async def stop(self):
        """Stop the server module"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("ServerModule stopped")
    
    async def process(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process full spectrogram on server (server-only mode)
        
        Args:
            spectrogram: Input spectrogram tensor
            
        Returns:
            Tuple of (embedding, metadata)
        """
        start_time = time.time()
        
        # Move to device
        spectrogram = spectrogram.to(self.device)
        
        # Full forward pass through complete model
        with torch.no_grad():
            # For server-only mode, we'd use a complete model
            # Here we simulate by processing through full pipeline
            embedding = self.encoder.full_forward(spectrogram.unsqueeze(0)).squeeze(0)
        
        # Add to aggregator
        device_id = "server_direct"
        current_time = time.time()
        self.aggregator.add_device_embedding(device_id, embedding, current_time)
        
        # Update prototypes
        self.transmission_policy.update_prototypes(
            embedding.unsqueeze(0), device_id
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        metadata = {
            'processing_time': processing_time,
            'mode': 'server_only',
            'timestamp': current_time,
            'device_id': device_id
        }
        
        self.metrics['embeddings_processed'] += 1
        self.metrics['processing_latency'].append(processing_time)
        
        return embedding.detach(), metadata
    
    async def process_continuation(self, edge_output: Dict[str, Any],
                                 split_point: int) -> Dict[str, Any]:
        """
        Continue processing from edge partial output
        
        Args:
            edge_output: Partial output from edge device
            split_point: Layer where split occurred
            
        Returns:
            Server processing results
        """
        start_time = time.time()
        
        # Extract partial features
        partial_features = edge_output['partial_features'].to(self.device)
        device_id = edge_output.get('device_id', 'edge_device')
        
        # Continue forward pass from split point
        with torch.no_grad():
            embedding = self.encoder.forward_from_split(
                partial_features.unsqueeze(0), split_point
            ).squeeze(0)
        
        # Add to processing queue for refinement
        await self.processing_queue.put({
            'embedding': embedding,
            'device_id': device_id,
            'metadata': edge_output.get('metadata', {}),
            'split_point': split_point
        })
        
        # Add to aggregator
        current_time = time.time()
        self.aggregator.add_device_embedding(device_id, embedding, current_time)
        
        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        self.metrics['embeddings_received'] += 1
        self.metrics['device_transmissions'][device_id] += 1
        
        return {
            'embedding': embedding.detach(),
            'metadata': {
                'server_processing_time': processing_time,
                'split_point': split_point,
                'device_id': device_id,
                'timestamp': current_time
            }
        }
    
    async def receive_edge_embedding(self, embedding_data: Dict[str, Any]) -> bool:
        """
        Receive and process embedding from edge device
        
        Args:
            embedding_data: Dictionary containing embedding and metadata
            
        Returns:
            Success status
        """
        try:
            device_id = embedding_data.get('device_id', 'unknown')
            embedding = embedding_data['embedding'].to(self.device)
            
            # Add to aggregator
            current_time = time.time()
            self.aggregator.add_device_embedding(device_id, embedding, current_time)
            
            # Update prototypes
            self.transmission_policy.update_prototypes(
                embedding.unsqueeze(0), device_id
            )
            
            # Add to processing queue
            await self.processing_queue.put(embedding_data)
            
            self.metrics['embeddings_received'] += 1
            self.metrics['device_transmissions'][device_id] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error receiving edge embedding: {e}")
            return False
    
    async def _batch_processing_loop(self):
        """Main loop for batch processing of embeddings"""
        while self.is_running:
            try:
                # Collect embeddings for batch processing
                batch_embeddings = []
                batch_metadata = []
                
                # Wait for embeddings or timeout
                try:
                    # Get first embedding with timeout
                    embedding_data = await asyncio.wait_for(
                        self.processing_queue.get(), timeout=self.batch_timeout
                    )
                    batch_embeddings.append(embedding_data['embedding'])
                    batch_metadata.append(embedding_data.get('metadata', {}))
                    
                    # Collect more embeddings up to batch size
                    for _ in range(self.batch_size - 1):
                        try:
                            embedding_data = await asyncio.wait_for(
                                self.processing_queue.get(), timeout=0.1
                            )
                            batch_embeddings.append(embedding_data['embedding'])
                            batch_metadata.append(embedding_data.get('metadata', {}))
                        except asyncio.TimeoutError:
                            break
                    
                    # Process batch if we have embeddings
                    if batch_embeddings:
                        await self._process_batch(batch_embeddings, batch_metadata)
                        
                except asyncio.TimeoutError:
                    # No embeddings received, continue
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, embeddings: List[torch.Tensor], 
                           metadata_list: List[Dict[str, Any]]):
        """Process a batch of embeddings with hybrid loss"""
        if not embeddings:
            return
            
        start_time = time.time()
        
        # Stack embeddings into batch
        batch_embeddings = torch.stack(embeddings)
        
        # Group embeddings by device for distribution alignment
        device_groups = defaultdict(list)
        for i, metadata in enumerate(metadata_list):
            device_id = metadata.get('device_id', 'unknown')
            device_groups[device_id].append(i)
        
        # Calculate hybrid loss
        total_loss = 0.0
        sw_loss_val = 0.0
        laplacian_loss_val = 0.0
        
        # Sliced-Wasserstein loss for distribution alignment
        if len(device_groups) > 1:
            # Get global distribution for comparison
            global_dist = self.aggregator.get_global_distribution(500)
            
            for device_id, indices in device_groups.items():
                if len(indices) > 1:
                    device_embeddings = batch_embeddings[indices]
                    
                    # Calculate SW distance to global distribution
                    sw_loss = self.sw_loss(
                        device_embeddings, 
                        global_dist.to(self.device)
                    )
                    sw_loss_val += sw_loss
                    total_loss += sw_loss
        
        # Laplacian regularization for local structure preservation
        laplacian_loss = self.laplacian_loss(batch_embeddings)
        laplacian_loss_val += laplacian_loss
        total_loss += 0.5 * laplacian_loss  # Weight λ = 0.5
        
        # Backward pass
        if total_loss > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
        
        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        self.metrics['loss_values'].append(total_loss.item() if total_loss > 0 else 0.0)
        self.metrics['sw_loss_values'].append(sw_loss_val.item() if isinstance(sw_loss_val, torch.Tensor) else sw_loss_val)
        self.metrics['laplacian_loss_values'].append(laplacian_loss_val.item())
        self.metrics['embeddings_processed'] += len(embeddings)
        self.metrics['processing_latency'].append(processing_time)
        
        if len(self.metrics['loss_values']) % 100 == 0:
            avg_loss = np.mean(list(self.metrics['loss_values'])[-100:])
            self.logger.info(f"Processed batch of {len(embeddings)} embeddings, avg loss: {avg_loss:.4f}")
    
    async def _refinement_loop(self):
        """Periodic refinement of global model"""
        while self.is_running:
            try:
                # Perform global refinement every 60 seconds
                await asyncio.sleep(60.0)
                
                # Get representative samples from all devices
                all_embeddings = []
                for device_id in self.aggregator.device_embeddings.keys():
                    device_dist = self.aggregator.estimate_device_distribution(device_id)
                    if device_dist is not None:
                        all_embeddings.append(device_dist)
                
                if all_embeddings:
                    # Combine all embeddings
                    combined_embeddings = torch.cat(all_embeddings, dim=0)
                    
                    # Perform refinement with global loss
                    await self._global_refinement(combined_embeddings)
                    
            except Exception as e:
                self.logger.error(f"Error in refinement loop: {e}")
    
    async def _global_refinement(self, embeddings: torch.Tensor):
        """Perform global model refinement"""
        if embeddings.size(0) < 10:
            return
            
        embeddings = embeddings.to(self.device)
        
        # Calculate global Laplacian regularization
        laplacian_loss = self.laplacian_loss(embeddings)
        
        # Optimize global structure
        self.optimizer.zero_grad()
        (0.1 * laplacian_loss).backward()
        self.optimizer.step()
        
        self.logger.info(f"Global refinement completed with {embeddings.size(0)} embeddings")
    
    def get_transmission_decision(self, embedding_data: Dict[str, Any],
                                network_state: Dict[str, float]) -> Dict[str, Any]:
        """Get transmission decision for edge device"""
        return self.transmission_policy.get_transmission_decision(
            embedding_data, network_state
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'embeddings_received': self.metrics['embeddings_received'],
            'embeddings_processed': self.metrics['embeddings_processed'],
            'device_transmissions': dict(self.metrics['device_transmissions'])
        }
        
        # Calculate loss statistics
        if self.metrics['loss_values']:
            metrics['avg_loss'] = np.mean(self.metrics['loss_values'])
            metrics['std_loss'] = np.std(self.metrics['loss_values'])
        
        if self.metrics['sw_loss_values']:
            metrics['avg_sw_loss'] = np.mean(self.metrics['sw_loss_values'])
        
        if self.metrics['laplacian_loss_values']:
            metrics['avg_laplacian_loss'] = np.mean(self.metrics['laplacian_loss_values'])
        
        if self.metrics['processing_latency']:
            metrics['avg_processing_latency'] = np.mean(self.metrics['processing_latency'])
        
        # Distribution alignment metrics
        metrics['active_devices'] = len(self.aggregator.device_embeddings)
        metrics['global_pool_size'] = len(self.aggregator.global_pool)
        
        return metrics
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """Get statistics per device"""
        device_stats = {}
        
        for device_id in self.aggregator.device_embeddings.keys():
            device_stats[device_id] = {
                'embeddings_count': len(self.aggregator.device_embeddings[device_id]),
                'transmissions': self.metrics['device_transmissions'][device_id],
                'last_seen': (
                    max(self.aggregator.device_timestamps[device_id])
                    if self.aggregator.device_timestamps[device_id] else 0
                )
            }
        
        return device_stats
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing"""
        return {
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': {k: list(v) if hasattr(v, '__iter__') else v 
                       for k, v in self.metrics.items()},
            'prototype_centers': self.prototype_centers,
            'aggregator_state': {
                'device_embeddings': dict(self.aggregator.device_embeddings),
                'device_timestamps': dict(self.aggregator.device_timestamps)
            }
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint"""
        self.encoder.load_state_dict(state['encoder_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Restore metrics
        for key, value in state['metrics'].items():
            if key in self.metrics:
                if hasattr(self.metrics[key], 'extend'):
                    self.metrics[key].extend(value)
                else:
                    self.metrics[key] = value
        
        # Restore aggregator state
        if 'aggregator_state' in state:
            aggregator_state = state['aggregator_state']
            self.aggregator.device_embeddings = defaultdict(list, aggregator_state['device_embeddings'])
            self.aggregator.device_timestamps = defaultdict(list, aggregator_state['device_timestamps'])
        
        self.logger.info("ServerModule state loaded from checkpoint")
