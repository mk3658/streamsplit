"""
StreamSplit: Framework for Edge Audio Learning with Dynamic Computation Splitting
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum

from .edge_module import EdgeModule
from .server_module import ServerModule
from .dynamic_splitting import DynamicSplittingAgent
from ..utils.audio_processing import AudioProcessor
from ..utils.metrics import MetricsCollector

class StreamSplitMode(Enum):
    EDGE_ONLY = "edge_only"
    SERVER_ONLY = "server_only"
    DYNAMIC_SPLIT = "dynamic_split"

@dataclass
class StreamSplitConfig:
    """Configuration for StreamSplit framework"""
    # Device configuration
    device_id: str = "edge_device_1"
    device_type: str = "raspberry_pi_4b"
    
    # Audio processing parameters
    sample_rate: int = 16000
    window_duration: float = 0.025  # 25ms Hann window
    hop_length: float = 0.010      # 10ms hop
    n_fft: int = 512
    n_mels: int = 128
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    momentum: float = 0.999
    temperature: float = 0.1
    
    # Memory bank parameters
    memory_bank_size_min: int = 64
    memory_bank_size_max: int = 512
    
    # Uncertainty thresholds
    uncertainty_threshold_base: float = 0.5
    transmission_threshold: float = 0.7
    
    # Resource monitoring
    resource_monitor_interval: float = 1.0  # seconds
    cpu_threshold: float = 0.70
    
    # Network simulation parameters
    bandwidth_range: Tuple[float, float] = (0.5, 8.0)  # Mbps
    latency_range: Tuple[float, float] = (50, 200)     # ms
    
    # Split agent parameters
    split_agent_lr: float = 1e-3
    split_reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.split_reward_weights is None:
            self.split_reward_weights = {
                'accuracy': 1.0,
                'resource_usage': -0.5,
                'latency': -0.3,
                'privacy_risk': -0.2
            }

class StreamSplitFramework:
    """
    Main StreamSplit framework coordinating edge and server modules
    with dynamic computation splitting.
    """
    
    def __init__(self, config: StreamSplitConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.audio_processor = AudioProcessor(config)
        self.metrics_collector = MetricsCollector()
        
        # Initialize modules
        self.edge_module = EdgeModule(config)
        self.server_module = ServerModule(config)
        self.splitting_agent = DynamicSplittingAgent(config)
        
        # Framework state
        self.mode = StreamSplitMode.DYNAMIC_SPLIT
        self.is_running = False
        self.current_split_point = None
        
        # Performance tracking
        self.performance_history = {
            'accuracy': [],
            'bandwidth_usage': [],
            'latency': [],
            'energy_consumption': [],
            'resource_utilization': []
        }
        
        self.logger.info(f"StreamSplit framework initialized for device {config.device_id}")
    
    async def start(self, mode: StreamSplitMode = StreamSplitMode.DYNAMIC_SPLIT):
        """Start the StreamSplit framework"""
        self.mode = mode
        self.is_running = True
        
        self.logger.info(f"Starting StreamSplit in {mode.value} mode")
        
        # Start modules based on mode
        if mode in [StreamSplitMode.EDGE_ONLY, StreamSplitMode.DYNAMIC_SPLIT]:
            await self.edge_module.start()
            
        if mode in [StreamSplitMode.SERVER_ONLY, StreamSplitMode.DYNAMIC_SPLIT]:
            await self.server_module.start()
            
        if mode == StreamSplitMode.DYNAMIC_SPLIT:
            await self.splitting_agent.start()
            
        # Start main processing loop
        self._processing_task = asyncio.create_task(self._main_processing_loop())
        self._monitoring_task = asyncio.create_task(self._resource_monitoring_loop())
        
    async def stop(self):
        """Stop the StreamSplit framework"""
        self.is_running = False
        
        # Cancel tasks
        if hasattr(self, '_processing_task'):
            self._processing_task.cancel()
        if hasattr(self, '_monitoring_task'):
            self._monitoring_task.cancel()
            
        # Stop modules
        await self.edge_module.stop()
        await self.server_module.stop()
        await self.splitting_agent.stop()
        
        self.logger.info("StreamSplit framework stopped")
    
    async def process_audio_stream(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Process streaming audio data through the framework
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            Processing results including embeddings and metadata
        """
        # Preprocess audio
        spectrogram = self.audio_processor.extract_spectrogram(audio_data)
        
        if self.mode == StreamSplitMode.EDGE_ONLY:
            return await self._process_edge_only(spectrogram)
        elif self.mode == StreamSplitMode.SERVER_ONLY:
            return await self._process_server_only(spectrogram)
        else:
            return await self._process_dynamic_split(spectrogram)
    
    async def _process_edge_only(self, spectrogram: torch.Tensor) -> Dict[str, Any]:
        """Process audio entirely on edge device"""
        start_time = asyncio.get_event_loop().time()
        
        # Edge processing
        embedding, metadata = await self.edge_module.process(spectrogram)
        
        end_time = asyncio.get_event_loop().time()
        latency = (end_time - start_time) * 1000  # ms
        
        return {
            'embedding': embedding,
            'metadata': metadata,
            'latency': latency,
            'mode': 'edge_only',
            'split_point': None
        }
    
    async def _process_server_only(self, spectrogram: torch.Tensor) -> Dict[str, Any]:
        """Process audio entirely on server"""
        start_time = asyncio.get_event_loop().time()
        
        # Send raw spectrogram to server
        embedding, metadata = await self.server_module.process(spectrogram)
        
        end_time = asyncio.get_event_loop().time()
        latency = (end_time - start_time) * 1000  # ms
        
        return {
            'embedding': embedding,
            'metadata': metadata,
            'latency': latency,
            'mode': 'server_only',
            'split_point': None
        }
    
    async def _process_dynamic_split(self, spectrogram: torch.Tensor) -> Dict[str, Any]:
        """Process audio with dynamic computation splitting"""
        start_time = asyncio.get_event_loop().time()
        
        # Get current split decision
        split_decision = await self.splitting_agent.get_split_decision()
        self.current_split_point = split_decision['split_point']
        
        # Process on edge up to split point
        edge_output = await self.edge_module.process_partial(
            spectrogram, split_point=self.current_split_point
        )
        
        # Determine transmission based on uncertainty
        should_transmit = self._should_transmit(edge_output)
        
        if should_transmit:
            # Continue processing on server
            server_output = await self.server_module.process_continuation(
                edge_output, split_point=self.current_split_point
            )
            final_embedding = server_output['embedding']
            metadata = {**edge_output['metadata'], **server_output['metadata']}
        else:
            # Complete processing on edge
            edge_complete = await self.edge_module.process_complete(edge_output)
            final_embedding = edge_complete['embedding']
            metadata = edge_complete['metadata']
        
        end_time = asyncio.get_event_loop().time()
        latency = (end_time - start_time) * 1000  # ms
        
        # Update metrics
        self._update_performance_metrics(latency, should_transmit, split_decision)
        
        return {
            'embedding': final_embedding,
            'metadata': metadata,
            'latency': latency,
            'mode': 'dynamic_split',
            'split_point': self.current_split_point,
            'transmitted': should_transmit
        }
    
    def _should_transmit(self, edge_output: Dict[str, Any]) -> bool:
        """Determine whether to transmit edge output to server"""
        uncertainty = self._calculate_uncertainty(edge_output)
        network_quality = self.splitting_agent.get_network_quality()
        resource_state = self.edge_module.get_resource_state()
        
        # Adaptive threshold based on network and resources
        threshold = self.config.uncertainty_threshold_base * (
            1 + 0.5 * (1 - network_quality) + 0.3 * resource_state['cpu_utilization']
        )
        
        return uncertainty > threshold
    
    def _calculate_uncertainty(self, output: Dict[str, Any]) -> float:
        """Calculate uncertainty for transmission decision"""
        metadata = output['metadata']
        
        # Weighted combination of uncertainty metrics
        uncertainty = 0.0
        weights = [0.4, 0.3, 0.3]  # consistency, entropy, prototype
        
        if 'consistency_uncertainty' in metadata:
            uncertainty += weights[0] * metadata['consistency_uncertainty']
        if 'entropy_uncertainty' in metadata:
            uncertainty += weights[1] * metadata['entropy_uncertainty']
        if 'prototype_uncertainty' in metadata:
            uncertainty += weights[2] * metadata['prototype_uncertainty']
            
        return uncertainty
    
    async def _main_processing_loop(self):
        """Main processing loop for continuous operation"""
        while self.is_running:
            try:
                # This would be replaced with actual audio stream input
                # For now, we simulate processing interval
                await asyncio.sleep(0.1)
                
                # Update splitting agent with recent performance
                if self.mode == StreamSplitMode.DYNAMIC_SPLIT:
                    await self.splitting_agent.update_performance(
                        self.performance_history
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in main processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _resource_monitoring_loop(self):
        """Monitor device resources and network conditions"""
        while self.is_running:
            try:
                # Collect resource metrics
                resources = self.edge_module.get_resource_metrics()
                network = self.splitting_agent.get_network_metrics()
                
                # Update metrics collector
                self.metrics_collector.update_resources(resources)
                self.metrics_collector.update_network(network)
                
                # Log resource status periodically
                if len(self.performance_history['resource_utilization']) % 10 == 0:
                    self.logger.info(
                        f"Resources - CPU: {resources['cpu']:.1f}%, "
                        f"Memory: {resources['memory']:.1f}%, "
                        f"Network: {network['bandwidth']:.1f}Mbps"
                    )
                
                await asyncio.sleep(self.config.resource_monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(5.0)
    
    def _update_performance_metrics(self, latency: float, transmitted: bool, 
                                   split_decision: Dict[str, Any]):
        """Update performance tracking metrics"""
        # Track bandwidth usage
        bandwidth_used = 0.0
        if transmitted:
            # Estimate bandwidth based on embedding size and metadata
            embedding_size = 128 * 4  # 128D float32 embedding
            metadata_size = 100  # Estimated metadata size in bytes
            bandwidth_used = (embedding_size + metadata_size) / 1024  # KB
        
        # Update history
        self.performance_history['latency'].append(latency)
        self.performance_history['bandwidth_usage'].append(bandwidth_used)
        
        # Keep only recent history (last 1000 samples)
        for key in self.performance_history:
            if len(self.performance_history[key]) > 1000:
                self.performance_history[key] = self.performance_history[key][-1000:]
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of framework performance"""
        summary = {}
        
        # Calculate averages
        for metric, values in self.performance_history.items():
            if values:
                summary[f"avg_{metric}"] = np.mean(values)
                summary[f"std_{metric}"] = np.std(values)
        
        # Calculate efficiency metrics
        if self.performance_history['bandwidth_usage']:
            total_bandwidth = sum(self.performance_history['bandwidth_usage'])
            summary['bandwidth_reduction'] = max(0, 1 - total_bandwidth / 1000)  # Relative to baseline
        
        return summary
    
    def save_checkpoint(self, filepath: str):
        """Save framework state checkpoint"""
        checkpoint = {
            'config': self.config,
            'performance_history': self.performance_history,
            'current_split_point': self.current_split_point,
            'edge_state': self.edge_module.get_state(),
            'server_state': self.server_module.get_state(),
            'splitting_agent_state': self.splitting_agent.get_state()
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load framework state from checkpoint"""
        checkpoint = torch.load(filepath)
        
        self.performance_history = checkpoint['performance_history']
        self.current_split_point = checkpoint['current_split_point']
        
        self.edge_module.load_state(checkpoint['edge_state'])
        self.server_module.load_state(checkpoint['server_state'])
        self.splitting_agent.load_state(checkpoint['splitting_agent_state'])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")

# Factory function for easy initialization
def create_streamsplit(config_dict: Dict[str, Any]) -> StreamSplitFramework:
    """Create StreamSplit framework from configuration dictionary"""
    config = StreamSplitConfig(**config_dict)
    return StreamSplitFramework(config)
