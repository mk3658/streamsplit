"""
Distributed Training Coordination for StreamSplit Framework
Implements coordination between edge and server training with dynamic splitting
Based on the complete StreamSplit paper and convergence guarantees in Section 3.4
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import websockets
import aiohttp
from websockets.exceptions import ConnectionClosedError

from .edge_trainer import EdgeTrainer, TrainingConfig as EdgeConfig
from .server_trainer import ServerTrainer, ServerConfig
from ..core.dynamic_splitting import DynamicSplittingAgent, SplitDecision
from ..models.losses import HybridSWLaplacianLoss
from ..utils.metrics import MetricsCollector
from ..utils.data_utils import OptimalTransport

# Export main classes
__all__ = ['DistributedCoordinator', 'CommunicationManager', 'SynchronizationProtocol',
           'ConvergenceTracker', 'FederatedAggregator', 'create_distributed_coordinator']

class TrainingMode(Enum):
    """Training modes for distributed coordination"""
    EDGE_ONLY = "edge_only"
    SERVER_ONLY = "server_only"
    FEDERATED = "federated"
    DYNAMIC_SPLIT = "dynamic_split"

class SynchronizationProtocol(Enum):
    """Synchronization protocols between edge and server"""
    ASYNCHRONOUS = "async"
    PERIODIC_SYNC = "periodic"
    ADAPTIVE_SYNC = "adaptive"
    EVENT_DRIVEN = "event_driven"

@dataclass
class DistributedConfig:
    """Configuration for distributed training coordination"""
    # Communication settings
    communication_backend: str = "websocket"  # websocket, http, grpc
    server_host: str = "localhost"
    server_port: int = 8888
    edge_id: str = "edge_device_1"
    
    # Synchronization settings
    sync_protocol: SynchronizationProtocol = SynchronizationProtocol.ADAPTIVE_SYNC
    sync_interval: float = 30.0  # seconds
    adaptive_sync_threshold: float = 0.1  # accuracy delta threshold
    
    # Training coordination
    training_mode: TrainingMode = TrainingMode.DYNAMIC_SPLIT
    max_edge_only_steps: int = 100
    min_server_sync_interval: float = 10.0
    
    # Convergence tracking
    convergence_window: int = 100
    convergence_threshold: float = 1e-4
    max_training_steps: int = 10000
    
    # Resource management
    bandwidth_limit: float = 5.0  # Mbps
    priority_queue_size: int = 1000
    compression_enabled: bool = True
    
    # Fault tolerance
    max_retries: int = 3
    heartbeat_interval: float = 5.0
    connection_timeout: float = 30.0

class CommunicationMessage:
    """Message format for edge-server communication"""
    
    def __init__(self, message_type: str, sender_id: str, data: Dict[str, Any], 
                 priority: int = 0, compression: bool = False):
        self.message_type = message_type
        self.sender_id = sender_id
        self.data = data
        self.priority = priority
        self.compression = compression
        self.timestamp = time.time()
        self.message_id = f"{sender_id}_{int(self.timestamp * 1000000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type,
            'sender_id': self.sender_id,
            'data': self.data,
            'priority': self.priority,
            'compression': self.compression,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, msg_dict: Dict[str, Any]) -> 'CommunicationMessage':
        """Create message from dictionary"""
        msg = cls(
            message_type=msg_dict['message_type'],
            sender_id=msg_dict['sender_id'],
            data=msg_dict['data'],
            priority=msg_dict.get('priority', 0),
            compression=msg_dict.get('compression', False)
        )
        msg.timestamp = msg_dict.get('timestamp', time.time())
        msg.message_id = msg_dict.get('message_id', msg.message_id)
        return msg

class CommunicationManager:
    """
    Manages communication between edge and server
    Handles message routing, compression, and fault tolerance
    """
    
    def __init__(self, config: DistributedConfig, is_server: bool = False):
        self.config = config
        self.is_server = is_server
        self.logger = logging.getLogger(__name__)
        
        # Connection management
        self.websocket = None
        self.http_session = None
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Message queues
        self.outgoing_queue = asyncio.PriorityQueue(maxsize=config.priority_queue_size)
        self.incoming_queue = asyncio.Queue()
        self.message_handlers = {}
        
        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.connection_failures = 0
        
        # Background tasks
        self.communication_tasks = []
        self.heartbeat_task = None
        
        # Compression
        if config.compression_enabled:
            import gzip
            self.compressor = gzip
        else:
            self.compressor = None
        
        self.logger.info(f"CommunicationManager initialized for {'server' if is_server else 'edge'}")
    
    async def start(self):
        """Start communication manager"""
        if self.is_server:
            await self._start_server()
        else:
            await self._start_client()
        
        # Start background tasks
        self.communication_tasks = [
            asyncio.create_task(self._message_sender()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._heartbeat_sender()),
        ]
        
        self.logger.info("CommunicationManager started")
    
    async def stop(self):
        """Stop communication manager"""
        # Cancel background tasks
        for task in self.communication_tasks:
            task.cancel()
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Close connections
        await self._close_connections()
        
        self.logger.info("CommunicationManager stopped")
    
    async def _start_server(self):
        """Start WebSocket server"""
        async def handler(websocket, path):
            self.logger.info(f"Edge device connected from {websocket.remote_address}")
            self.websocket = websocket
            self.connected = True
            
            try:
                await self._handle_client_messages(websocket)
            except ConnectionClosedError:
                self.logger.warning("Edge device disconnected")
            finally:
                self.connected = False
                self.websocket = None
        
        # Start WebSocket server
        self.server = await websockets.serve(
            handler, 
            self.config.server_host, 
            self.config.server_port
        )
        
        self.logger.info(f"WebSocket server started on {self.config.server_host}:{self.config.server_port}")
    
    async def _start_client(self):
        """Start WebSocket client (edge device)"""
        server_uri = f"ws://{self.config.server_host}:{self.config.server_port}"
        
        for attempt in range(self.config.max_retries):
            try:
                self.websocket = await websockets.connect(
                    server_uri,
                    timeout=self.config.connection_timeout
                )
                self.connected = True
                self.logger.info(f"Connected to server at {server_uri}")
                
                # Start message handler
                asyncio.create_task(self._handle_server_messages())
                break
                
            except Exception as e:
                self.connection_failures += 1
                self.logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ConnectionError(f"Failed to connect after {self.config.max_retries} attempts")
    
    async def _handle_client_messages(self, websocket):
        """Handle messages from edge device (server side)"""
        async for message in websocket:
            try:
                msg_dict = json.loads(message)
                msg = CommunicationMessage.from_dict(msg_dict)
                await self.incoming_queue.put(msg)
                self.messages_received += 1
                self.bytes_received += len(message.encode())
            except Exception as e:
                self.logger.error(f"Error processing client message: {e}")
    
    async def _handle_server_messages(self):
        """Handle messages from server (edge side)"""
        try:
            async for message in self.websocket:
                try:
                    msg_dict = json.loads(message)
                    msg = CommunicationMessage.from_dict(msg_dict)
                    await self.incoming_queue.put(msg)
                    self.messages_received += 1
                    self.bytes_received += len(message.encode())
                except Exception as e:
                    self.logger.error(f"Error processing server message: {e}")
        except ConnectionClosedError:
            self.logger.warning("Connection to server lost")
            self.connected = False
    
    async def _message_sender(self):
        """Background task to send queued messages"""
        while True:
            try:
                # Get message from priority queue
                priority, msg = await self.outgoing_queue.get()
                
                if self.connected and self.websocket:
                    # Serialize message
                    msg_json = json.dumps(msg.to_dict())
                    
                    # Apply compression if enabled
                    if self.compressor and msg.compression:
                        msg_json = self.compressor.compress(msg_json.encode()).decode('latin-1')
                    
                    # Send message
                    await self.websocket.send(msg_json)
                    self.messages_sent += 1
                    self.bytes_sent += len(msg_json.encode())
                    
                    self.logger.debug(f"Sent message {msg.message_id} of type {msg.message_type}")
                else:
                    # Connection lost, requeue message with higher priority
                    await self.outgoing_queue.put((priority - 1, msg))
                    await asyncio.sleep(1.0)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")
                await asyncio.sleep(1.0)
    
    async def _message_processor(self):
        """Background task to process incoming messages"""
        while True:
            try:
                msg = await self.incoming_queue.get()
                
                # Handle message based on type
                if msg.message_type in self.message_handlers:
                    handler = self.message_handlers[msg.message_type]
                    try:
                        await handler(msg)
                    except Exception as e:
                        self.logger.error(f"Error handling message {msg.message_id}: {e}")
                else:
                    self.logger.warning(f"No handler for message type {msg.message_type}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1.0)
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeat messages"""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.connected:
                    heartbeat_msg = CommunicationMessage(
                        message_type="heartbeat",
                        sender_id=self.config.edge_id if not self.is_server else "server",
                        data={"timestamp": time.time()}
                    )
                    await self.send_message(heartbeat_msg, priority=10)  # Low priority
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
    
    async def send_message(self, message: CommunicationMessage, priority: int = 0):
        """Send message with specified priority"""
        await self.outgoing_queue.put((priority, message))
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")
    
    async def _close_connections(self):
        """Close all connections"""
        with self.connection_lock:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            if hasattr(self, 'server'):
                self.server.close()
                await self.server.wait_closed()
            
            if self.http_session:
                await self.http_session.close()
            
            self.connected = False
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            'connected': self.connected,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'connection_failures': self.connection_failures,
            'outgoing_queue_size': self.outgoing_queue.qsize(),
            'incoming_queue_size': self.incoming_queue.qsize()
        }

class ConvergenceTracker:
    """
    Track convergence of distributed training based on Section 3.4
    Implements convergence bounds from Theorem 1
    """
    
    def __init__(self, window_size: int = 100, convergence_threshold: float = 1e-4):
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Convergence metrics
        self.loss_history = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=window_size)
        self.parameter_distances = deque(maxlen=window_size)
        
        # Convergence bounds (from Theorem 1)
        self.mu = 1e-4  # Strong convexity parameter
        self.L = 1.0    # Smoothness parameter
        self.sigma = 0.1  # Transmission noise bound
        self.epsilon = 0.05  # Splitting approximation error
        
        # Convergence state
        self.converged = False
        self.convergence_step = None
        self.convergence_rate = None
        
    def update(self, loss: float, accuracy: float, param_distance: float = None):
        """Update convergence tracking with new metrics"""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        
        if param_distance is not None:
            self.parameter_distances.append(param_distance)
        
        # Check convergence
        self._check_convergence()
    
    def _check_convergence(self):
        """Check if training has converged based on loss and accuracy stability"""
        if len(self.loss_history) < self.window_size:
            return
        
        # Calculate loss variance over window
        loss_variance = np.var(list(self.loss_history))
        
        # Calculate accuracy stability
        acc_variance = np.var(list(self.accuracy_history))
        
        # Check convergence criteria
        loss_converged = loss_variance < self.convergence_threshold
        acc_converged = acc_variance < 0.001  # 0.1% accuracy variance
        
        if loss_converged and acc_converged and not self.converged:
            self.converged = True
            self.convergence_step = len(self.loss_history)
            self.convergence_rate = self._estimate_convergence_rate()
            
            self.logger.info(f"Training converged at step {self.convergence_step}")
            self.logger.info(f"Estimated convergence rate: {self.convergence_rate}")
    
    def _estimate_convergence_rate(self) -> float:
        """Estimate convergence rate based on loss trajectory"""
        if len(self.loss_history) < 2:
            return 0.0
        
        losses = np.array(list(self.loss_history))
        
        # Fit exponential decay to estimate convergence rate
        t = np.arange(len(losses))
        
        try:
            # Log-linear fit for exponential convergence
            log_losses = np.log(losses - losses[-1] + 1e-8)
            coeffs = np.polyfit(t, log_losses, 1)
            convergence_rate = -coeffs[0]  # Negative slope gives convergence rate
        except:
            convergence_rate = 0.0
        
        return max(0.0, convergence_rate)
    
    def get_convergence_bound(self, T: int) -> float:
        """
        Calculate theoretical convergence bound from Theorem 1
        E[|L(f_T) - L(f*)|] ≤ Lσ²/(2μ²T) + Lε²/2
        """
        # Convex case bound
        bound = (self.L * self.sigma**2) / (2 * self.mu**2 * T) + (self.L * self.epsilon**2) / 2
        return bound
    
    def get_parameter_convergence_bound(self, T: int) -> float:
        """
        Calculate parameter convergence bound
        E[||f_T - f*||²] ≤ C₁/T + C₂ε²
        """
        C1 = 2 * self.sigma**2 / self.mu**2
        C2 = 2
        return C1 / T + C2 * self.epsilon**2
    
    def is_converged(self) -> bool:
        """Check if training has converged"""
        return self.converged
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive convergence metrics"""
        if len(self.loss_history) == 0:
            return {}
        
        metrics = {
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'convergence_rate': self.convergence_rate,
            'current_loss': self.loss_history[-1],
            'current_accuracy': self.accuracy_history[-1],
            'loss_variance': np.var(list(self.loss_history)),
            'accuracy_variance': np.var(list(self.accuracy_history)),
            'theoretical_bound': self.get_convergence_bound(len(self.loss_history)),
            'parameter_bound': self.get_parameter_convergence_bound(len(self.loss_history))
        }
        
        if self.parameter_distances:
            metrics['parameter_distance'] = self.parameter_distances[-1]
            metrics['parameter_variance'] = np.var(list(self.parameter_distances))
        
        return metrics

class FederatedAggregator:
    """
    Federated aggregation for multi-device scenarios
    Implements server-side aggregation from Section 3.2.2
    """
    
    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method
        self.logger = logging.getLogger(__name__)
        
        # Device tracking
        self.device_weights = defaultdict(float)
        self.device_updates = defaultdict(list)
        self.device_last_seen = defaultdict(float)
        
        # Aggregation state
        self.global_model_state = None
        self.aggregation_round = 0
        
        # Optimal transport for distribution alignment
        self.optimal_transport = OptimalTransport()
        
    def add_device_update(self, device_id: str, model_update: Dict[str, torch.Tensor],
                         weight: float = 1.0, embeddings: torch.Tensor = None):
        """Add model update from device"""
        self.device_weights[device_id] = weight
        self.device_updates[device_id].append(model_update)
        self.device_last_seen[device_id] = time.time()
        
        # Store embeddings for distribution alignment if provided
        if embeddings is not None:
            self.device_updates[device_id][-1]['embeddings'] = embeddings
        
        self.logger.debug(f"Added update from device {device_id} with weight {weight}")
    
    def aggregate(self, min_devices: int = 1) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregate updates from participating devices"""
        # Check if we have enough devices
        active_devices = [
            device_id for device_id, updates in self.device_updates.items()
            if updates and time.time() - self.device_last_seen[device_id] < 300  # 5 minutes
        ]
        
        if len(active_devices) < min_devices:
            self.logger.warning(f"Not enough active devices for aggregation: {len(active_devices)}")
            return None
        
        if self.aggregation_method == "fedavg":
            aggregated = self._federated_averaging(active_devices)
        elif self.aggregation_method == "fedprox":
            aggregated = self._federated_proximal(active_devices)
        elif self.aggregation_method == "distribution_aware":
            aggregated = self._distribution_aware_aggregation(active_devices)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # Update global model state
        self.global_model_state = aggregated
        self.aggregation_round += 1
        
        # Clear device updates after aggregation
        for device_id in active_devices:
            self.device_updates[device_id].clear()
        
        self.logger.info(f"Aggregated updates from {len(active_devices)} devices (round {self.aggregation_round})")
        return aggregated
    
    def _federated_averaging(self, device_ids: List[str]) -> Dict[str, torch.Tensor]:
        """Standard FedAvg aggregation"""
        aggregated = {}
        total_weight = 0.0
        
        # Collect all updates with weights
        weighted_updates = []
        for device_id in device_ids:
            if self.device_updates[device_id]:
                update = self.device_updates[device_id][-1]  # Most recent update
                weight = self.device_weights[device_id]
                weighted_updates.append((update, weight))
                total_weight += weight
        
        if not weighted_updates:
            return {}
        
        # Initialize aggregated state with first update
        first_update, _ = weighted_updates[0]
        for key in first_update:
            if isinstance(first_update[key], torch.Tensor):
                aggregated[key] = torch.zeros_like(first_update[key])
        
        # Weighted averaging
        for update, weight in weighted_updates:
            normalized_weight = weight / total_weight
            for key in aggregated:
                if key in update and isinstance(update[key], torch.Tensor):
                    aggregated[key] += normalized_weight * update[key]
        
        return aggregated
    
    def _federated_proximal(self, device_ids: List[str]) -> Dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term"""
        # For now, fall back to FedAvg
        # Could implement proximal term based on global model distance
        return self._federated_averaging(device_ids)
    
    def _distribution_aware_aggregation(self, device_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Distribution-aware aggregation using optimal transport
        Aligns device distributions before aggregating
        """
        # Collect embeddings from devices
        device_embeddings = {}
        device_updates = {}
        
        for device_id in device_ids:
            if self.device_updates[device_id]:
                update = self.device_updates[device_id][-1]
                if 'embeddings' in update:
                    device_embeddings[device_id] = update['embeddings']
                    device_updates[device_id] = {k: v for k, v in update.items() if k != 'embeddings'}
        
        if len(device_embeddings) < 2:
            # Fall back to standard FedAvg if not enough embeddings
            return self._federated_averaging(device_ids)
        
        # Create global reference distribution
        all_embeddings = torch.cat(list(device_embeddings.values()), dim=0)
        global_distribution = all_embeddings  # Could be improved with clustering
        
        # Compute alignment weights based on Wasserstein distance
        alignment_weights = {}
        for device_id, embeddings in device_embeddings.items():
            # Compute Wasserstein distance to global distribution
            wd = self.optimal_transport.wasserstein_distance(embeddings, global_distribution)
            # Convert distance to weight (closer distributions get higher weight)
            alignment_weights[device_id] = 1.0 / (1.0 + wd.item())
        
        # Normalize alignment weights
        total_alignment_weight = sum(alignment_weights.values())
        for device_id in alignment_weights:
            alignment_weights[device_id] /= total_alignment_weight
        
        # Aggregate with alignment weights
        aggregated = {}
        for device_id in device_ids:
            if device_id in device_updates:
                update = device_updates[device_id]
                weight = alignment_weights.get(device_id, self.device_weights[device_id])
                
                if not aggregated:
                    # Initialize
                    for key, value in update.items():
                        if isinstance(value, torch.Tensor):
                            aggregated[key] = weight * value
                else:
                    # Accumulate
                    for key, value in update.items():
                        if key in aggregated and isinstance(value, torch.Tensor):
                            aggregated[key] += weight * value
        
        return aggregated
    
    def get_device_stats(self) -> Dict[str, Any]:
        """Get statistics about participating devices"""
        return {
            'active_devices': len(self.device_updates),
            'total_aggregation_rounds': self.aggregation_round,
            'device_weights': dict(self.device_weights),
            'device_last_seen': dict(self.device_last_seen)
        }

class DistributedCoordinator:
    """
    Main coordinator for distributed training
    Orchestrates edge and server training with dynamic splitting
    """
    
    def __init__(self, config: DistributedConfig, edge_config: EdgeConfig, 
                 server_config: ServerConfig, is_server: bool = False):
        self.config = config
        self.edge_config = edge_config
        self.server_config = server_config
        self.is_server = is_server
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.communication_manager = CommunicationManager(config, is_server)
        self.metrics_collector = MetricsCollector()
        self.convergence_tracker = ConvergenceTracker(
            window_size=config.convergence_window,
            convergence_threshold=config.convergence_threshold
        )
        
        # Training components
        self.edge_trainer = None
        self.server_trainer = None
        self.splitting_agent = None
        self.federated_aggregator = FederatedAggregator()
        
        # Coordination state
        self.training_mode = config.training_mode
        self.current_split_point = None
        self.last_sync_time = 0
        self.training_step = 0
        self.is_training = False
        
        # Performance tracking
        self.performance_history = {
            'loss': deque(maxlen=1000),
            'accuracy': deque(maxlen=1000),
            'bandwidth_usage': deque(maxlen=1000),
            'latency': deque(maxlen=1000),
            'split_decisions': deque(maxlen=1000)
        }
        
        # Synchronization state
        self.pending_updates = {}
        self.sync_lock = asyncio.Lock()
        
        # Register message handlers
        self._register_message_handlers()
        
        self.logger.info(f"DistributedCoordinator initialized for {'server' if is_server else 'edge'}")
    
    async def start(self, edge_trainer: EdgeTrainer = None, 
                   server_trainer: ServerTrainer = None,
                   splitting_agent: DynamicSplittingAgent = None):
        """Start distributed training coordination"""
        self.edge_trainer = edge_trainer
        self.server_trainer = server_trainer
        self.splitting_agent = splitting_agent
        
        # Start communication manager
        await self.communication_manager.start()
        
        # Start metrics collection
        self.metrics_collector.start_monitoring()
        
        # Start coordination tasks
        if not self.is_server:
            # Edge device tasks
            asyncio.create_task(self._edge_coordination_loop())
            asyncio.create_task(self._periodic_sync_sender())
        else:
            # Server tasks
            asyncio.create_task(self._server_coordination_loop())
            asyncio.create_task(self._aggregation_loop())
        
        self.is_training = True
        self.logger.info("Distributed training coordination started")
    
    async def stop(self):
        """Stop distributed training coordination"""
        self.is_training = False
        
        # Stop components
        await self.communication_manager.stop()
        self.metrics_collector.stop_monitoring()
        
        self.logger.info("Distributed training coordination stopped")
    
    def _register_message_handlers(self):
        """Register handlers for different message types"""
        handlers = {
            'model_update': self._handle_model_update,
            'split_decision': self._handle_split_decision,
            'sync_request': self._handle_sync_request,
            'sync_response': self._handle_sync_response,
            'embeddings': self._handle_embeddings,
            'performance_metrics': self._handle_performance_metrics,
            'heartbeat': self._handle_heartbeat,
            'training_complete': self._handle_training_complete
        }
        
        for msg_type, handler in handlers.items():
            self.communication_manager.register_handler(msg_type, handler)
    
    async def _edge_coordination_loop(self):
        """Main coordination loop for edge device"""
        while self.is_training:
            try:
                # Get current split decision
                if self.splitting_agent and self.training_mode == TrainingMode.DYNAMIC_SPLIT:
                    split_decision = await self.splitting_agent.get_split_decision()
                    self.current_split_point = split_decision['split_point']
                    
                    # Send split decision to server
                    msg = CommunicationMessage(
                        message_type='split_decision',
                        sender_id=self.config.edge_id,
                        data=split_decision
                    )
                    await self.communication_manager.send_message(msg, priority=1)
                
                # Train based on current mode
                if self.training_mode in [TrainingMode.EDGE_ONLY, TrainingMode.DYNAMIC_SPLIT]:
                    if self.edge_trainer:
                        # Perform edge training step
                        # Note: This would be called with actual audio data in real implementation
                        pass
                
                # Check if sync is needed
                await self._check_and_sync()
                
                # Update convergence tracking
                if self.edge_trainer:
                    metrics = self.edge_trainer.get_training_metrics()
                    loss = metrics.get('mean_loss', 0.0)
                    # Accuracy would come from periodic evaluation
                    accuracy = 0.8  # Placeholder
                    self.convergence_tracker.update(loss, accuracy)
                
                self.training_step += 1
                
                # Check convergence
                if self.convergence_tracker.is_converged():
                    await self._signal_training_complete()
                    break
                
                # Adaptive sleep based on training intensity
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in edge coordination loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _server_coordination_loop(self):
        """Main coordination loop for server"""
        while self.is_training:
            try:
                # Process pending updates
                await self._process_pending_updates()
                
                # Train server model if in appropriate mode
                if self.training_mode in [TrainingMode.SERVER_ONLY, TrainingMode.DYNAMIC_SPLIT]:
                    if self.server_trainer:
                        # Server training would happen here
                        pass
                
                # Monitor connected devices
                await self._monitor_devices()
                
                # Update convergence tracking (server side)
                if self.server_trainer:
                    metrics = self.server_trainer.get_performance_metrics()
                    loss = metrics.get('avg_loss', 0.0)
                    # Placeholder accuracy
                    accuracy = 0.85
                    self.convergence_tracker.update(loss, accuracy)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in server coordination loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _periodic_sync_sender(self):
        """Send periodic synchronization from edge to server"""
        while self.is_training:
            try:
                await asyncio.sleep(self.config.sync_interval)
                
                if self.config.sync_protocol == SynchronizationProtocol.PERIODIC_SYNC:
                    await self._send_sync_request()
                
            except Exception as e:
                self.logger.error(f"Error in periodic sync sender: {e}")
    
    async def _aggregation_loop(self):
        """Federated aggregation loop for server"""
        while self.is_training:
            try:
                await asyncio.sleep(30.0)  # Aggregate every 30 seconds
                
                if self.training_mode == TrainingMode.FEDERATED:
                    # Perform federated aggregation
                    aggregated_update = self.federated_aggregator.aggregate(min_devices=1)
                    
                    if aggregated_update and self.server_trainer:
                        # Apply aggregated update to server model
                        await self._apply_aggregated_update(aggregated_update)
                        
                        # Broadcast updated model to all devices
                        await self._broadcast_model_update(aggregated_update)
                
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
    
    async def _check_and_sync(self):
        """Check if synchronization is needed and initiate if so"""
        current_time = time.time()
        time_since_sync = current_time - self.last_sync_time
        
        should_sync = False
        
        if self.config.sync_protocol == SynchronizationProtocol.PERIODIC_SYNC:
            should_sync = time_since_sync >= self.config.sync_interval
        elif self.config.sync_protocol == SynchronizationProtocol.ADAPTIVE_SYNC:
            # Adaptive sync based on performance metrics
            if self.edge_trainer and time_since_sync >= self.config.min_server_sync_interval:
                metrics = self.edge_trainer.get_training_metrics()
                loss_variance = metrics.get('std_loss', 0.0)
                should_sync = loss_variance > self.config.adaptive_sync_threshold
        elif self.config.sync_protocol == SynchronizationProtocol.EVENT_DRIVEN:
            # Event-driven sync based on significant changes
            should_sync = await self._check_significant_change()
        
        if should_sync:
            await self._send_sync_request()
            self.last_sync_time = current_time
    
    async def _send_sync_request(self):
        """Send synchronization request to server"""
        if not self.edge_trainer:
            return
        
        # Prepare model state for sync
        model_state = self.edge_trainer.get_state()
        
        # Get performance metrics
        performance_metrics = self.edge_trainer.get_training_metrics()
        
        # Create sync message
        sync_data = {
            'device_id': self.config.edge_id,
            'training_step': self.training_step,
            'model_state': model_state,
            'performance_metrics': performance_metrics,
            'split_point': self.current_split_point,
            'timestamp': time.time()
        }
        
        msg = CommunicationMessage(
            message_type='sync_request',
            sender_id=self.config.edge_id,
            data=sync_data,
            compression=self.config.compression_enabled
        )
        
        await self.communication_manager.send_message(msg, priority=0)  # High priority
        self.logger.debug(f"Sent sync request for step {self.training_step}")
    
    async def _check_significant_change(self) -> bool:
        """Check if there's been a significant change warranting sync"""
        # Implement logic to detect significant changes
        # For example, loss improvement, accuracy change, etc.
        return False  # Placeholder
    
    async def _process_pending_updates(self):
        """Process pending updates from edge devices"""
        async with self.sync_lock:
            for device_id, update_data in list(self.pending_updates.items()):
                try:
                    await self._apply_device_update(device_id, update_data)
                    del self.pending_updates[device_id]
                except Exception as e:
                    self.logger.error(f"Error processing update from {device_id}: {e}")
    
    async def _apply_device_update(self, device_id: str, update_data: Dict[str, Any]):
        """Apply update from edge device"""
        if self.training_mode == TrainingMode.FEDERATED:
            # Add to federated aggregator
            model_state = update_data.get('model_state', {})
            if 'edge_state' in model_state:
                edge_state = model_state['edge_state']
                model_weights = edge_state.get('encoder_state_dict', {})
                
                # Extract embeddings if available
                embeddings = None
                if 'memory_bank_embeddings' in edge_state:
                    embeddings = edge_state['memory_bank_embeddings']
                
                self.federated_aggregator.add_device_update(
                    device_id, model_weights, weight=1.0, embeddings=embeddings
                )
        
        # Update performance tracking
        performance_metrics = update_data.get('performance_metrics', {})
        self._update_performance_history(device_id, performance_metrics)
    
    async def _apply_aggregated_update(self, aggregated_update: Dict[str, torch.Tensor]):
        """Apply aggregated update to server model"""
        if self.server_trainer:
            # This would update the server model with federated weights
            # Implementation depends on specific model architecture
            pass
    
    async def _broadcast_model_update(self, model_update: Dict[str, torch.Tensor]):
        """Broadcast model update to all connected devices"""
        broadcast_data = {
            'global_model_state': model_update,
            'aggregation_round': self.federated_aggregator.aggregation_round,
            'timestamp': time.time()
        }
        
        msg = CommunicationMessage(
            message_type='model_update',
            sender_id='server',
            data=broadcast_data,
            compression=self.config.compression_enabled
        )
        
        # Note: In a real implementation, this would be sent to all connected devices
        await self.communication_manager.send_message(msg, priority=0)
    
    async def _monitor_devices(self):
        """Monitor connected devices and their health"""
        # Get device statistics
        device_stats = self.federated_aggregator.get_device_stats()
        comm_stats = self.communication_manager.get_communication_stats()
        
        # Log device status
        if device_stats['active_devices'] > 0:
            self.logger.debug(f"Monitoring {device_stats['active_devices']} active devices")
    
    def _update_performance_history(self, device_id: str, metrics: Dict[str, Any]):
        """Update performance history with device metrics"""
        # Extract relevant metrics
        loss = metrics.get('mean_loss', 0.0)
        if loss > 0:
            self.performance_history['loss'].append(loss)
        
        # Update other metrics as available
        if 'memory_bank_utilization' in metrics:
            # Could track additional metrics here
            pass
    
    async def _signal_training_complete(self):
        """Signal that training has completed"""
        msg = CommunicationMessage(
            message_type='training_complete',
            sender_id=self.config.edge_id if not self.is_server else 'server',
            data={'convergence_metrics': self.convergence_tracker.get_convergence_metrics()}
        )
        
        await self.communication_manager.send_message(msg, priority=0)
        self.is_training = False
        self.logger.info("Training completion signaled")
    
    # Message handlers
    async def _handle_model_update(self, msg: CommunicationMessage):
        """Handle model update message"""
        if not self.is_server and self.edge_trainer:
            # Edge device receiving server model update
            global_state = msg.data.get('global_model_state', {})
            
            # Apply global model update
            if global_state and hasattr(self.edge_trainer, 'apply_server_update'):
                self.edge_trainer.apply_server_update(global_state)
                self.logger.debug(f"Applied model update from server")
    
    async def _handle_split_decision(self, msg: CommunicationMessage):
        """Handle split decision message"""
        if self.is_server:
            split_decision = msg.data
            device_id = msg.sender_id
            
            # Update server's knowledge of device split point
            self.logger.debug(f"Device {device_id} split decision: {split_decision['split_point']}")
            
            # Could use this information for server-side optimization
    
    async def _handle_sync_request(self, msg: CommunicationMessage):
        """Handle synchronization request from edge device"""
        if self.is_server:
            device_id = msg.sender_id
            sync_data = msg.data
            
            # Store update for processing
            async with self.sync_lock:
                self.pending_updates[device_id] = sync_data
            
            # Send sync response
            response_data = {
                'device_id': device_id,
                'status': 'acknowledged',
                'timestamp': time.time()
            }
            
            # Include server model state if needed
            if self.training_mode == TrainingMode.DYNAMIC_SPLIT and self.server_trainer:
                response_data['server_state'] = self.server_trainer.get_state()
            
            response_msg = CommunicationMessage(
                message_type='sync_response',
                sender_id='server',
                data=response_data
            )
            
            await self.communication_manager.send_message(response_msg, priority=1)
    
    async def _handle_sync_response(self, msg: CommunicationMessage):
        """Handle synchronization response from server"""
        if not self.is_server:
            response_data = msg.data
            status = response_data.get('status', 'unknown')
            
            self.logger.debug(f"Received sync response: {status}")
            
            # Apply server state if provided
            if 'server_state' in response_data and self.edge_trainer:
                # Could apply server updates here
                pass
    
    async def _handle_embeddings(self, msg: CommunicationMessage):
        """Handle embeddings message"""
        if self.is_server and self.server_trainer:
            embeddings_data = msg.data
            device_id = msg.sender_id
            
            # Process embeddings
            await self.server_trainer.receive_edge_embedding(embeddings_data)
    
    async def _handle_performance_metrics(self, msg: CommunicationMessage):
        """Handle performance metrics message"""
        device_id = msg.sender_id
        metrics = msg.data
        
        # Update metrics collector
        self.metrics_collector.update_performance(metrics)
        
        # Update performance history
        self._update_performance_history(device_id, metrics)
    
    async def _handle_heartbeat(self, msg: CommunicationMessage):
        """Handle heartbeat message"""
        # Update device last seen time
        device_id = msg.sender_id
        timestamp = msg.data.get('timestamp', time.time())
        
        # Could track device health here
        self.logger.debug(f"Heartbeat from {device_id} at {timestamp}")
    
    async def _handle_training_complete(self, msg: CommunicationMessage):
        """Handle training completion message"""
        device_id = msg.sender_id
        convergence_metrics = msg.data.get('convergence_metrics', {})
        
        self.logger.info(f"Device {device_id} reported training complete")
        self.logger.info(f"Convergence metrics: {convergence_metrics}")
        
        # Could coordinate global training completion here
    
    # Public interface methods
    async def train_step(self, audio_data: np.ndarray = None, 
                        spectrogram: torch.Tensor = None) -> Dict[str, Any]:
        """
        Perform a distributed training step
        
        Args:
            audio_data: Raw audio data
            spectrogram: Preprocessed spectrogram
            
        Returns:
            Training step results
        """
        results = {}
        
        if self.training_mode == TrainingMode.EDGE_ONLY:
            if self.edge_trainer and spectrogram is not None:
                results = self.edge_trainer.train_step(spectrogram)
        
        elif self.training_mode == TrainingMode.SERVER_ONLY:
            if self.server_trainer and spectrogram is not None:
                results = await self.server_trainer.process(spectrogram)
        
        elif self.training_mode == TrainingMode.DYNAMIC_SPLIT:
            if self.edge_trainer and self.splitting_agent and spectrogram is not None:
                # Get split decision
                split_decision = await self.splitting_agent.get_split_decision()
                split_point = split_decision['split_point']
                
                # Process on edge up to split point
                edge_result = await self.edge_trainer.edge_module.process_partial(
                    spectrogram, split_point
                )
                
                # Determine if we should transmit to server
                should_transmit = self._should_transmit_to_server(edge_result)
                
                if should_transmit and self.server_trainer:
                    # Send to server for completion
                    server_result = await self.server_trainer.process_continuation(
                        edge_result, split_point
                    )
                    results = {**edge_result, **server_result}
                else:
                    # Complete on edge
                    complete_result = await self.edge_trainer.edge_module.process_complete(
                        edge_result
                    )
                    results = complete_result
                
                # Update split agent with performance
                await self.splitting_agent.update_performance(self.performance_history)
        
        # Update training step counter
        self.training_step += 1
        
        # Update performance history
        if 'loss' in results:
            self.performance_history['loss'].append(results['loss'])
        
        return results
    
    def _should_transmit_to_server(self, edge_result: Dict[str, Any]) -> bool:
        """Determine whether to transmit edge result to server"""
        # Extract uncertainty from edge result
        uncertainty = edge_result.get('uncertainty', {})
        consistency_uncertainty = uncertainty.get('consistency_uncertainty', 0.0)
        
        # Simple threshold-based decision
        return consistency_uncertainty > 0.5
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            'training_step': self.training_step,
            'training_mode': self.training_mode.value,
            'is_training': self.is_training,
            'current_split_point': self.current_split_point,
            'convergence_metrics': self.convergence_tracker.get_convergence_metrics(),
            'communication_stats': self.communication_manager.get_communication_stats(),
            'performance_history_length': {
                key: len(history) for key, history in self.performance_history.items()
            }
        }
        
        if self.is_server:
            status['federated_stats'] = self.federated_aggregator.get_device_stats()
        
        return status
    
    def save_checkpoint(self, filepath: str):
        """Save distributed training checkpoint"""
        checkpoint = {
            'config': self.config,
            'training_step': self.training_step,
            'training_mode': self.training_mode.value,
            'current_split_point': self.current_split_point,
            'performance_history': {
                key: list(history) for key, history in self.performance_history.items()
            },
            'convergence_metrics': self.convergence_tracker.get_convergence_metrics(),
            'communication_stats': self.communication_manager.get_communication_stats()
        }
        
        # Add trainer states if available
        if self.edge_trainer:
            checkpoint['edge_trainer_state'] = self.edge_trainer.get_state()
        
        if self.server_trainer:
            checkpoint['server_trainer_state'] = self.server_trainer.get_state()
        
        if self.splitting_agent:
            checkpoint['splitting_agent_state'] = self.splitting_agent.get_state()
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        self.logger.info(f"Distributed training checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load distributed training checkpoint"""
        checkpoint = torch.load(filepath)
        
        self.training_step = checkpoint.get('training_step', 0)
        self.training_mode = TrainingMode(checkpoint.get('training_mode', 'dynamic_split'))
        self.current_split_point = checkpoint.get('current_split_point')
        
        # Restore performance history
        perf_history = checkpoint.get('performance_history', {})
        for key, history in perf_history.items():
            if key in self.performance_history:
                self.performance_history[key] = deque(history, maxlen=1000)
        
        # Restore trainer states if available
        if 'edge_trainer_state' in checkpoint and self.edge_trainer:
            self.edge_trainer.load_state(checkpoint['edge_trainer_state'])
        
        if 'server_trainer_state' in checkpoint and self.server_trainer:
            self.server_trainer.load_state(checkpoint['server_trainer_state'])
        
        if 'splitting_agent_state' in checkpoint and self.splitting_agent:
            self.splitting_agent.load_state(checkpoint['splitting_agent_state'])
        
        self.logger.info(f"Distributed training checkpoint loaded from {filepath}")

# Factory functions and utilities

async def create_distributed_coordinator(config_dict: Dict[str, Any],
                                       edge_config: EdgeConfig,
                                       server_config: ServerConfig,
                                       is_server: bool = False) -> DistributedCoordinator:
    """Create distributed coordinator with configuration"""
    config = DistributedConfig(**config_dict)
    coordinator = DistributedCoordinator(config, edge_config, server_config, is_server)
    return coordinator

async def run_distributed_training(edge_coordinator: DistributedCoordinator,
                                 server_coordinator: DistributedCoordinator,
                                 num_steps: int = 1000):
    """Run distributed training with both edge and server coordinators"""
    # Start both coordinators
    await edge_coordinator.start()
    await server_coordinator.start()
    
    try:
        # Training loop
        for step in range(num_steps):
            # Generate synthetic audio for testing
            test_spectrogram = torch.randn(128, 64)
            
            # Train on edge
            edge_result = await edge_coordinator.train_step(spectrogram=test_spectrogram)
            
            # Check convergence
            if edge_coordinator.convergence_tracker.is_converged():
                logging.info(f"Training converged at step {step}")
                break
            
            # Small delay between steps
            await asyncio.sleep(0.01)
    
    finally:
        # Clean shutdown
        await edge_coordinator.stop()
        await server_coordinator.stop()

# Example usage and testing

async def test_distributed_training():
    """Test distributed training functionality"""
    # Configuration
    distributed_config = {
        'edge_id': 'test_edge_1',
        'server_host': 'localhost',
        'server_port': 8889,
        'sync_interval': 5.0,
        'training_mode': 'dynamic_split'
    }
    
    edge_config = EdgeConfig(learning_rate=1e-4, batch_size=32)
    server_config = ServerConfig(learning_rate=5e-4, batch_size=256)
    
    # Create coordinators
    edge_coordinator = await create_distributed_coordinator(
        distributed_config, edge_config, server_config, is_server=False
    )
    
    server_coordinator = await create_distributed_coordinator(
        distributed_config, edge_config, server_config, is_server=True
    )
    
    # Run training
    logging.info("Starting distributed training test...")
    await run_distributed_training(edge_coordinator, server_coordinator, num_steps=100)
    logging.info("Distributed training test completed!")

if __name__ == "__main__":
    # Run test
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_distributed_training())