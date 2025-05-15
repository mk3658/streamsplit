#!/usr/bin/env python3
"""
StreamSplit Server Runner
Implements the complete server-side functionality as described in the StreamSplit paper
Based on Section 3.2 and related components
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
import json

# StreamSplit imports
from src.core.streamsplit import StreamSplitFramework, StreamSplitConfig, StreamSplitMode
from src.core.server_module import ServerModule
from src.core.dynamic_splitting import DynamicSplittingAgent
from src.training.server_trainer import ServerTrainer, ServerTrainingConfig
from src.models.encoders import MobileNetV3ServerEncoder, create_server_encoder
from src.utils.audio_processing import AudioProcessor, AudioConfig
from src.utils.metrics import MetricsCollector
from src.training.distributed_training import DistributedCoordinator, DistributedConfig
from src.models.losses import HybridSWLaplacianLoss

class ServerManager:
    """
    Main manager for StreamSplit server
    Coordinates all server components and handles lifecycle management
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.logger = None
        
        # Core components
        self.streamsplit_framework = None
        self.server_module = None
        self.server_trainer = None
        self.splitting_agent = None
        self.audio_processor = None
        self.metrics_collector = None
        self.distributed_coordinator = None
        
        # State management
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.start_time = None
        self.processed_embeddings = 0
        self.connected_devices = {}
        
        # Background tasks
        self.background_tasks = []
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ Loaded configuration from {self.config_path}")
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.config.get('monitoring', {}).get('log_level', 'INFO'))
        log_file = self.config.get('monitoring', {}).get('log_file', 'logs/server.log')
        
        # Create logs directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging with structured format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ServerManager initialized with config: {self.config_path}")
    
    def initialize_components(self):
        """Initialize all StreamSplit server components"""
        try:
            self.logger.info("Initializing StreamSplit server components...")
            
            # 1. Audio Processor (for server-only mode)
            self._initialize_audio_processor()
            
            # 2. Server Encoder Model
            self._initialize_server_model()
            
            # 3. Server Module (aggregation and refinement)
            self._initialize_server_module()
            
            # 4. Server Trainer
            self._initialize_server_trainer()
            
            # 5. Dynamic Splitting Agent (server-side coordination)
            self._initialize_splitting_agent()
            
            # 6. Metrics Collector
            self._initialize_metrics_collector()
            
            # 7. Distributed Coordinator (if enabled)
            self._initialize_distributed_coordinator()
            
            # 8. StreamSplit Framework
            self._initialize_streamsplit_framework()
            
            self.logger.info("✓ All server components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize components: {e}")
            self.logger.error(traceback.format_exc())
            sys.exit(1)
    
    def _initialize_audio_processor(self):
        """Initialize audio processing pipeline for server-only mode"""
        audio_config = AudioConfig(
            sample_rate=self.config['audio']['sample_rate'],
            window_duration=self.config['audio']['window_duration'],
            hop_length=self.config['audio']['hop_length'],
            n_fft=self.config['audio']['n_fft'],
            n_mels=self.config['audio']['n_mels'],
            fmin=self.config['audio']['fmin'],
            fmax=self.config['audio'].get('fmax'),
            power=self.config['audio']['power'],
            normalized=self.config['audio']['normalized']
        )
        
        self.audio_processor = AudioProcessor(audio_config)
        self.logger.info("✓ Audio processor initialized")
    
    def _initialize_server_model(self):
        """Initialize server encoder model"""
        # Detect device (GPU/CPU)
        if self.config.get('optimization', {}).get('use_gpu', True) and torch.cuda.is_available():
            device = 'cuda'
            gpu_ids = self.config.get('optimization', {}).get('gpu_device_ids', [0])
            if gpu_ids:
                device = f'cuda:{gpu_ids[0]}'
        else:
            device = 'cpu'
        self.device = device
        
        # Create server encoder
        self.server_model = MobileNetV3ServerEncoder(
            intermediate_dim=self.config['model']['intermediate_dim'],
            embedding_dim=self.config['model']['embedding_dim'],
            num_layers=self.config['model']['num_layers']
        ).to(device)
        
        # Enable model compilation if requested (PyTorch 2.0+)
        if self.config.get('optimization', {}).get('compile_model', False):
            try:
                self.server_model = torch.compile(self.server_model)
                self.logger.info("✓ Model compilation enabled")
            except Exception as e:
                self.logger.warning(f"Failed to compile model: {e}")
        
        self.logger.info(f"✓ Server model initialized on {device}")
    
    def _initialize_server_module(self):
        """Initialize server module with aggregation and hybrid loss"""
        # Create a config-like object for ServerModule
        class ServerConfig:
            def __init__(self, config_dict):
                # Server parameters
                self.learning_rate = config_dict['training']['learning_rate']
                self.batch_size = config_dict['training']['batch_size']
                self.update_frequency = config_dict['training']['update_frequency']
                
                # Hybrid loss parameters
                self.sw_projections = config_dict['loss']['sliced_wasserstein']['num_projections']
                self.sw_weight = config_dict['loss']['sliced_wasserstein']['weight']
                self.laplacian_k_neighbors = config_dict['loss']['laplacian']['k_neighbors']
                self.laplacian_weight = config_dict['loss']['laplacian']['weight']
                self.laplacian_sigma = config_dict['loss']['laplacian']['sigma']
                
                # Aggregation parameters
                self.temporal_window = config_dict['aggregation']['temporal_window']
                self.min_devices_per_update = config_dict['aggregation']['min_devices_per_update']
                self.max_age_threshold = config_dict['aggregation']['max_age_threshold']
                
                # Uncertainty parameters
                self.uncertainty_threshold_base = config_dict['transmission']['uncertainty']['base_threshold']
        
        server_config = ServerConfig(self.config)
        self.server_module = ServerModule(server_config)
        self.logger.info("✓ Server module initialized")
    
    def _initialize_server_trainer(self):
        """Initialize server trainer for distributed learning"""
        trainer_config = ServerTrainingConfig(
            learning_rate=self.config['training']['learning_rate'],
            batch_size=self.config['training']['batch_size'],
            update_frequency=self.config['training']['update_frequency'],
            temporal_window=self.config['aggregation']['temporal_window'],
            
            # Hybrid loss parameters
            sw_projections=self.config['loss']['sliced_wasserstein']['num_projections'],
            sw_weight=self.config['loss']['sliced_wasserstein']['weight'],
            laplacian_k_neighbors=self.config['loss']['laplacian']['k_neighbors'],
            laplacian_weight=self.config['loss']['laplacian']['weight'],
            laplacian_sigma=self.config['loss']['laplacian']['sigma'],
            
            # Aggregation parameters
            min_devices_per_update=self.config['aggregation']['min_devices_per_update'],
            max_age_threshold=self.config['aggregation']['max_age_threshold'],
            prototype_update_rate=self.config['prototypes']['update_rate'],
            
            # Optimization parameters
            scheduler_type=self.config['training']['scheduler_type'],
            weight_decay=self.config['training'].get('weight_decay', 1e-6),
            gradient_clip_norm=self.config['training'].get('gradient_clip_norm', 1.0)
        )
        
        self.server_trainer = ServerTrainer(trainer_config, self.server_model, device=str(self.device))
        self.logger.info("✓ Server trainer initialized")
    
    def _initialize_splitting_agent(self):
        """Initialize dynamic splitting coordination"""
        if not self.config['splitting']['enabled']:
            self.logger.info("Dynamic splitting disabled on server")
            return
        
        # Create a config-like object for server-side splitting coordination
        class SplittingConfig:
            def __init__(self, config_dict):
                # Server-side splitting parameters
                self.bandwidth_range = (0.5, 8.0)  # Will be updated based on connected devices
                self.latency_range = (50, 200)
                self.split_agent_lr = config_dict['splitting']['agent']['learning_rate']
                self.split_reward_weights = config_dict['splitting']['server_reward_weights']
        
        splitting_config = SplittingConfig(self.config)
        self.splitting_agent = DynamicSplittingAgent(splitting_config)
        self.logger.info("✓ Dynamic splitting agent initialized")
    
    def _initialize_metrics_collector(self):
        """Initialize metrics collection system"""
        self.metrics_collector = MetricsCollector(
            device=str(self.device),
            measurement_interval=self.config['monitoring']['metrics_interval']
        )
        self.logger.info("✓ Metrics collector initialized")
    
    def _initialize_distributed_coordinator(self):
        """Initialize distributed training coordinator"""
        if not self.config.get('distributed', {}).get('enabled', True):
            self.logger.info("Distributed coordination disabled")
            return
        
        # Create distributed config
        distributed_config = DistributedConfig(
            communication_backend=self.config['networking']['websocket'].get('enabled', True) and 'websocket' or 'http',
            server_host=self.config['server']['bind_address'],
            server_port=self.config['server']['port'],
            edge_id='server',  # Server identifier
            sync_protocol='adaptive_sync',
            training_mode='dynamic_split',
            max_connections=self.config['server']['max_connections']
        )
        
        # Create placeholder edge config (not used on server side)
        from src.training.edge_trainer import TrainingConfig as EdgeConfig
        edge_config = EdgeConfig()
        
        # Create server config for distributed coordinator
        server_config = ServerTrainingConfig()
        
        self.distributed_coordinator = DistributedCoordinator(
            distributed_config, edge_config, server_config, is_server=True
        )
        self.logger.info("✓ Distributed coordinator initialized")
    
    def _initialize_streamsplit_framework(self):
        """Initialize main StreamSplit framework for server-only mode"""
        # Create StreamSplit config
        streamsplit_config = StreamSplitConfig(
            device_id='server',
            device_type='gpu_server',
            sample_rate=self.config['audio']['sample_rate'],
            window_duration=self.config['audio']['window_duration'],
            hop_length=self.config['audio']['hop_length'],
            n_fft=self.config['audio']['n_fft'],
            n_mels=self.config['audio']['n_mels'],
            batch_size=self.config['training']['batch_size'],
            learning_rate=self.config['training']['learning_rate'],
            momentum=0.999,  # Not used on server
            temperature=0.1,   # Not used on server
            memory_bank_size_min=64,  # Not used on server
            memory_bank_size_max=512, # Not used on server
            uncertainty_threshold_base=self.config['transmission']['uncertainty']['base_threshold'],
            transmission_threshold=self.config['transmission']['uncertainty']['base_threshold'],
            resource_monitor_interval=self.config['monitoring']['metrics_interval'],
            cpu_threshold=0.8,  # Server CPU threshold
            bandwidth_range=(0.5, 8.0),
            latency_range=(50, 200),
            split_agent_lr=self.config['splitting']['agent']['learning_rate'] if self.config['splitting']['enabled'] else 1e-3,
            split_reward_weights=self.config['splitting'].get('server_reward_weights', {})
        )
        
        self.streamsplit_framework = StreamSplitFramework(streamsplit_config)
        self.logger.info("✓ StreamSplit framework initialized")
    
    async def start_server(self):
        """Start the StreamSplit server and all its components"""
        try:
            self.logger.info("Starting StreamSplit server...")
            self.start_time = time.time()
            self.is_running = True
            
            # Start metrics collection
            if self.metrics_collector:
                self.metrics_collector.start_monitoring()
                self.logger.info("✓ Metrics collection started")
            
            # Start server module
            await self.server_module.start()
            self.logger.info("✓ Server module started")
            
            # Start server trainer
            if self.server_trainer:
                await self.server_trainer.start_training()
                self.logger.info("✓ Server trainer started")
            
            # Start splitting agent
            if self.splitting_agent:
                await self.splitting_agent.start()
                self.logger.info("✓ Splitting agent started")
            
            # Start distributed coordinator
            if self.distributed_coordinator:
                await self.distributed_coordinator.start(
                    edge_trainer=None,  # Server doesn't have edge trainer
                    server_trainer=self.server_trainer,
                    splitting_agent=self.splitting_agent
                )
                self.logger.info("✓ Distributed coordinator started")
            
            # Determine mode based on configuration
            if self.config.get('splitting', {}).get('enabled', True):
                mode = StreamSplitMode.DYNAMIC_SPLIT
            else:
                mode = StreamSplitMode.SERVER_ONLY
            
            # Start StreamSplit framework
            await self.streamsplit_framework.start(mode=mode)
            self.logger.info(f"✓ StreamSplit framework started in {mode.value} mode")
            
            # Load checkpoint if available
            await self._load_checkpoint_if_available()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("🚀 StreamSplit server fully operational!")
            self.logger.info(f"Server listening on {self.config['server']['bind_address']}:{self.config['server']['port']}")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to start server: {e}")
            self.logger.error(traceback.format_exc())
            await self.stop_server()
            sys.exit(1)
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # Device monitoring task
        self.background_tasks.append(
            asyncio.create_task(self._device_monitoring_loop())
        )
        
        # Performance reporting task
        self.background_tasks.append(
            asyncio.create_task(self._performance_reporting_loop())
        )
        
        # Checkpoint saving task
        if self.config.get('checkpointing', {}).get('enabled', False):
            self.background_tasks.append(
                asyncio.create_task(self._periodic_checkpoint_task())
            )
        
        # Metrics export task
        if self.config.get('monitoring', {}).get('export_metrics', False):
            self.background_tasks.append(
                asyncio.create_task(self._periodic_metrics_export())
            )
    
    async def _load_checkpoint_if_available(self):
        """Load checkpoint if checkpointing is enabled and checkpoint exists"""
        if not self.config.get('checkpointing', {}).get('enabled', False):
            return
        
        checkpoint_dir = Path(self.config['checkpointing']['checkpoint_dir'])
        checkpoint_files = list(checkpoint_dir.glob('server_checkpoint_*.pt'))
        
        if checkpoint_files:
            # Load most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            try:
                # Load StreamSplit framework checkpoint
                self.streamsplit_framework.load_checkpoint(str(latest_checkpoint))
                self.logger.info(f"✓ Loaded checkpoint: {latest_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint {latest_checkpoint}: {e}")
    
    async def stop_server(self):
        """Stop the StreamSplit server and cleanup"""
        self.logger.info("Stopping StreamSplit server...")
        self.is_running = False
        
        # Signal shutdown to main loops
        self.shutdown_event.set()
        
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for background tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Save final checkpoint
            await self._save_checkpoint()
            
            # Stop StreamSplit framework
            if self.streamsplit_framework:
                await self.streamsplit_framework.stop()
                self.logger.info("✓ StreamSplit framework stopped")
            
            # Stop distributed coordinator
            if self.distributed_coordinator:
                await self.distributed_coordinator.stop()
                self.logger.info("✓ Distributed coordinator stopped")
            
            # Stop server trainer
            if self.server_trainer:
                await self.server_trainer.stop_training()
                self.logger.info("✓ Server trainer stopped")
            
            # Stop splitting agent
            if self.splitting_agent:
                await self.splitting_agent.stop()
                self.logger.info("✓ Splitting agent stopped")
            
            # Stop server module
            if self.server_module:
                await self.server_module.stop()
                self.logger.info("✓ Server module stopped")
            
            # Stop metrics collection
            if self.metrics_collector:
                self.metrics_collector.stop_monitoring()
                self.logger.info("✓ Metrics collection stopped")
            
            # Print final statistics
            self._print_final_statistics()
            
            self.logger.info("✓ Server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _device_monitoring_loop(self):
        """Monitor connected edge devices"""
        while self.is_running:
            try:
                if self.server_module:
                    # Get device statistics
                    device_stats = self.server_module.get_device_statistics()
                    
                    # Update connected devices tracking
                    current_time = time.time()
                    active_devices = 0
                    
                    for device_id, stats in device_stats.items():
                        if stats.get('is_active', False):
                            active_devices += 1
                            self.connected_devices[device_id] = {
                                'last_seen': stats.get('last_seen', current_time),
                                'embeddings_count': stats.get('embeddings_count', 0),
                                'status': 'active'
                            }
                        else:
                            if device_id in self.connected_devices:
                                self.connected_devices[device_id]['status'] = 'inactive'
                    
                    # Log device status periodically
                    if len(device_stats) > 0:
                        self.logger.debug(f"Active devices: {active_devices}/{len(device_stats)}")
                
                # Sleep before next check
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in device monitoring: {e}")
                await asyncio.sleep(5.0)
    
    async def _performance_reporting_loop(self):
        """Periodic performance reporting"""
        report_interval = self.config.get('monitoring', {}).get('report_interval', 60.0)
        
        while self.is_running:
            try:
                await asyncio.sleep(report_interval)
                
                if self.is_running:
                    # Get performance metrics
                    if self.server_trainer:
                        metrics = self.server_trainer.get_training_metrics()
                        self.logger.info(
                            f"Server Performance - "
                            f"Step: {metrics.get('global_step', 0)}, "
                            f"Loss: {metrics.get('average_loss', 0):.4f}, "
                            f"Devices: {metrics.get('active_devices', 0)}, "
                            f"Embeddings: {metrics.get('total_embeddings_processed', 0)}"
                        )
                    
                    # Get resource metrics
                    if self.metrics_collector:
                        resource_metrics = self.metrics_collector.resource_metrics.get_resource_summary()
                        cpu_metric = resource_metrics.get('cpu_efficiency')
                        memory_metric = resource_metrics.get('memory_efficiency')
                        
                        if cpu_metric and memory_metric:
                            self.logger.info(
                                f"Resource Usage - "
                                f"CPU: {cpu_metric.value:.1f}%, "
                                f"Memory: {memory_metric.value:.1f} MB"
                            )
                
                # Check for performance issues
                await self._check_performance_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance reporting: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_performance_health(self):
        """Check for performance issues and log warnings"""
        if not self.server_trainer:
            return
        
        metrics = self.server_trainer.get_training_metrics()
        
        # Check if no devices are connected
        active_devices = metrics.get('active_devices', 0)
        if active_devices == 0:
            self.logger.warning("No active edge devices connected")
        
        # Check if loss is increasing
        avg_loss = metrics.get('average_loss', 0)
        if avg_loss > 2.0:  # Threshold for high loss
            self.logger.warning(f"High training loss detected: {avg_loss:.4f}")
        
        # Check resource usage
        if self.metrics_collector:
            cpu_metrics = self.metrics_collector.resource_metrics.get_cpu_metrics()
            if cpu_metrics.value > 90.0:
                self.logger.warning(f"High CPU usage: {cpu_metrics.value:.1f}%")
    
    async def _periodic_checkpoint_task(self):
        """Periodic checkpoint saving task"""
        save_interval = self.config['checkpointing']['save_interval']
        
        while self.is_running:
            try:
                await asyncio.sleep(save_interval)
                if self.is_running:
                    await self._save_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in checkpoint task: {e}")
    
    async def _periodic_metrics_export(self):
        """Periodic metrics export task"""
        export_interval = self.config['monitoring']['export_interval']
        
        while self.is_running:
            try:
                await asyncio.sleep(export_interval)
                if self.is_running and self.metrics_collector:
                    # Export metrics
                    export_path = self.config['monitoring'].get('export_path', 'metrics/server_metrics.json')
                    export_format = self.config['monitoring'].get('export_format', 'json')
                    
                    # Create directory if needed
                    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Export with timestamp
                    timestamp = int(time.time())
                    base_path = Path(export_path)
                    timestamped_path = base_path.parent / f"{base_path.stem}_{timestamp}{base_path.suffix}"
                    
                    self.metrics_collector.export_metrics(str(timestamped_path), export_format)
                    self.logger.info(f"Metrics exported to {timestamped_path}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics export task: {e}")
    
    async def _save_checkpoint(self):
        """Save checkpoint if checkpointing is enabled"""
        if not self.config.get('checkpointing', {}).get('enabled', False):
            return
        
        checkpoint_dir = Path(self.config['checkpointing']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        checkpoint_path = checkpoint_dir / f"server_checkpoint_{timestamp}.pt"
        
        try:
            # Save StreamSplit framework checkpoint
            self.streamsplit_framework.save_checkpoint(str(checkpoint_path))
            self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Save server trainer checkpoint separately
            if self.server_trainer:
                trainer_checkpoint_path = checkpoint_dir / f"server_trainer_{timestamp}.pt"
                self.server_trainer.save_checkpoint(str(trainer_checkpoint_path))
                self.logger.info(f"✓ Trainer checkpoint saved: {trainer_checkpoint_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(checkpoint_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints based on retention policy"""
        max_checkpoints = self.config['checkpointing'].get('max_checkpoints', 10)
        
        # Clean StreamSplit checkpoints
        checkpoint_files = sorted(
            checkpoint_dir.glob('server_checkpoint_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoint_files[max_checkpoints:]:
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
        
        # Clean trainer checkpoints
        trainer_files = sorted(
            checkpoint_dir.glob('server_trainer_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in trainer_files[max_checkpoints:]:
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old trainer checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old trainer checkpoint {old_checkpoint}: {e}")
    
    def _print_final_statistics(self):
        """Print final performance statistics"""
        if self.start_time:
            runtime = time.time() - self.start_time
            self.logger.info("=== Server Statistics ===")
            self.logger.info(f"Runtime: {runtime:.2f} seconds")
            self.logger.info(f"Embeddings processed: {self.processed_embeddings}")
            self.logger.info(f"Connected devices (peak): {len(self.connected_devices)}")
            
            # Get final performance summary from framework
            if self.streamsplit_framework:
                summary = self.streamsplit_framework.get_performance_summary()
                for key, value in summary.items():
                    self.logger.info(f"{key}: {value}")
            
            # Get final training metrics
            if self.server_trainer:
                metrics = self.server_trainer.get_training_metrics()
                self.logger.info(f"Final loss: {metrics.get('average_loss', 0):.4f}")
                self.logger.info(f"Total updates: {metrics.get('total_updates', 0)}")
    
    async def main_loop(self):
        """Main server loop"""
        self.logger.info("Starting main server loop...")
        
        try:
            # Main server loop - handle incoming connections and process requests
            # In a real implementation, this would handle WebSocket connections
            # from edge devices and coordinate distributed training
            
            self.logger.info("Server running... (Press Ctrl+C to stop)")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for the server"""
    parser = argparse.ArgumentParser(description='StreamSplit Server')
    parser.add_argument('--config', type=str, default='config/server_config.yaml',
                       help='Path to server configuration file')
    parser.add_argument('--port', type=int, help='Override server port from config')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Override log level from config')
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU usage (if available)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU-only operation')
    parser.add_argument('--no-splitting', action='store_true',
                       help='Disable dynamic splitting (server-only mode)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    
    args = parser.parse_args()
    
    # Initialize server manager
    server_manager = ServerManager(args.config)
    
    try:
        # Load configuration
        server_manager.load_config()
        
        # Apply command line overrides
        if args.port:
            server_manager.config['server']['port'] = args.port
        
        if args.log_level:
            server_manager.config.setdefault('monitoring', {})['log_level'] = args.log_level
        
        if args.debug:
            server_manager.config.setdefault('monitoring', {})['log_level'] = 'DEBUG'
            server_manager.config.setdefault('debug', {})['enabled'] = True
        
        if args.gpu:
            server_manager.config.setdefault('optimization', {})['use_gpu'] = True
        
        if args.cpu_only:
            server_manager.config.setdefault('optimization', {})['use_gpu'] = False
        
        if args.no_splitting:
            server_manager.config['splitting']['enabled'] = False
        
        # Setup logging
        server_manager.setup_logging()
        
        # Setup signal handlers
        server_manager.setup_signal_handlers()
        
        # Initialize all components
        server_manager.initialize_components()
        
        # Start the server
        await server_manager.start_server()
        
        # Run main loop
        await server_manager.main_loop()
        
    except KeyboardInterrupt:
        server_manager.logger.info("Received keyboard interrupt")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean shutdown
        await server_manager.stop_server()


# Additional utility functions

async def run_server_only_inference(config_path: str, input_audio: np.ndarray):
    """
    Run server in inference mode for a single audio input
    Useful for testing server-only processing
    """
    server_manager = ServerManager(config_path)
    
    try:
        # Load and setup
        server_manager.load_config()
        server_manager.setup_logging()
        server_manager.initialize_components()
        
        # Start server components (without networking)
        await server_manager.server_module.start()
        if server_manager.server_trainer:
            await server_manager.server_trainer.start_training()
        
        # Process audio
        spectrogram = server_manager.audio_processor.extract_spectrogram(input_audio)
        result = await server_manager.server_module.process(spectrogram)
        
        # Stop components
        await server_manager.server_module.stop()
        if server_manager.server_trainer:
            await server_manager.server_trainer.stop_training()
        
        return result
        
    except Exception as e:
        server_manager.logger.error(f"Error in server-only inference: {e}")
        raise


async def run_distributed_training_demo(server_config_path: str, duration: int = 300):
    """
    Run a distributed training demonstration
    
    Args:
        server_config_path: Path to server configuration
        duration: Duration in seconds to run the demo
    """
    server_manager = ServerManager(server_config_path)
    
    try:
        # Load and setup
        server_manager.load_config()
        server_manager.setup_logging()
        server_manager.initialize_components()
        
        # Start server
        await server_manager.start_server()
        
        # Run for specified duration
        server_manager.logger.info(f"Running distributed training demo for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # In a real scenario, edge devices would be connecting and sending data
            # Here we can simulate some activity or just wait
            await asyncio.sleep(5.0)
            
            # Log current status
            if server_manager.server_trainer:
                metrics = server_manager.server_trainer.get_training_metrics()
                server_manager.logger.info(
                    f"Demo Progress - "
                    f"Active devices: {metrics.get('active_devices', 0)}, "
                    f"Embeddings processed: {metrics.get('total_embeddings_processed', 0)}, "
                    f"Loss: {metrics.get('average_loss', 0):.4f}"
                )
        
        server_manager.logger.info("Distributed training demo completed")
        
    except Exception as e:
        server_manager.logger.error(f"Error in distributed training demo: {e}")
        raise
    finally:
        # Clean shutdown
        await server_manager.stop_server()


def validate_server_config(config_path: str) -> bool:
    """
    Validate server configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = [
            'server', 'model', 'training', 'loss', 
            'aggregation', 'networking', 'monitoring'
        ]
        
        for section in required_sections:
            if section not in config:
                print(f"Error: Missing required section '{section}' in config")
                return False
        
        # Check critical parameters
        critical_params = [
            ('server', 'port'),
            ('model', 'embedding_dim'),
            ('training', 'learning_rate'),
            ('training', 'batch_size'),
            ('loss', 'sliced_wasserstein', 'num_projections'),
            ('loss', 'laplacian', 'k_neighbors')
        ]
        
        for param_path in critical_params:
            current = config
            for key in param_path:
                if key not in current:
                    print(f"Error: Missing parameter '{'.'.join(param_path)}' in config")
                    return False
                current = current[key]
        
        # Validate value ranges
        if config['server']['port'] <= 0 or config['server']['port'] > 65535:
            print(f"Error: Invalid port number {config['server']['port']}")
            return False
        
        if config['training']['learning_rate'] <= 0:
            print(f"Error: Invalid learning rate {config['training']['learning_rate']}")
            return False
        
        if config['training']['batch_size'] <= 0:
            print(f"Error: Invalid batch size {config['training']['batch_size']}")
            return False
        
        print("✓ Configuration validation passed")
        return True
        
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        return False
    except Exception as e:
        print(f"Error: Failed to validate configuration: {e}")
        return False


def print_server_info(config_path: str):
    """Print server configuration information"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("=== StreamSplit Server Configuration ===")
        print(f"Server ID: {config.get('server', {}).get('id', 'Unknown')}")
        print(f"Bind Address: {config.get('server', {}).get('bind_address', '0.0.0.0')}")
        print(f"Port: {config.get('server', {}).get('port', 8888)}")
        print(f"Max Connections: {config.get('server', {}).get('max_connections', 100)}")
        
        print("\n=== Model Configuration ===")
        print(f"Embedding Dimension: {config.get('model', {}).get('embedding_dim', 128)}")
        print(f"Intermediate Dimension: {config.get('model', {}).get('intermediate_dim', 256)}")
        print(f"Number of Layers: {config.get('model', {}).get('num_layers', 4)}")
        
        print("\n=== Training Configuration ===")
        print(f"Learning Rate: {config.get('training', {}).get('learning_rate', 5e-4)}")
        print(f"Batch Size: {config.get('training', {}).get('batch_size', 256)}")
        print(f"Scheduler: {config.get('training', {}).get('scheduler_type', 'cosine')}")
        
        print("\n=== Loss Configuration ===")
        sw_config = config.get('loss', {}).get('sliced_wasserstein', {})
        lap_config = config.get('loss', {}).get('laplacian', {})
        print(f"SW Projections: {sw_config.get('num_projections', 100)}")
        print(f"SW Weight: {sw_config.get('weight', 1.0)}")
        print(f"Laplacian K-neighbors: {lap_config.get('k_neighbors', 5)}")
        print(f"Laplacian Weight: {lap_config.get('weight', 0.5)}")
        
        print("\n=== Aggregation Configuration ===")
        agg_config = config.get('aggregation', {})
        print(f"Temporal Window: {agg_config.get('temporal_window', 30.0)}s")
        print(f"Min Devices per Update: {agg_config.get('min_devices_per_update', 2)}")
        print(f"Max Age Threshold: {agg_config.get('max_age_threshold', 60.0)}s")
        
        print("\n=== Optimization Configuration ===")
        opt_config = config.get('optimization', {})
        print(f"Use GPU: {opt_config.get('use_gpu', True)}")
        print(f"Mixed Precision: {opt_config.get('mixed_precision', True)}")
        print(f"Model Compilation: {opt_config.get('compile_model', False)}")
        
    except Exception as e:
        print(f"Error reading configuration: {e}")


if __name__ == "__main__":
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == 'validate':
            config_path = sys.argv[2] if len(sys.argv) > 2 else 'config/server_config.yaml'
            if validate_server_config(config_path):
                sys.exit(0)
            else:
                sys.exit(1)
        elif sys.argv[1] == 'info':
            config_path = sys.argv[2] if len(sys.argv) > 2 else 'config/server_config.yaml'
            print_server_info(config_path)
            sys.exit(0)
        elif sys.argv[1] == 'demo':
            config_path = sys.argv[2] if len(sys.argv) > 2 else 'config/server_config.yaml'
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 300
            
            async def run_demo():
                await run_distributed_training_demo(config_path, duration)
            
            asyncio.run(run_demo())
            sys.exit(0)
    
    # Run the server normally
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutdown complete.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)