#!/usr/bin/env python3
"""
StreamSplit Edge Device Runner
Implements the complete edge device functionality as described in the StreamSplit paper
Based on Section 3.1 and related components
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

# StreamSplit imports
from src.core.streamsplit import StreamSplitFramework, StreamSplitConfig, StreamSplitMode
from src.core.edge_module import EdgeModule
from src.core.dynamic_splitting import DynamicSplittingAgent
from src.training.edge_trainer import EdgeTrainer, TrainingConfig
from src.models.encoders import MobileNetV3EdgeEncoder
from src.utils.audio_processing import AudioProcessor, AudioConfig
from src.utils.metrics import MetricsCollector
from src.training.distributed_training import DistributedCoordinator, DistributedConfig

class EdgeDeviceManager:
    """
    Main manager for StreamSplit edge device
    Coordinates all components and handles lifecycle management
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.logger = None
        
        # Core components
        self.streamsplit_framework = None
        self.edge_module = None
        self.edge_trainer = None
        self.splitting_agent = None
        self.audio_processor = None
        self.metrics_collector = None
        self.distributed_coordinator = None
        
        # State management
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.start_time = None
        self.processed_samples = 0
        
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
        log_file = self.config.get('monitoring', {}).get('log_file', 'logs/edge_device.log')
        
        # Create logs directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EdgeDeviceManager initialized with config: {self.config_path}")
    
    def initialize_components(self):
        """Initialize all StreamSplit components"""
        try:
            self.logger.info("Initializing StreamSplit components...")
            
            # 1. Audio Processor (Section 3.1.1)
            self._initialize_audio_processor()
            
            # 2. Edge Encoder Model
            self._initialize_edge_model()
            
            # 3. Edge Module (Memory bank, contrastive learning)
            self._initialize_edge_module()
            
            # 4. Edge Trainer
            self._initialize_edge_trainer()
            
            # 5. Dynamic Splitting Agent (Section 3.3)
            self._initialize_splitting_agent()
            
            # 6. Metrics Collector
            self._initialize_metrics_collector()
            
            # 7. Distributed Coordinator (if enabled)
            self._initialize_distributed_coordinator()
            
            # 8. StreamSplit Framework
            self._initialize_streamsplit_framework()
            
            self.logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize components: {e}")
            self.logger.error(traceback.format_exc())
            sys.exit(1)
    
    def _initialize_audio_processor(self):
        """Initialize audio processing pipeline"""
        audio_config = AudioConfig(
            sample_rate=self.config['audio']['sample_rate'],
            window_duration=self.config['audio']['window_duration'],
            hop_length=self.config['audio']['hop_length'],
            n_fft=self.config['audio']['n_fft'],
            n_mels=self.config['audio']['n_mels'],
            fmin=self.config['audio']['fmin'],
            fmax=self.config['audio'].get('fmax'),
            power=self.config['audio']['power'],
            normalized=self.config['audio']['normalized'],
            reduced_freq_factor=self.config['audio']['reduced_freq_factor'],
            reduced_stride_factor=self.config['audio']['reduced_stride_factor']
        )
        
        self.audio_processor = AudioProcessor(audio_config)
        self.logger.info("✓ Audio processor initialized")
    
    def _initialize_edge_model(self):
        """Initialize edge encoder model"""
        # Detect device (CPU/GPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Create edge encoder
        self.edge_model = MobileNetV3EdgeEncoder(
            input_dim=self.config['audio']['n_mels'],
            embedding_dim=self.config['model']['embedding_dim'],
            width_multiplier=self.config['model']['width_multiplier']
        ).to(device)
        
        self.logger.info(f"✓ Edge model initialized on {device}")
    
    def _initialize_edge_module(self):
        """Initialize edge module with memory bank and contrastive learning"""
        # Create a config-like object for EdgeModule
        class EdgeConfig:
            def __init__(self, config_dict):
                # Audio parameters
                self.sample_rate = config_dict['audio']['sample_rate']
                self.n_mels = config_dict['audio']['n_mels']
                
                # Training parameters
                self.temperature = config_dict['training']['temperature']
                self.momentum = config_dict['training']['momentum']
                self.learning_rate = config_dict['training']['learning_rate']
                
                # Memory bank parameters
                self.memory_bank_size_min = config_dict['memory_bank']['min_size']
                self.memory_bank_size_max = config_dict['memory_bank']['max_size']
                
                # Resource thresholds
                self.cpu_threshold = config_dict['resources']['cpu_threshold']
        
        edge_config = EdgeConfig(self.config)
        self.edge_module = EdgeModule(edge_config)
        self.logger.info("✓ Edge module initialized")
    
    def _initialize_edge_trainer(self):
        """Initialize edge trainer for local learning"""
        trainer_config = TrainingConfig(
            learning_rate=self.config['training']['learning_rate'],
            momentum=self.config['training']['momentum'],
            temperature=self.config['training']['temperature'],
            consistency_weight=self.config['training']['consistency_weight'],
            batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            memory_bank_min_size=self.config['memory_bank']['min_size'],
            memory_bank_max_size=self.config['memory_bank']['max_size'],
            age_decay_factor=self.config['training']['age_decay_factor'],
            resource_threshold=self.config['resources']['cpu_threshold'],
            das_components=self.config['memory_bank']['das']['n_components'],
            das_alpha=self.config['memory_bank']['das']['alpha'],
            das_learning_rate=self.config['memory_bank']['das']['learning_rate']
        )
        
        self.edge_trainer = EdgeTrainer(trainer_config, self.edge_model, device=str(self.device))
        self.logger.info("✓ Edge trainer initialized")
    
    def _initialize_splitting_agent(self):
        """Initialize dynamic splitting agent"""
        if not self.config['splitting']['enabled']:
            self.logger.info("Dynamic splitting disabled")
            return
        
        # Create a config-like object for splitting agent
        class SplittingConfig:
            def __init__(self, config_dict):
                self.split_agent_lr = config_dict['splitting']['agent']['learning_rate']
                self.split_reward_weights = config_dict['splitting']['agent']['reward_weights']
                self.bandwidth_range = config_dict.get('communication', {}).get('bandwidth_range', [0.5, 8.0])
                self.latency_range = config_dict.get('communication', {}).get('latency_range', [50, 200])
        
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
        """Initialize distributed training coordinator if enabled"""
        if not self.config.get('communication', {}).get('selective_transmission', False):
            self.logger.info("Distributed coordination disabled")
            return
        
        # Create distributed config
        distributed_config = DistributedConfig(
            edge_id=self.config['device']['id'],
            server_host=self.config['communication']['server_host'],
            server_port=self.config['communication']['server_port'],
            sync_protocol='adaptive_sync',
            training_mode='dynamic_split'
        )
        
        # Create edge config for distributed coordinator
        edge_config = TrainingConfig()  # Use defaults, will be overridden
        
        # Create dummy server config (not used on edge)
        class ServerConfig:
            pass
        
        server_config = ServerConfig()
        
        self.distributed_coordinator = DistributedCoordinator(
            distributed_config, edge_config, server_config, is_server=False
        )
        self.logger.info("✓ Distributed coordinator initialized")
    
    def _initialize_streamsplit_framework(self):
        """Initialize main StreamSplit framework"""
        # Create StreamSplit config
        streamsplit_config = StreamSplitConfig(
            device_id=self.config['device']['id'],
            device_type=self.config['device']['type'],
            sample_rate=self.config['audio']['sample_rate'],
            window_duration=self.config['audio']['window_duration'],
            hop_length=self.config['audio']['hop_length'],
            n_fft=self.config['audio']['n_fft'],
            n_mels=self.config['audio']['n_mels'],
            batch_size=self.config['training']['batch_size'],
            learning_rate=self.config['training']['learning_rate'],
            momentum=self.config['training']['momentum'],
            temperature=self.config['training']['temperature'],
            memory_bank_size_min=self.config['memory_bank']['min_size'],
            memory_bank_size_max=self.config['memory_bank']['max_size'],
            uncertainty_threshold_base=self.config['uncertainty']['base_threshold'],
            transmission_threshold=self.config['uncertainty']['base_threshold'],
            resource_monitor_interval=self.config['resources']['monitor_interval'],
            cpu_threshold=self.config['resources']['cpu_threshold'],
            bandwidth_range=tuple(self.config.get('communication', {}).get('bandwidth_range', [0.5, 8.0])),
            latency_range=tuple(self.config.get('communication', {}).get('latency_range', [50, 200])),
            split_agent_lr=self.config['splitting']['agent']['learning_rate'],
            split_reward_weights=self.config['splitting']['agent']['reward_weights']
        )
        
        self.streamsplit_framework = StreamSplitFramework(streamsplit_config)
        self.logger.info("✓ StreamSplit framework initialized")
    
    async def start_device(self):
        """Start the edge device and all its components"""
        try:
            self.logger.info("Starting StreamSplit edge device...")
            self.start_time = time.time()
            self.is_running = True
            
            # Start metrics collection
            if self.metrics_collector:
                self.metrics_collector.start_monitoring()
                self.logger.info("✓ Metrics collection started")
            
            # Start edge module
            await self.edge_module.start()
            self.logger.info("✓ Edge module started")
            
            # Start splitting agent
            if self.splitting_agent:
                await self.splitting_agent.start()
                self.logger.info("✓ Splitting agent started")
            
            # Start distributed coordinator
            if self.distributed_coordinator:
                await self.distributed_coordinator.start(
                    edge_trainer=self.edge_trainer,
                    splitting_agent=self.splitting_agent
                )
                self.logger.info("✓ Distributed coordinator started")
            
            # Determine training mode
            mode = StreamSplitMode.DYNAMIC_SPLIT
            if not self.config['splitting']['enabled']:
                mode = StreamSplitMode.EDGE_ONLY
            
            # Start StreamSplit framework
            await self.streamsplit_framework.start(mode=mode)
            self.logger.info(f"✓ StreamSplit framework started in {mode.value} mode")
            
            # Load checkpoint if available
            await self._load_checkpoint_if_available()
            
            self.logger.info("🚀 Edge device fully operational!")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to start device: {e}")
            self.logger.error(traceback.format_exc())
            await self.stop_device()
            sys.exit(1)
    
    async def _load_checkpoint_if_available(self):
        """Load checkpoint if checkpointing is enabled and checkpoint exists"""
        if not self.config.get('checkpointing', {}).get('enabled', False):
            return
        
        checkpoint_dir = Path(self.config['checkpointing']['checkpoint_dir'])
        checkpoint_files = list(checkpoint_dir.glob('*.pt'))
        
        if checkpoint_files:
            # Load most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            try:
                self.streamsplit_framework.load_checkpoint(str(latest_checkpoint))
                self.logger.info(f"✓ Loaded checkpoint: {latest_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint {latest_checkpoint}: {e}")
    
    async def stop_device(self):
        """Stop the edge device and cleanup"""
        self.logger.info("Stopping StreamSplit edge device...")
        self.is_running = False
        
        try:
            # Save checkpoint before shutdown
            await self._save_checkpoint()
            
            # Stop StreamSplit framework
            if self.streamsplit_framework:
                await self.streamsplit_framework.stop()
                self.logger.info("✓ StreamSplit framework stopped")
            
            # Stop distributed coordinator
            if self.distributed_coordinator:
                await self.distributed_coordinator.stop()
                self.logger.info("✓ Distributed coordinator stopped")
            
            # Stop splitting agent
            if self.splitting_agent:
                await self.splitting_agent.stop()
                self.logger.info("✓ Splitting agent stopped")
            
            # Stop edge module
            if self.edge_module:
                await self.edge_module.stop()
                self.logger.info("✓ Edge module stopped")
            
            # Stop metrics collection
            if self.metrics_collector:
                self.metrics_collector.stop_monitoring()
                self.logger.info("✓ Metrics collection stopped")
            
            # Cleanup edge trainer
            if self.edge_trainer:
                self.edge_trainer.shutdown()
                self.logger.info("✓ Edge trainer cleanup complete")
            
            # Print final statistics
            self._print_final_statistics()
            
            self.logger.info("✓ Edge device shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _save_checkpoint(self):
        """Save checkpoint if checkpointing is enabled"""
        if not self.config.get('checkpointing', {}).get('enabled', False):
            return
        
        checkpoint_dir = Path(self.config['checkpointing']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        checkpoint_path = checkpoint_dir / f"edge_checkpoint_{timestamp}.pt"
        
        try:
            self.streamsplit_framework.save_checkpoint(str(checkpoint_path))
            self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(checkpoint_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints based on retention policy"""
        max_checkpoints = self.config['checkpointing'].get('max_checkpoints', 5)
        checkpoint_files = sorted(
            checkpoint_dir.glob('edge_checkpoint_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoint_files[max_checkpoints:]:
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
    
    def _print_final_statistics(self):
        """Print final performance statistics"""
        if self.start_time:
            runtime = time.time() - self.start_time
            self.logger.info("=== Edge Device Statistics ===")
            self.logger.info(f"Runtime: {runtime:.2f} seconds")
            self.logger.info(f"Samples processed: {self.processed_samples}")
            if runtime > 0:
                self.logger.info(f"Processing rate: {self.processed_samples/runtime:.2f} samples/sec")
            
            # Get performance summary from framework
            if self.streamsplit_framework:
                summary = self.streamsplit_framework.get_performance_summary()
                for key, value in summary.items():
                    self.logger.info(f"{key}: {value}")
    
    async def run_main_loop(self):
        """Main execution loop for the edge device"""
        self.logger.info("Starting main execution loop...")
        
        # Setup periodic tasks
        tasks = []
        
        # Periodic checkpoint saving
        if self.config.get('checkpointing', {}).get('enabled', False):
            interval = self.config['checkpointing']['save_interval']
            tasks.append(asyncio.create_task(
                self._periodic_checkpoint_task(interval)
            ))
        
        # Metrics export
        if self.config.get('monitoring', {}).get('export_metrics', False):
            interval = self.config['monitoring']['export_interval']
            tasks.append(asyncio.create_task(
                self._periodic_metrics_export(interval)
            ))
        
        # Performance monitoring
        tasks.append(asyncio.create_task(self._performance_monitoring_task()))
        
        try:
            # Main processing loop would go here
            # In a real implementation, this would process incoming audio streams
            self.logger.info("Edge device running... (Press Ctrl+C to stop)")
            
            # Simulate audio processing (in real deployment, this would be actual audio)
            await self._simulate_audio_processing()
            
        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _periodic_checkpoint_task(self, interval: float):
        """Periodic checkpoint saving task"""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                if self.is_running:
                    await self._save_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in checkpoint task: {e}")
    
    async def _periodic_metrics_export(self, interval: float):
        """Periodic metrics export task"""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                if self.is_running and self.metrics_collector:
                    # Export metrics
                    export_path = self.config['monitoring']['export_path']
                    export_format = self.config['monitoring']['export_format']
                    
                    # Create directory if needed
                    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Export with timestamp
                    timestamp = int(time.time())
                    base_path = Path(export_path)
                    timestamped_path = base_path.parent / f"{base_path.stem}_{timestamp}{base_path.suffix}"
                    
                    self.metrics_collector.export_metrics(str(timestamped_path), export_format)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics export task: {e}")
    
    async def _performance_monitoring_task(self):
        """Monitor and log performance metrics"""
        log_interval = 30.0  # Log every 30 seconds
        
        while self.is_running:
            try:
                await asyncio.sleep(log_interval)
                
                if self.is_running:
                    # Get current metrics
                    if self.edge_trainer:
                        metrics = self.edge_trainer.get_training_metrics()
                        self.logger.info(f"Training metrics - Loss: {metrics.get('mean_loss', 0):.4f}, "
                                       f"Memory bank: {metrics.get('memory_bank_utilization', 0):.2%}, "
                                       f"CPU: {metrics.get('current_resources', {}).get('cpu', 0):.1%}")
                    
                    if self.splitting_agent:
                        agent_metrics = self.splitting_agent.get_performance_metrics()
                        self.logger.info(f"Splitting metrics - Avg reward: {agent_metrics.get('average_reward', 0):.3f}, "
                                       f"Episodes: {agent_metrics.get('episodes_completed', 0)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    async def _simulate_audio_processing(self):
        """Simulate audio processing for testing purposes"""
        if not self.config.get('debug', {}).get('synthetic_audio', False):
            # In real deployment, this would wait for actual audio input
            await self.shutdown_event.wait()
            return
        
        self.logger.info("Running in synthetic audio mode for testing...")
        
        # Generate synthetic audio data for testing
        sample_rate = self.config['audio']['sample_rate']
        duration = 1.0  # 1 second chunks
        
        while self.is_running:
            try:
                # Generate synthetic audio (1 second of 440Hz sine wave)
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
                
                # Process through StreamSplit
                result = await self.streamsplit_framework.process_audio_stream(audio_data)
                
                self.processed_samples += len(audio_data)
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in audio processing simulation: {e}")
                await asyncio.sleep(1.0)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for the edge device"""
    parser = argparse.ArgumentParser(description='StreamSplit Edge Device')
    parser.add_argument('--config', type=str, default='config/edge_config.yaml',
                       help='Path to edge configuration file')
    parser.add_argument('--device-id', type=str, help='Override device ID from config')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Override log level from config')
    parser.add_argument('--synthetic', action='store_true',
                       help='Run with synthetic audio for testing')
    parser.add_argument('--no-splitting', action='store_true',
                       help='Disable dynamic splitting (edge-only mode)')
    
    args = parser.parse_args()
    
    # Initialize edge device manager
    edge_manager = EdgeDeviceManager(args.config)
    
    try:
        # Load configuration
        edge_manager.load_config()
        
        # Apply command line overrides
        if args.device_id:
            edge_manager.config['device']['id'] = args.device_id
        
        if args.log_level:
            edge_manager.config.setdefault('monitoring', {})['log_level'] = args.log_level
        
        if args.synthetic:
            edge_manager.config.setdefault('debug', {})['synthetic_audio'] = True
        
        if args.no_splitting:
            edge_manager.config['splitting']['enabled'] = False
        
        # Setup logging
        edge_manager.setup_logging()
        
        # Setup signal handlers
        edge_manager.setup_signal_handlers()
        
        # Initialize all components
        edge_manager.initialize_components()
        
        # Start the device
        await edge_manager.start_device()
        
        # Run main loop
        await edge_manager.run_main_loop()
        
    except KeyboardInterrupt:
        edge_manager.logger.info("Received keyboard interrupt")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean shutdown
        await edge_manager.stop_device()


if __name__ == "__main__":
    # Run the edge device
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)