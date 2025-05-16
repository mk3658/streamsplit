# StreamSplit: Theoretical Guarantees for Edge Audio Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](#)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Performance Highlights](#performance-highlights)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Comprehensive Examples](#comprehensive-examples)
- [Configuration](#configuration)
- [Research Paper](#research-paper)
- [API Reference](#api-reference)
- [Experiments and Evaluation](#experiments-and-evaluation)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

StreamSplit is a novel framework for real-time continuous audio representation learning at the edge. It addresses the limitations of centralized methods under extreme computational, memory, and bandwidth restrictions by providing a streaming contrastive learning approach with dynamic edge-server computation splitting.

**Key Innovations:**
- Streaming contrastive learning with theoretical convergence guarantees
- Dynamic computation splitting between edge and server
- Hybrid loss function combining Sliced-Wasserstein distance and Laplacian regularization
- Adaptive resource management for edge devices

## Architecture

![StreamSplit Framework](./images/architecture_diagram.png)

*Figure 1: End-to-end overview of StreamSplit showing the edge-server processing pipeline with dynamic computation splitting.*

### Key Components

#### 🔹 Edge Device Processing
- **Raw audio acquisition** with FFT & augmentations
- **Adaptive feature extraction** based on available resources
- **Device-side model execution** (Conv1→...→Convk)
- **Streaming memory bank** with contrastive learning
- **Local buffer and gradient queue** management

#### 🔹 Dynamic Split Decision
- **RL Agent** monitoring system resources and network conditions
- **Uncertainty Module** assessing embedding quality
- **Performance metrics** feedback loop
- **Real-time split point adjustment** within the deep encoder

#### 🔹 Server-Side Processing
- **Completion of model inference** (Convk+1→...→ConvL)
- **Hybrid loss computation** (Sliced-Wasserstein + Laplacian)
- **Global model updates** and weight synchronization
- **Embedding metadata** management

## Key Features

### 🎯 Core Capabilities
- **Streaming Contrastive Framework**: Learn from embedding distributions with theoretical convergence guarantees
- **Hybrid Loss Function**: Combines Sliced-Wasserstein distance with Laplacian regularization
- **Dynamic Computation Splitting**: Adaptively distributes workload between edge and server based on resources
- **Theoretical Guarantees**: Proven convergence bounds for both convex and non-convex cases

### ⚡ Edge Optimizations
- **Optimized for resource-constrained devices** like Raspberry Pi
- **Adaptive resolution processing** based on CPU/memory availability
- **Efficient FFT implementation** using KissFFT
- **Memory bank with Distribution-Aware Sampling**
- **Local contrastive learning** with age-weighted negatives

### 🌐 Network Efficiency
- **Selective transmission** based on uncertainty estimation
- **Compression and batching** for bandwidth optimization
- **Network condition adaptation**
- **Fault-tolerant communication**

## Performance Highlights

| Metric | StreamSplit Performance | Improvement |
|--------|------------------------|-------------|
| **Accuracy** | 97.8% of server-only | Within 2% gap |
| **Bandwidth Reduction** | 77.1% less | vs server-only |
| **Latency Reduction** | 72.6% lower | vs server-only |
| **Energy Savings** | 52.3% reduction | vs edge-only |

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/mk3658/streamsplit.git
cd streamsplit

# Install dependencies
pip install -r requirements.txt

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### System Requirements

**For Edge Devices:**
- Python 3.8+
- PyTorch 1.9+
- Raspberry Pi 4B (4GB RAM) or equivalent
- USB microphone
- Wi-Fi/cellular connectivity (0.5+ Mbps)

**For Server:**
- Python 3.8+
- PyTorch 1.9+ (CUDA recommended)
- Multi-core CPU or GPU
- 16GB+ RAM
- High-speed network connection

### Package Dependencies

<details>
<summary>Core Dependencies</summary>

```bash
torch>=1.9.0
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
scikit-learn>=1.0.0
PyYAML>=5.4.0
dataclasses-json>=0.5.7
```
</details>

<details>
<summary>Edge Device Dependencies</summary>

```bash
psutil>=5.8.0
websockets>=10.0
asyncio
# For audio recording:
pyaudio>=0.2.11
soundfile>=0.10.0
```
</details>

## Quick Start

### 1. Basic Framework Usage

```python
import asyncio
import numpy as np
from src.core.streamsplit import StreamSplitFramework, StreamSplitConfig, StreamSplitMode

# Configure framework
config = StreamSplitConfig(
    device_id="edge_device_1",
    device_type="raspberry_pi_4b",
    sample_rate=16000,
    n_mels=128,
    embedding_dim=128,
    learning_rate=1e-4,
    memory_bank_size_min=64,
    memory_bank_size_max=512
)

# Initialize and start framework
framework = StreamSplitFramework(config)

async def run_example():
    await framework.start(mode=StreamSplitMode.DYNAMIC_SPLIT)
    
    # Process audio (example with synthetic audio)
    audio_data = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
    result = await framework.process_audio_stream(audio_data)
    
    print(f"Processing mode: {result['mode']}")
    print(f"Latency: {result['latency']:.2f}ms")
    print(f"Split point: {result['split_point']}")
    
    await framework.stop()

# Run the example
asyncio.run(run_example())
```

### 2. Edge-Only Processing

```python
from src.core.edge_module import EdgeModule
from src.training.edge_trainer import EdgeTrainer, TrainingConfig
from src.models.encoders import MobileNetV3EdgeEncoder

# Create edge model
edge_model = MobileNetV3EdgeEncoder(
    input_dim=128,
    embedding_dim=128,
    width_multiplier=0.75
)

# Configure edge trainer
config = TrainingConfig(
    learning_rate=1e-4,
    temperature=0.1,
    momentum=0.999,
    batch_size=32
)

trainer = EdgeTrainer(config, edge_model, device='cpu')

# Process audio spectrogram
spectrogram = torch.randn(128, 64)  # Mel spectrogram
result = trainer.train_step(spectrogram)

print(f"Loss: {result['loss']['total_loss']:.4f}")
print(f"Memory bank size: {result['memory_bank_stats']['current_size']}")
```

### 3. Server-Side Processing

```python
from src.core.server_module import ServerModule
from src.training.server_trainer import ServerTrainer, ServerTrainingConfig

# Configure server trainer
config = ServerTrainingConfig(
    learning_rate=5e-4,
    batch_size=256,
    sw_projections=100,
    laplacian_k_neighbors=5,
    laplacian_weight=0.5
)

# Initialize and start
trainer = ServerTrainer(config, server_model, device='cuda')

async def run_server():
    await trainer.start_training()
    
    # Process embeddings from edge
    embeddings_data = {
        'embeddings': torch.randn(128),
        'timestamps': [time.time()],
        'metadata': {'device_id': 'edge_1', 'split_point': 8}
    }
    
    await trainer.receive_embeddings('edge_1', embeddings_data)
    
    metrics = trainer.get_training_metrics()
    print(f"Average loss: {metrics['average_loss']:.4f}")
```

## Comprehensive Examples

### Running the System

#### Edge Device
```bash
# Run edge device with default configuration
python scripts/run_edge.py --config config/edge_config.yaml

# Run with custom parameters
python scripts/run_edge.py \
    --config config/edge_config.yaml \
    --device-id edge_device_1 \
    --log-level INFO \
    --synthetic  # Use synthetic audio for testing

# Edge-only mode
python scripts/run_edge.py \
    --config config/edge_config.yaml \
    --no-splitting
```

#### Server
```bash
# Run server with default configuration
python scripts/run_server.py --config config/server_config.yaml

# Run with GPU acceleration
python scripts/run_server.py \
    --config config/server_config.yaml \
    --gpu

# CPU-only mode
python scripts/run_server.py \
    --config config/server_config.yaml \
    --cpu-only
```

### Dataset Preparation

#### AudioSet Preparation
```python
from src.data.audioset_preparation import AudioSetPreparer, AudioSetConfig

config = AudioSetConfig(
    output_dir="/data/audioset",
    subset_size=10000,
    duration=10.0,
    sample_rate=16000
)

preparer = AudioSetPreparer(config)
processed_segments = preparer.prepare_dataset()
```

#### On-Device Recording
```python
from src.data.ondevice_preparation import OnDeviceDatasetPreparer, OnDeviceConfig

config = OnDeviceConfig(
    output_dir="/data/ondevice",
    classes=["Conversation", "Kitchen Activities", "Traffic Noise"],
    duration=10.0,
    recordings_per_class=100
)

preparer = OnDeviceDatasetPreparer(config)
results = preparer.prepare_dataset(record_new=True)
```

### Evaluation and Experiments

```bash
# Run comprehensive evaluation
python scripts/evaluate.py \
    --config config/edge_config.yaml \
    --output-dir results \
    --model-path models/streamsplit_model.pth

# Evaluate specific datasets
python scripts/evaluate.py \
    --audioset-path data/audioset_subset \
    --ondevice-path data/ondevice_recordings \
    --output-dir results \
    --plot  # Generate visualizations
```

## Configuration

StreamSplit uses YAML configuration files for easy customization.

### Edge Configuration

Key sections in `config/edge_config.yaml`:

```yaml
# Device and audio settings
device:
  id: "edge_device_1"
  type: "raspberry_pi_4b"

audio:
  sample_rate: 16000
  n_mels: 128
  window_duration: 0.025

# Training parameters
training:
  learning_rate: 1e-4
  batch_size: 32
  temperature: 0.1

# Memory bank configuration  
memory_bank:
  min_size: 64
  max_size: 512

# Dynamic splitting
splitting:
  enabled: true
  reward_weights:
    accuracy: 1.0
    resource_usage: -0.5
    latency: -0.3
    privacy_risk: -0.2
```

### Server Configuration

Key sections in `config/server_config.yaml`:

```yaml
# Server settings
server:
  bind_address: "0.0.0.0"
  port: 8888

# Training parameters
training:
  learning_rate: 5e-4
  batch_size: 256

# Hybrid loss configuration
loss:
  sliced_wasserstein:
    num_projections: 100
    weight: 1.0
  laplacian:
    k_neighbors: 5
    weight: 0.5

# Aggregation settings
aggregation:
  temporal_window: 30.0
  min_devices_per_update: 2
```

For complete configuration options, see:
- [`config/edge_config.yaml`](config/edge_config.yaml)
- [`config/server_config.yaml`](config/server_config.yaml)

## Research Paper

This implementation accompanies our research submission:

**"StreamSplit: Theoretical Guarantees for Edge Audio Learning"**  
*Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025)*

### Key Contributions

1. **Streaming Contrastive Framework**: Novel approach with convergence guarantees for small-batch edge learning
2. **Hybrid Loss Function**: Sliced-Wasserstein + Laplacian regularization for robust representations
3. **Dynamic Computation Splitting**: Adaptive workload distribution based on resources and network
4. **Comprehensive Evaluation**: Extensive experiments on AudioSet and real-world edge deployments

### Theoretical Results

**Convergence Guarantee (Theorem 1):**
- Parameter convergence: `E[||f_T - f*||²] ≤ C₁/T + C₂ε²`
- Loss convergence: `E[|L(f_T) - L(f*)|] ≤ Lσ²/(2μ²T) + Lε²/2`

Where T is iterations, ε is approximation error from splitting, and σ² is gradient estimate variance.

## API Reference

<details>
<summary>Core Framework Classes</summary>

### `StreamSplitFramework`
```python
class StreamSplitFramework:
    def __init__(self, config: StreamSplitConfig)
    async def start(self, mode: StreamSplitMode)
    async def stop()
    async def process_audio_stream(self, audio: np.ndarray) -> Dict[str, Any]
    def save_checkpoint(self, filepath: str)
    def load_checkpoint(self, filepath: str)
```

### `EdgeModule`
```python
class EdgeModule:
    async def process(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]
    async def process_partial(self, spectrogram: torch.Tensor, split_point: int) -> Dict[str, Any]
    def get_resource_state(self) -> Dict[str, float]
```

### `ServerModule`
```python
class ServerModule:
    async def process(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]
    async def receive_edge_embedding(self, embedding_data: Dict[str, Any]) -> bool
    async def process_continuation(self, edge_output: Dict[str, Any], split_point: int) -> Dict[str, Any]
```
</details>

For detailed API documentation, see [docs/api_reference.md](docs/api_reference.md).

## Experiments and Evaluation

### Performance Metrics

StreamSplit provides comprehensive evaluation metrics:

- **Representation Quality**: Classification accuracy, Precision@K, embedding separability
- **Resource Efficiency**: CPU usage, memory consumption, energy efficiency  
- **Communication Efficiency**: Bandwidth reduction, transmission analysis
- **Adaptation Performance**: Split decision quality, response time

### Baseline Comparisons

Results compared against:
- **Edge-only**: MobileNetV3-Small with local learning
- **Server-only**: ResNet50 with centralized training
- **FedCL**: Federated contrastive learning
- **FSL**: Fixed split learning
- **LEO**: Lightweight edge optimization

### Running Experiments

```bash
# Reproducing paper results
python experiments/run_full_evaluation.py \
    --dataset audioset \
    --devices 5 \
    --output results/

# Custom experiments
python experiments/custom_experiment.py \
    --config experiments/audioset/config.yaml \
    --baseline fedcl \
    --devices 3
```

## Hardware Requirements

### Edge Device Specifications (Minimum)

| Component | Requirement |
|-----------|-------------|
| **CPU** | Quad-core ARM Cortex-A72 1.5GHz |
| **Memory** | 4GB RAM |
| **Storage** | 8GB for model and buffers |
| **Network** | Wi-Fi or cellular (0.5+ Mbps) |
| **Audio** | USB microphone |

### Server Requirements

| Component | Recommendation |
|-----------|---------------|
| **CPU** | Multi-core x86_64 or ARM64 |
| **Memory** | 16GB+ RAM |
| **GPU** | Optional (CUDA-compatible) |
| **Storage** | SSD for embedding database |
| **Network** | High-bandwidth (10+ Mbps) |

## Troubleshooting

### Common Issues

<details>
<summary>Import Errors</summary>

**Problem**: `ModuleNotFoundError` when importing StreamSplit modules

**Solution**: Add the src directory to your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```
</details>

<details>
<summary>CUDA Out of Memory</summary>

**Problem**: GPU memory errors on server

**Solutions**:
1. Reduce batch sizes in configuration files
2. Use CPU-only mode: `python scripts/run_server.py --cpu-only`
3. Enable gradient accumulation in edge config
</details>

<details>
<summary>Audio Processing Issues</summary>

**Problem**: Audio recording/processing errors

**Solutions**:
1. Install audio dependencies: `pip install pyaudio librosa soundfile`
2. Check audio device permissions
3. Use synthetic audio mode for testing: `--synthetic`
</details>

<details>
<summary>Network Connection Issues</summary>

**Problem**: Edge device cannot connect to server

**Solutions**:
1. Check firewall settings (port 8888)
2. Verify server address in edge config
3. Test with local server: `--server-host localhost`
</details>

### Performance Optimization

**Edge Device Optimization:**
- Enable reduced resolution mode for high CPU usage
- Adjust memory bank size based on available RAM
- Use compression for network transmission

**Server Optimization:**
- Use GPU acceleration when available
- Tune batch sizes for optimal throughput
- Enable mixed precision training

**Network Optimization:**
- Adjust uncertainty thresholds for transmission decisions
- Use adaptive bandwidth control
- Enable compression for slow connections

## Acknowledgments

- Built with PyTorch and optimized for Raspberry Pi deployment
- Uses KissFFT library for efficient edge processing
- Inspired by recent advances in contrastive learning and edge computing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support and Contact

- 📧 **Email**: [m.quan@deakin.edu.au](mailto:m.quan@deakin.edu.au)
- 📖 **Documentation**: [docs/](docs/)

---

**StreamSplit** - Enabling efficient audio AI at the edge with theoretical guarantees.

*Made with ❤️ for the edge computing and audio ML communities*
