# StreamSplit: Theoretical Guarantees for Edge Audio Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Overview

StreamSplit is a novel framework for real-time continuous audio representation learning at the edge. It addresses the challenges of resource-constrained edge devices by providing:

- **Streaming Contrastive Learning**: Edge-friendly contrastive learning with theoretical convergence guarantees
- **Hybrid Loss Function**: Combines Sliced-Wasserstein distance with Laplacian regularization for robust representation quality
- **Dynamic Computation Splitting**: Adaptive workload distribution between edge devices and servers using reinforcement learning
- **Resource-Aware Processing**: Handles varying network and computational constraints

## 🚀 Key Features

### 📊 Performance Highlights
- **97.8%** of server-only accuracy maintained
- **77.1%** bandwidth reduction vs server-only processing
- **72.6%** latency reduction
- **52.3%** energy consumption reduction

### 🔬 Theoretical Guarantees
- Proven convergence bounds for distributed learning
- Support for both strongly convex and non-convex loss functions
- Approximation error bounds for computation splitting

## 🏗️ Architecture

```
Edge Device                          Server
├── Audio Acquisition               ├── Embedding Aggregation
├── Feature Extraction              ├── Distribution Alignment
├── Streaming Contrastive Learning  ├── Hybrid Loss Computation
├── Uncertainty Estimation          ├── Model Refinement
└── Selective Transmission          └── Weight Distribution
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- psutil
- asyncio
- pyyaml

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/streamsplit.git
cd streamsplit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install StreamSplit in development mode
pip install -e .
```

## 🚦 Quick Start

### Edge Device Setup

```python
from streamsplit import StreamSplitEdge
from streamsplit.config import EdgeConfig

# Load configuration
config = EdgeConfig.from_file('config/edge_config.yaml')

# Initialize edge module
edge = StreamSplitEdge(config)

# Start processing
asyncio.run(edge.start_streaming())
```

### Server Setup

```python
from streamsplit import StreamSplitServer
from streamsplit.config import ServerConfig

# Load configuration  
config = ServerConfig.from_file('config/server_config.yaml')

# Initialize server
server = StreamSplitServer(config)

# Start server
asyncio.run(server.start())
```

## 📁 Project Structure

```
streamsplit/
├── src/
│   ├── core/                    # Core framework implementation
│   │   ├── streamsplit.py      # Main StreamSplit class
│   │   ├── edge_module.py      # Edge processing module
│   │   ├── server_module.py    # Server aggregation module
│   │   └── dynamic_splitting.py # RL-based splitting agent
│   ├── models/                  # Neural network architectures
│   │   ├── encoders.py         # MobileNetV3-Small encoder
│   │   ├── losses.py           # SW + Laplacian hybrid loss
│   │   └── transforms.py       # Dynamic model transformations
│   ├── utils/                   # Utility functions
│   │   ├── audio_processing.py # Audio preprocessing
│   │   ├── data_utils.py       # Data handling utilities
│   │   └── metrics.py          # Evaluation metrics
│   └── training/                # Training procedures
│       ├── edge_trainer.py     # Edge training logic
│       ├── server_trainer.py   # Server training logic
│       └── distributed_training.py # Distributed coordination
├── config/                      # Configuration files
│   ├── edge_config.yaml        # Edge device configuration
│   └── server_config.yaml      # Server configuration
├── scripts/                     # Utility scripts
│   ├── run_edge.py             # Run edge device
│   ├── run_server.py           # Run server
│   ├── simulate_network.py     # Network simulation
│   └── evaluate.py             # Evaluation script
├── experiments/                 # Experiment configurations
│   ├── audioset/               # AudioSet experiments
│   └── ondevice/               # On-device experiments
└── docs/                        # Documentation
    ├── api_reference.md        # API documentation
    ├── architecture.md         # Architecture details
    └── theory.md               # Theoretical analysis
```

## 🔧 Configuration

### Edge Configuration (`config/edge_config.yaml`)

```yaml
device:
  type: "raspberry_pi_4b"
  cpu_threshold: 0.7
  memory_limit: 1024  # MB
  audio_device: "default"

audio:
  sample_rate: 16000
  fft_size: 512
  hop_length: 256
  n_mels: 128

model:
  encoder: "mobilenet_v3_small"
  width_mult: 0.75
  embedding_dim: 128

contrastive:
  temperature: 0.1
  memory_bank_size: [64, 512]  # [min, max]
  negative_sampling: "distribution_aware"
  
optimization:
  learning_rate: 1e-4
  momentum: 0.999
  gradient_accumulation: 4
  
uncertainty:
  threshold: 0.5
  weights: [0.4, 0.3, 0.3]  # consistency, entropy, prototype
```

### Server Configuration (`config/server_config.yaml`)

```yaml
server:
  address: "0.0.0.0"
  port: 8888
  max_connections: 100

model:
  encoder: "mobilenet_v3_small"
  embedding_dim: 128
  
aggregation:
  batch_size: 256
  update_frequency: 10
  temporal_window: 30

loss:
  sliced_wasserstein:
    num_projections: 100
    weight: 1.0
  laplacian:
    k_neighbors: 5
    weight: 0.5
    
optimization:
  learning_rate: 5e-4
  scheduler: "cosine"
  weight_decay: 1e-6

splitting:
  algorithm: "ppo"
  state_dim: 25
  action_dim: 10
  reward_weights: [0.3, 0.2, 0.2, 0.3]  # accuracy, resource, latency, privacy
```

## 📊 Experimental Results

### AudioSet Evaluation

| Method | Accuracy (%) | Bandwidth (MB/h) | Latency (ms) | Energy (W) |
|--------|-------------|------------------|--------------|-------------|
| Server-Only | 76.2 ± 0.8 | 2240.5 | 100 | 4.3 |
| Edge-Only | 68.4 ± 1.2 | 0 | 50 | 4.5 |
| FedCL | 71.3 ± 1.0 | 845.2 | 80 | 4.1 |
| **StreamSplit** | **74.5 ± 0.7** | **512.4** | **30** | **2.1** |

### On-Device Dataset Results

- **Accuracy**: 79.8% (within 2% of server-only)
- **Resource Efficiency**: 42.7% CPU usage in constrained mode
- **Battery Life**: Extended from 4.2h to 8.8h

## 🧪 Running Experiments

### Basic Usage

```bash
# Run edge device
python scripts/run_edge.py --config config/edge_config.yaml

# Run server
python scripts/run_server.py --config config/server_config.yaml

# Evaluate performance
python scripts/evaluate.py --exp_dir experiments/audioset
```

### Network Simulation

```bash
# Simulate varying network conditions
python scripts/simulate_network.py \
    --bandwidth_range 0.5 8.0 \
    --latency_range 50 200 \
    --duration 3600  # 1 hour
```

### Custom Dataset

```python
from streamsplit.data import AudioDataset
from streamsplit.training import EdgeTrainer

# Create custom dataset
dataset = AudioDataset(
    audio_dir="/path/to/audio",
    labels_file="labels.csv",
    transform=your_transform
)

# Train on custom data
trainer = EdgeTrainer(config)
trainer.train(dataset)
```

## 📈 Performance Tuning

### Resource Constraints

```python
# Adjust for memory-constrained devices
config.contrastive.memory_bank_size = [32, 256]
config.model.width_mult = 0.5

# Enable aggressive compression
config.compression.enabled = True
config.compression.method = "quantization"
config.compression.bits = 8
```

### Network Optimization

```python
# Optimize for low bandwidth
config.uncertainty.threshold = 0.7  # Higher threshold = less transmission
config.transmission.compression = "gzip"
config.transmission.batch_size = 8
```

## 🔍 Monitoring and Logging

StreamSplit provides comprehensive monitoring:

```python
from streamsplit.monitoring import MetricsCollector

# Initialize metrics collection
collector = MetricsCollector()

# Track key metrics
collector.track("accuracy", accuracy)
collector.track("latency", processing_time)
collector.track("energy", power_consumption)

# Export metrics
collector.export_prometheus(port=9090)  # Prometheus format
collector.export_tensorboard("logs/")   # TensorBoard format
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run type checking
mypy src/
```

## 📚 Citation

If you use StreamSplit in your research, please cite:


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AudioSet dataset for evaluation
- KissFFT library for optimized FFT implementation
- Research community for valuable feedback

---

*StreamSplit: Theoretical Guarantees for Edge Audio Learning* 🎵
