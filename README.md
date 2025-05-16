# StreamSplit: Theoretical Guarantees for Edge Audio Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](#)

## Overview

StreamSplit is a novel framework for real-time continuous audio representation learning at the edge. It addresses the limitations of centralized methods under extreme computational, memory, and bandwidth restrictions by providing a streaming contrastive learning approach with dynamic edge-server computation splitting.

## Architecture

![StreamSplit Framework](./images/architecture_diagram.png)

*Figure 1: End-to-end overview of StreamSplit showing the edge-server processing pipeline with dynamic computation splitting.*

### Key Components

The StreamSplit framework consists of three main components:

1. **Edge Device Processing**
   - Raw audio acquisition with FFT & augmentations
   - Adaptive feature extraction based on available resources
   - Device-side model execution (Conv1→...→Convk)
   - Streaming memory bank with contrastive learning
   - Local buffer and gradient queue management

2. **Dynamic Split Decision**
   - RL Agent monitoring system resources and network conditions
   - Uncertainty Module assessing embedding quality
   - Performance metrics feedback loop
   - Real-time split point adjustment within the deep encoder

3. **Server-Side Processing**
   - Completion of model inference (Convk+1→...→ConvL)
   - Hybrid loss computation (Sliced-Wasserstein + Laplacian)
   - Global model updates and weight synchronization
   - Embedding metadata management

## Key Features

- **Streaming Contrastive Framework**: Learn from embedding distributions with theoretical convergence guarantees
- **Hybrid Loss Function**: Combines Sliced-Wasserstein distance with Laplacian regularization
- **Dynamic Computation Splitting**: Adaptively distributes workload between edge and server based on resources
- **Theoretical Guarantees**: Proven convergence bounds for both convex and non-convex cases
- **Edge Optimization**: Optimized for resource-constrained devices like Raspberry Pi

## Performance

- **Accuracy**: 97.8% of server-only performance (within 2% gap)
- **Bandwidth Reduction**: 77.1% less than server-only approach
- **Latency Reduction**: 72.6% lower than server-only processing
- **Energy Savings**: 52.3% reduction compared to edge-only methods

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy
- librosa (for audio processing)
- Raspberry Pi 4B (for edge deployment)

### Required Dependencies

```bash
pip install torch>=1.9.0 torchvision torchaudio
pip install numpy scipy librosa
pip install scikit-learn matplotlib seaborn
pip install psutil websockets asyncio
pip install pyyaml dataclasses-json
```

### Quick Install

```bash
git clone https://github.com/mk3658/streamsplit.git
cd streamsplit
pip install -r requirements.txt

# Add the src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Hardware Requirements

**Edge Device (Minimum):**
- Raspberry Pi 4B with 4GB RAM
- USB 2.0 microphone
- Wi-Fi/cellular connectivity (0.5+ Mbps)

**Server:**
- Multi-core CPU (x86_64 or ARM64)
- 16GB+ RAM
- GPU recommended for faster processing
- High-speed network connection

## Quick Start

### Basic Framework Usage

```python
import asyncio
import numpy as np
import torch
from src.core.streamsplit import StreamSplitFramework, StreamSplitConfig, StreamSplitMode
from src.models.encoders import MobileNetV3EdgeEncoder, MobileNetV3ServerEncoder

# Configure framework
config = StreamSplitConfig(
    device_id="edge_device_1",
    device_type="raspberry_pi_4b",
    sample_rate=16000,
    n_mels=128,
    embedding_dim=128,
    learning_rate=1e-4,
    memory_bank_size_min=64,
    memory_bank_size_max=512,
    split_reward_weights={
        'accuracy': 1.0,
        'resource_usage': -0.5,
        'latency': -0.3,
        'privacy_risk': -0.2
    }
)

# Initialize framework
framework = StreamSplitFramework(config)

# Start in dynamic split mode
async def run_streamsplit():
    await framework.start(mode=StreamSplitMode.DYNAMIC_SPLIT)
    
    # Process audio stream (example with synthetic audio)
    audio_data = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
    result = await framework.process_audio_stream(audio_data)
    
    print(f"Processing mode: {result['mode']}")
    print(f"Latency: {result['latency']:.2f}ms")
    print(f"Split point: {result['split_point']}")
    
    await framework.stop()

# Run the example
asyncio.run(run_streamsplit())
```

### Edge-Only Processing

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
edge_config = TrainingConfig(
    learning_rate=1e-4,
    temperature=0.1,
    momentum=0.999,
    batch_size=32,
    memory_bank_min_size=64,
    memory_bank_max_size=512
)

# Initialize edge trainer
edge_trainer = EdgeTrainer(edge_config, edge_model, device='cpu')

# Process audio spectrogram
spectrogram = torch.randn(128, 64)  # Mel spectrogram
result = edge_trainer.train_step(spectrogram)

print(f"Loss: {result['loss']['total_loss']:.4f}")
print(f"Memory bank size: {result['memory_bank_stats']['current_size']}")
```

### Server-Side Processing

```python
import asyncio
from src.core.server_module import ServerModule
from src.training.server_trainer import ServerTrainer, ServerTrainingConfig
from src.models.encoders import MobileNetV3ServerEncoder

# Create server model
server_model = MobileNetV3ServerEncoder(
    intermediate_dim=256,
    embedding_dim=128,
    num_layers=4
)

# Configure server trainer
server_config = ServerTrainingConfig(
    learning_rate=5e-4,
    batch_size=256,
    sw_projections=100,
    laplacian_k_neighbors=5,
    laplacian_weight=0.5
)

# Initialize server trainer
server_trainer = ServerTrainer(server_config, server_model, device='cuda')

# Start server training
async def run_server():
    await server_trainer.start_training()
    
    # Simulate receiving embeddings from edge device
    embeddings_data = {
        'embeddings': torch.randn(128),
        'timestamps': [time.time()],
        'metadata': {'device_id': 'edge_1', 'split_point': 8}
    }
    
    await server_trainer.receive_embeddings('edge_1', embeddings_data)
    
    # Get training metrics
    metrics = server_trainer.get_training_metrics()
    print(f"Global step: {metrics['global_step']}")
    print(f"Average loss: {metrics['average_loss']:.4f}")
    
    await server_trainer.stop_training()

asyncio.run(run_server())
```

### Dynamic Splitting

```python
from src.core.dynamic_splitting import DynamicSplittingAgent

# Create splitting config (normally loaded from YAML)
class SplittingConfig:
    def __init__(self):
        self.bandwidth_range = (0.5, 8.0)
        self.latency_range = (50, 200)
        self.split_agent_lr = 1e-3
        self.split_reward_weights = {
            'accuracy': 1.0,
            'resource_usage': -0.5,
            'latency': -0.3,
            'privacy_risk': -0.2
        }

splitting_config = SplittingConfig()
splitting_agent = DynamicSplittingAgent(splitting_config)

# Start splitting agent
async def run_splitting():
    await splitting_agent.start()
    
    # Get split decision
    split_decision = await splitting_agent.get_split_decision()
    print(f"Split point: {split_decision['split_point']}")
    print(f"Confidence: {split_decision['confidence']:.3f}")
    print(f"Reasoning: {split_decision['reasoning']}")
    
    await splitting_agent.stop()

asyncio.run(run_splitting())
```

## Running the Framework

### Edge Device

```bash
# Run edge device with default configuration
python scripts/run_edge.py --config config/edge_config.yaml

# Run with custom parameters
python scripts/run_edge.py \
    --config config/edge_config.yaml \
    --device-id edge_device_1 \
    --log-level INFO \
    --synthetic  # Use synthetic audio for testing

# Run without dynamic splitting (edge-only mode)
python scripts/run_edge.py \
    --config config/edge_config.yaml \
    --no-splitting
```

### Server

```bash
# Run server with default configuration
python scripts/run_server.py --config config/server_config.yaml

# Run with custom parameters
python scripts/run_server.py \
    --config config/server_config.yaml \
    --port 8888 \
    --log-level INFO \
    --gpu  # Force GPU usage

# Run in CPU-only mode
python scripts/run_server.py \
    --config config/server_config.yaml \
    --cpu-only

# Validate configuration before running
python scripts/run_server.py validate config/server_config.yaml
```

## Experiments

### Running Evaluations

```bash
# Run comprehensive evaluation
python scripts/evaluate.py \
    --config config/edge_config.yaml \
    --output-dir results \
    --model-path models/streamsplit_model.pth

# Evaluate with specific datasets (when available)
python scripts/evaluate.py \
    --audioset-path data/audioset_subset \
    --ondevice-path data/ondevice_recordings \
    --output-dir results \
    --plot  # Generate visualizations

# Quick evaluation with synthetic data
python scripts/evaluate.py \
    --output-dir results \
    --plot
```

### Evaluation Results

The evaluation script provides comprehensive metrics:

- **Representation Quality**: Classification accuracy, Precision@K, embedding separability
- **Resource Efficiency**: CPU usage, memory consumption, energy efficiency
- **Communication Efficiency**: Bandwidth reduction, transmission analysis
- **Adaptation Performance**: Split decision quality, response time

### Custom Evaluation

```python
from src.utils.metrics import MetricsCollector, evaluate_embeddings
import torch
import numpy as np

# Create sample embeddings and labels
embeddings = torch.randn(1000, 128)  # 1000 samples, 128D embeddings
labels = np.random.randint(0, 10, 1000)  # 10 classes

# Evaluate embeddings
results = evaluate_embeddings(
    embeddings, 
    labels, 
    device='cpu',
    include_visualization=True
)

# Print results
print(f"Classification Accuracy: {results['classification_accuracy'].value:.3f}")
print(f"Precision@10: {results['precision_at_10'].value:.3f}")
print(f"Embedding Separability: {results['embedding_separability'].value:.3f}")

# Use metrics collector for comprehensive monitoring
collector = MetricsCollector(device='cpu')
collector.start_monitoring()

# ... run your experiments ...

collector.stop_monitoring()
report = collector.get_comprehensive_report()
collector.export_metrics('results/metrics.json', format='json')
```

## Configuration

StreamSplit uses YAML configuration files for both edge and server components.

### Edge Configuration

See `config/edge_config.yaml` for complete configuration options:

```yaml
# Key configuration sections
device:
  id: "edge_device_1"
  type: "raspberry_pi_4b"

audio:
  sample_rate: 16000
  n_mels: 128
  window_duration: 0.025

training:
  learning_rate: 1e-4
  temperature: 0.1
  batch_size: 32

memory_bank:
  min_size: 64
  max_size: 512

splitting:
  enabled: true
  agent:
    learning_rate: 1e-3
    reward_weights:
      accuracy: 1.0
      resource_usage: -0.5
      latency: -0.3
      privacy_risk: -0.2
```

### Server Configuration

See `config/server_config.yaml` for complete configuration options:

```yaml
# Key configuration sections
server:
  bind_address: "0.0.0.0"
  port: 8888

training:
  learning_rate: 5e-4
  batch_size: 256

loss:
  sliced_wasserstein:
    num_projections: 100
    weight: 1.0
  laplacian:
    k_neighbors: 5
    weight: 0.5

aggregation:
  temporal_window: 30.0
  min_devices_per_update: 2
```

## Model Training

### Training from Scratch

```bash
# Note: Training scripts are implemented in the trainer classes
# Use the run_edge.py and run_server.py scripts for training

# Edge training
python scripts/run_edge.py --config config/edge_config.yaml

# Server training  
python scripts/run_server.py --config config/server_config.yaml
```

### Pre-trained Models

Pre-trained models will be made available upon paper publication. For now, the framework initializes with random weights and learns from scratch.

```python
# Loading models (when available)
from src.models.encoders import MobileNetV3EdgeEncoder

edge_model = MobileNetV3EdgeEncoder(input_dim=128, embedding_dim=128)
# edge_model.load_state_dict(torch.load('path/to/pretrained_edge_model.pth'))
```

## Network Simulation

StreamSplit includes a comprehensive network simulation tool for testing under various conditions:

```bash
# Run network simulation
python scripts/simulate_network.py \
    --duration 300 \
    --scenario daily \
    --enable-shaping \
    --plot

# Available scenarios: daily, mobile, edge, random
# Simulates realistic bandwidth, latency, and packet loss patterns
```

## API Reference

### Core Classes

#### `StreamSplitFramework`

```python
class StreamSplitFramework:
    def __init__(self, config: StreamSplitConfig)
    async def start(self, mode: StreamSplitMode = StreamSplitMode.DYNAMIC_SPLIT)
    async def stop()
    async def process_audio_stream(self, audio: np.ndarray) -> Dict[str, Any]
    def save_checkpoint(self, filepath: str)
    def load_checkpoint(self, filepath: str)
```

#### `EdgeModule`

```python
class EdgeModule:
    async def process(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]
    async def process_partial(self, spectrogram: torch.Tensor, split_point: int) -> Dict[str, Any]
    def get_resource_state(self) -> Dict[str, float]
```

#### `ServerModule`

```python
class ServerModule:
    async def process(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]
    async def process_continuation(self, edge_output: Dict[str, Any], split_point: int) -> Dict[str, Any]
    async def receive_edge_embedding(self, embedding_data: Dict[str, Any]) -> bool
```

#### `DynamicSplittingAgent`

```python
class DynamicSplittingAgent:
    async def get_split_decision(self) -> Dict[str, Any]
    async def update_performance(self, performance_history: Dict[str, List[float]])
    def get_network_quality(self) -> float
```

For detailed API documentation, see [docs/api_reference.md](docs/api_reference.md).

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
- Parameter convergence: `E[||fT - f*||²] ≤ C₁/T + C₂ε²`
- Loss convergence: `E[|L(fT) - L(f*)|] ≤ Lσ²/(2μ²T) + Lε²/2`

Where T is iterations, ε is approximation error from splitting, and σ² is gradient estimate variance.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to add the `src` directory to your Python path:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **CUDA Out of Memory**: Reduce batch sizes in configuration files or use CPU-only mode:
   ```bash
   python scripts/run_server.py --cpu-only
   ```

3. **Audio Processing Issues**: Ensure librosa is properly installed:
   ```bash
   pip install librosa>=0.8.0
   ```

4. **Raspberry Pi Performance**: Use reduced resolution mode for resource-constrained devices (automatically enabled when CPU usage > 70%).

### Performance Optimization

- **Edge Device**: Enable reduced resolution mode, adjust memory bank size based on available RAM
- **Server**: Use GPU acceleration, tune batch sizes for optimal throughput
- **Network**: Adjust uncertainty thresholds for transmission decisions based on bandwidth availability

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and optimized for Raspberry Pi deployment
- Uses KissFFT library for efficient edge processing
- Inspired by recent advances in contrastive learning and edge computing

## Support

- 📧 Email: [m.quan@deakin.edu.au](mailto:m.quan@deakin.edu.au)
- 📖 Documentation: [docs/](docs/)

---

**StreamSplit** - Enabling efficient audio AI at the edge with theoretical guarantees.