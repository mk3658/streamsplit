# StreamSplit API Reference

## Framework Overview

StreamSplit provides a dynamic edge-server framework for continuous audio representation learning with theoretical convergence guarantees.

## Core Classes

### `StreamSplit`

Main framework class orchestrating edge-server audio learning.

```python
class StreamSplit:
    def __init__(self, edge_config: EdgeConfig, server_config: ServerConfig, 
                 split_config: SplitConfig)
    def process_audio_stream(self, audio: AudioStream) -> ProcessingResult
    def update_split_point(self, layer_idx: int) -> None
```

**Parameters:**
- `edge_config`: Configuration for edge device
- `server_config`: Configuration for server
- `split_config`: Dynamic splitting configuration

## Edge Module

### `StreamingContrastiveModule`

Handles real-time contrastive learning on edge devices.

```python
class StreamingContrastiveModule:
    def __init__(self, model_config: ModelConfig, memory_bank_size: int = 512,
                 temperature: float = 0.1)
    def extract_features(self, audio: np.ndarray) -> np.ndarray
    def compute_local_loss(self, anchor: Embedding, positive: Embedding,
                          negatives: List[Embedding]) -> float
    def update_memory_bank(self, embedding: Embedding) -> None
    def distribution_aware_sampling(self, batch_size: int) -> List[Embedding]
```

**Key Methods:**

**`extract_features(audio)`**
- Adaptive feature extraction using policy πθ(xt)
- Automatically adjusts resolution based on resources
- Returns 128D embeddings

**`compute_local_loss(anchor, positive, negatives)`**
- Contrastive loss with age weighting: γ^(t-t(e⁻))
- Formula: `Llocal = -log(exp(s(et,e⁺t)/τ) / (exp(s(et,e⁺t)/τ) + Σexp(s(et,e⁻)/τ)·w(e⁻)))`

### `AdaptiveFeatureExtractor`

Dynamically adjusts audio processing based on resources.

```python
class AdaptiveFeatureExtractor:
    def __init__(self, threshold: float = 0.7)
    def extract(self, audio: np.ndarray, resources: ResourceState) -> np.ndarray
    def set_policy(self, policy: ExtractionPolicy) -> None
```

## Server Module

### `ServerAggregationModule`

Handles embedding aggregation and global refinement.

```python
class ServerAggregationModule:
    def __init__(self, model_config: ModelConfig, num_prototypes: int = 100,
                 sw_projections: int = 100)
    def aggregate_embeddings(self, device_embeddings: Dict[str, List[Embedding]]) -> Distribution
    def compute_hybrid_loss(self, p: Distribution, q: Distribution) -> float
    def refine_embeddings(self, embeddings: List[Embedding]) -> List[Embedding]
```

**Key Methods:**

**`aggregate_embeddings(device_embeddings)`**
- Intra-device temporal aggregation: `p̂i(e) = (1/|Ti|)Σ K(e, e(i)t)`
- Cross-device distribution alignment using Sliced-Wasserstein

**`compute_hybrid_loss(p, q)`**
- Combined loss: `Lserver = LSW + λLLap`
- Sliced-Wasserstein: `LSW(p,q) = (1/L Σ W²₂(pθl,qθl))^(1/2)`
- Laplacian: `LLap = (1/N²)Σᵢⱼ Wᵢⱼ||eᵢ-eⱼ||²₂`

### `SelectiveTransmissionModule`

Controls which embeddings are transmitted to server.

```python
class SelectiveTransmissionModule:
    def __init__(self, uncertainty_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0))
    def should_transmit(self, embedding: Embedding, state: SystemState) -> bool
    def compute_uncertainty(self, embedding: Embedding) -> float
```

**Uncertainty Components:**
- Consistency: `Uconsistency(et) = ||fθ(xt) - fθ(A(xt))||²`
- Entropy: `Uentropy(et) = H(p(y|et))`
- Prototype: `Uprototype(et) = min_k ||et - pk||²`

## Dynamic Splitting

### `SplitAgent`

Reinforcement learning agent for computation splitting decisions.

```python
class SplitAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256)
    def select_action(self, state: State) -> int
    def update_policy(self, transition: Transition) -> None
    def train_step(self, batch: TransitionBatch) -> Dict[str, float]
```

**State Components:**
- `rt`: Resource state (CPU, memory, battery)
- `nt`: Network conditions (bandwidth, latency, packet loss)
- `pt`: Performance metrics
- `ht`: Decision history

**Reward Function:**
```
R(st, at) = λ₁Accuracy(st,at) - λ₂ResourceUsage(st,at) 
          - λ₃Latency(st,at) - λ₄PrivacyRisk(st,at)
```

### `GraphPartitioner`

Handles computational graph partitioning.

```python
class GraphPartitioner:
    def __init__(self, graph: ComputationalGraph)
    def find_optimal_cut(self, constraints: ResourceConstraints) -> Tuple[List[Node], List[Node]]
    def apply_transformation(self, split_point: int, transform_type: str) -> None
```

**Available Transformations:**
- Bottleneck insertion
- Layer quantization  
- Conditional computation

## Audio Processing

### `OptimizedFFT`

Efficient FFT implementation for edge devices.

```python
class OptimizedFFT:
    def __init__(self, fft_size: int = 512, window_size: int = 400, hop_size: int = 160)
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray
    def set_precision(self, precision: str = 'fp32') -> None
```

**Specifications:**
- Window: 25ms Hann window
- Hop size: 10ms (60% overlap)
- FFT size: 512 points with zero padding
- Max latency: 2.1ms including windowing

### `AudioAugmentation`

Lightweight augmentations for positive pair generation.

```python
class AudioAugmentation:
    def time_shift(self, audio: np.ndarray, max_shift: float = 0.25) -> np.ndarray
    def frequency_mask(self, spectrogram: np.ndarray, mask_fraction: float = 0.1) -> np.ndarray
    def amplitude_scale(self, audio: np.ndarray, scale_range: Tuple[float, float] = (0.75, 1.25)) -> np.ndarray
```

## Configuration Classes

### `EdgeConfig`

```python
@dataclass
class EdgeConfig:
    model_path: str
    memory_bank_size: int = 512
    temperature: float = 0.1
    momentum: float = 0.999
    learning_rate: float = 1e-4
    resource_threshold: float = 0.7
    gradient_accumulation: int = 4
```

### `ServerConfig`

```python
@dataclass
class ServerConfig:
    model_path: str
    batch_size: int = 256
    learning_rate: float = 5e-4
    num_prototypes: int = 100
    sw_projections: int = 100
    lambda_laplacian: float = 0.1
```

### `SplitConfig`

```python
@dataclass
class SplitConfig:
    reward_weights: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.2)
    ppo_clip_epsilon: float = 0.2
    constraint_penalty: float = 0.1
    adaptation_window: int = 30
    update_frequency: int = 100
```

## Data Structures

### `Embedding`

```python
@dataclass
class Embedding:
    vector: np.ndarray      # 128D embedding vector
    timestamp: float        # Unix timestamp
    device_id: str         # Edge device identifier
    uncertainty: float     # Computed uncertainty score
    metadata: Dict         # Additional metadata
```

### `Distribution`

```python
@dataclass
class Distribution:
    embeddings: List[Embedding]
    mean: np.ndarray
    covariance: np.ndarray
    mixture_weights: List[float]
    components: List[GaussianComponent]
```

### `SystemState`

```python
@dataclass
class SystemState:
    resources: ResourceState
    network: NetworkState
    performance: PerformanceMetrics
    history: List[int]
```

### `ResourceState`

```python
@dataclass
class ResourceState:
    cpu_usage: float       # 0.0 to 1.0
    memory_usage: float    # 0.0 to 1.0  
    battery_level: float   # 0.0 to 1.0
    temperature: float     # Device temperature
```

### `NetworkState`

```python
@dataclass
class NetworkState:
    bandwidth: float       # Mbps
    latency: float         # ms
    packet_loss: float     # 0.0 to 1.0
    jitter: float         # ms
```

## Utility Functions

### Core Utilities

```python
def sliced_wasserstein_distance(p: Distribution, q: Distribution, 
                               num_projections: int = 100) -> float
    """Compute Sliced-Wasserstein distance between distributions."""

def laplacian_regularization(embeddings: np.ndarray, k_neighbors: int = 5) -> float
    """Compute Laplacian regularization term."""

def distribution_aware_sampling(memory_bank: MemoryBank, gmm: GMM, 
                               alpha: float = 1.0) -> List[int]
    """Sample negatives using Distribution-Aware Sampling."""

def adaptive_bandwidth_selection(embeddings: List[Embedding], 
                                k_neighbors: int = 10) -> float
    """Adaptively set Gaussian kernel bandwidth."""
```

### Metrics and Evaluation

```python
def compute_downstream_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float
    """Evaluate representation quality via linear probing."""

def precision_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int = 10) -> float
    """Compute Precision@K for nearest neighbor retrieval."""

def measure_resource_efficiency(device_metrics: DeviceMetrics) -> Dict[str, float]
    """Measure CPU, memory, and energy efficiency."""
```

## Constants

```python
# Audio Processing
SAMPLE_RATE = 16000
WINDOW_SIZE = 400          # 25ms at 16kHz
HOP_SIZE = 160            # 10ms at 16kHz  
FFT_SIZE = 512
OVERLAP_RATIO = 0.6

# Model Architecture
EMBEDDING_DIM = 128
NUM_FREQUENCY_BINS = 128
NUM_TIME_FRAMES = 64

# Training Hyperparameters
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MOMENTUM = 0.999
DEFAULT_LEARNING_RATE = 1e-4
MIN_MEMORY_BANK_SIZE = 64
MAX_MEMORY_BANK_SIZE = 512

# Network Thresholds
MIN_BANDWIDTH_MBPS = 0.5
CRITICAL_BANDWIDTH_MBPS = 1.0
MAX_LATENCY_MS = 200
MIN_BATTERY_LEVEL = 0.2

# Uncertainty Thresholds
BASE_UNCERTAINTY_THRESHOLD = 0.5
UNCERTAINTY_ADAPTATION_RATE = 0.1
```

## Example Usage

### Basic Setup

```python
# Configure edge and server
edge_config = EdgeConfig(
    model_path="mobilenet_edge.pth",
    memory_bank_size=512,
    temperature=0.1
)

server_config = ServerConfig(
    model_path="mobilenet_server.pth",
    batch_size=256,
    num_prototypes=100
)

split_config = SplitConfig(
    reward_weights=(1.0, 0.5, 0.3, 0.2)
)

# Initialize framework
framework = StreamSplit(edge_config, server_config, split_config)
```

### Audio Processing

```python
# Load and process audio stream
audio_stream = load_audio_stream("continuous_audio.wav")
result = framework.process_audio_stream(audio_stream)

# Access results
embeddings = result.embeddings
accuracy = result.downstream_accuracy
bandwidth_used = result.communication_cost
energy_consumed = result.energy_consumption
```

### Dynamic Adaptation

```python
# Monitor and adapt to changing conditions
while streaming:
    state = framework.get_current_state()
    
    # Automatic adaptation
    if state.resources.cpu_usage > 0.9:
        framework.reduce_computation_load()
    
    if state.network.bandwidth < 1.0:
        framework.increase_edge_computation()
    
    # Manual split point update
    framework.update_split_point(layer_idx=5)
```

## Performance Guarantees

### Convergence Theorem

Under mild assumptions (μ-strongly convex loss, bounded transmission error), StreamSplit converges with:

- **Parameter convergence**: `E[||fT - f*||²] ≤ C₁/T + C₂ε²`
- **Loss convergence**: `E[|L(fT) - L(f*)|] ≤ Lσ²/(2μ²T) + Lε²/2`

Where:
- `T`: Number of iterations
- `ε`: Approximation error from splitting
- `σ²`: Variance in gradient estimates
- `μ`, `L`: Strong convexity and smoothness constants

### Resource Efficiency Targets

- **Bandwidth reduction**: 75% vs server-only
- **Latency reduction**: 70% vs server-only  
- **Energy savings**: 50% vs edge-only
- **Accuracy preservation**: Within 2% of server-only

## Hardware Requirements

### Edge Device Specifications

- **Minimum**: Raspberry Pi 4B (4GB RAM, ARM Cortex-A72)
- **CPU**: Quad-core 1.5GHz with NEON SIMD support
- **Memory**: 4GB RAM (StreamSplit uses ~820MB normal, ~680MB constrained)
- **Storage**: 8GB for model and buffers
- **Network**: Wi-Fi or cellular with 0.5+ Mbps

### Server Requirements

- **CPU**: Multi-core x86_64 or ARM64
- **Memory**: 16GB+ RAM for batch processing
- **GPU**: Optional but recommended for faster refinement
- **Storage**: High-speed SSD for embedding database
- **Network**: High-bandwidth connection (10+ Mbps)

---

*This API reference covers StreamSplit v1.0. For implementation details, see the extensive appendices in the research paper.*