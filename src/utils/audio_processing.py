"""
Audio Processing Utilities for StreamSplit Framework
Implements optimized audio preprocessing with KissFFT and edge-specific augmentations
Based on Section 3.1.1 and Appendix A, C, D of the StreamSplit paper
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import random
import time
import logging
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from scipy.signal import windows
import threading
import queue

# Try to import optimized FFT libraries
try:
    import pyfftw
    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False
    import scipy.fft as fft

# Export main classes for import
__all__ = ['AudioProcessor', 'AudioAugmentations', 'OptimizedFFT', 'StreamingAudioBuffer', 
           'PerformanceMonitor', 'AudioConfig']

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    window_duration: float = 0.025  # 25ms Hann window
    hop_length: float = 0.010      # 10ms hop
    n_fft: int = 512
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    power: float = 2.0
    normalized: bool = True
    
    # Edge-specific optimizations
    reduced_freq_factor: int = 2    # Frequency reduction factor
    reduced_stride_factor: int = 2  # Stride reduction factor

class OptimizedFFT:
    """
    Optimized FFT implementation using KissFFT-style optimizations
    Based on Appendix A and R of the paper
    """
    
    def __init__(self, n_fft: int = 512, use_pyfftw: bool = True):
        self.n_fft = n_fft
        self.use_pyfftw = use_pyfftw and HAS_PYFFTW
        self.logger = logging.getLogger(__name__)
        
        # Pre-allocate buffers for efficiency (fixed-size buffers as mentioned in paper)
        self._input_buffer = np.zeros(n_fft, dtype=np.complex64)
        self._output_buffer = np.zeros(n_fft, dtype=np.complex64)
        
        # Initialize FFTW if available
        if self.use_pyfftw:
            self._init_fftw()
        
        # Pre-compute twiddle factors for optimization
        self._twiddle_factors = self._compute_twiddle_factors()
        
        self.logger.info(f"OptimizedFFT initialized with n_fft={n_fft}, using_pyfftw={self.use_pyfftw}")
    
    def _init_fftw(self):
        """Initialize FFTW wisdom and plans for optimized computation"""
        try:
            # Create FFTW plan for reuse
            self._fftw_input = pyfftw.empty_aligned(self.n_fft, dtype='complex64')
            self._fftw_output = pyfftw.empty_aligned(self.n_fft, dtype='complex64')
            self._fftw_plan = pyfftw.FFTW(
                self._fftw_input, 
                self._fftw_output,
                direction='FFTW_FORWARD',
                flags=('FFTW_MEASURE',)
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize FFTW: {e}, falling back to scipy")
            self.use_pyfftw = False
    
    def _compute_twiddle_factors(self) -> np.ndarray:
        """Pre-compute twiddle factors for FFT optimization"""
        angles = -2 * np.pi * np.arange(self.n_fft // 2) / self.n_fft
        return np.exp(1j * angles).astype(np.complex64)
    
    def fft(self, x: np.ndarray) -> np.ndarray:
        """
        Optimized FFT computation with in-place operations
        Implements optimizations mentioned in Appendix R
        """
        # Ensure input is correct dtype
        if x.dtype != np.complex64:
            x = x.astype(np.complex64)
        
        if self.use_pyfftw:
            # Use FFTW for optimized computation
            self._fftw_input[:] = x
            self._fftw_plan()
            return self._fftw_output.copy()
        else:
            # Use scipy FFT as fallback
            return fft.fft(x, n=self.n_fft).astype(np.complex64)
    
    def stft(self, audio: np.ndarray, hop_length: int, 
            window: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimized Short-Time Fourier Transform"""
        if window is None:
            window = windows.hann(self.n_fft, sym=False).astype(np.float32)
        
        # Calculate number of frames
        n_frames = 1 + (len(audio) - self.n_fft) // hop_length
        
        # Pre-allocate output
        stft_matrix = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex64)
        
        # Windowed FFT with optimizations
        for i in range(n_frames):
            start = i * hop_length
            windowed_frame = audio[start:start + self.n_fft] * window
            
            # Pad if necessary
            if len(windowed_frame) < self.n_fft:
                windowed_frame = np.pad(windowed_frame, (0, self.n_fft - len(windowed_frame)))
            
            # Compute FFT
            fft_frame = self.fft(windowed_frame)
            stft_matrix[:, i] = fft_frame[:self.n_fft // 2 + 1]
        
        return stft_matrix

class AudioProcessor:
    """
    Main audio processor implementing StreamSplit-specific preprocessing
    Supports adaptive feature extraction and edge optimizations
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimized FFT
        self.fft_processor = OptimizedFFT(
            n_fft=config.n_fft,
            use_pyfftw=True
        )
        
        # Pre-compute mel filter bank
        self.mel_filters = self._create_mel_filterbank()
        
        # Pre-compute window function
        self.window = windows.hann(config.n_fft, sym=False).astype(np.float32)
        
        # Calculate frame parameters
        self.hop_length_samples = int(config.hop_length * config.sample_rate)
        self.window_length_samples = int(config.window_duration * config.sample_rate)
        
        # Edge optimization state
        self.reduced_mode = False
        self.current_resolution = 'full'
        
        self.logger.info(f"AudioProcessor initialized with sample_rate={config.sample_rate}")
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filter bank for spectrogram conversion"""
        fmax = self.config.fmax or self.config.sample_rate // 2
        return librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=fmax
        ).astype(np.float32)
    
    def extract_spectrogram(self, audio: np.ndarray, 
                          reduced_resolution: bool = False) -> torch.Tensor:
        """
        Extract mel spectrogram from audio with optional resolution reduction
        Implements adaptive feature extraction policy from Equation 1
        """
        # Ensure audio is float32 for processing
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Compute STFT
        stft_matrix = self.fft_processor.stft(
            audio, 
            hop_length=self.hop_length_samples,
            window=self.window
        )
        
        # Convert to power spectrogram
        power_spec = np.abs(stft_matrix) ** self.config.power
        
        # Apply mel filter bank
        mel_spec = np.dot(self.mel_filters, power_spec)
        
        # Apply logarithm
        mel_spec = np.log(mel_spec + 1e-10)
        
        # Apply resolution reduction if requested (Appendix C)
        if reduced_resolution:
            mel_spec = self._reduce_resolution(mel_spec)
            self.current_resolution = 'reduced'
        else:
            self.current_resolution = 'full'
        
        # Normalize if requested
        if self.config.normalized:
            mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-10)
        
        # Convert to tensor
        return torch.from_numpy(mel_spec).float()
    
    def _reduce_resolution(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Reduce spectrogram resolution for resource-constrained processing
        Implements Appendix C optimizations
        """
        # Reduce frequency resolution by averaging adjacent bins
        freq_bins, time_frames = mel_spec.shape
        
        # Reduce frequency dimension by factor of 2
        if freq_bins % self.config.reduced_freq_factor == 0:
            new_freq_bins = freq_bins // self.config.reduced_freq_factor
            reduced_freq = mel_spec.reshape(
                new_freq_bins, self.config.reduced_freq_factor, time_frames
            ).mean(axis=1)
        else:
            # Handle case where frequency bins don't divide evenly
            new_freq_bins = freq_bins // self.config.reduced_freq_factor
            reduced_freq = mel_spec[:new_freq_bins * self.config.reduced_freq_factor].reshape(
                new_freq_bins, self.config.reduced_freq_factor, time_frames
            ).mean(axis=1)
        
        # Reduce temporal resolution by increased stride
        reduced_spec = reduced_freq[:, ::self.config.reduced_stride_factor]
        
        return reduced_spec
    
    def set_reduced_mode(self, enabled: bool):
        """Enable/disable reduced resolution mode"""
        self.reduced_mode = enabled
        self.logger.info(f"Reduced resolution mode: {'enabled' if enabled else 'disabled'}")
    
    def process_streaming_audio(self, audio_stream: np.ndarray,
                               adaptive_policy: Optional[callable] = None) -> torch.Tensor:
        """
        Process streaming audio with adaptive resolution policy
        Implements dynamic resolution switching based on resources
        """
        # Apply adaptive policy if provided
        use_reduced = self.reduced_mode
        if adaptive_policy is not None:
            use_reduced = adaptive_policy()
        
        # Extract spectrogram with appropriate resolution
        return self.extract_spectrogram(audio_stream, reduced_resolution=use_reduced)
    
    def get_current_resolution(self) -> str:
        """Get current processing resolution mode"""
        return self.current_resolution

class AudioAugmentations:
    """
    Edge-optimized audio augmentations for positive pair generation
    Implements augmentations from Appendix D
    """
    
    def __init__(self, sample_rate: int = 16000, compute_on_device: bool = True):
        self.sample_rate = sample_rate
        self.compute_on_device = compute_on_device
        self.logger = logging.getLogger(__name__)
        
        # Augmentation parameters
        self.time_shift_range = (-0.1, 0.1)  # ±10% of duration
        self.freq_mask_param = 15  # Maximum frequency bins to mask
        self.amplitude_range = (0.75, 1.25)  # ±25% amplitude variation
        self.noise_factor = 0.01  # 1% noise level
        
        self.logger.info("AudioAugmentations initialized for edge processing")
    
    def time_shift(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time shifting augmentation
        A_time(x_t)[n] = x_t[n + δt] with δt ~ Uniform(-τ/4, τ/4)
        """
        if not self.compute_on_device:
            return spectrogram
        
        freq_bins, time_frames = spectrogram.shape
        
        # Calculate shift amount
        max_shift = int(time_frames * 0.1)  # 10% of total frames
        shift = random.randint(-max_shift, max_shift)
        
        if shift == 0:
            return spectrogram
        
        # Apply circular shift
        return torch.roll(spectrogram, shifts=shift, dims=1)
    
    def frequency_masking(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking augmentation
        A_freq(X_t)[f, τ] = X_t[f, τ] · ⊮_{f ∉ [f_0, f_0+Δf]}
        """
        if not self.compute_on_device:
            return spectrogram
        
        freq_bins, time_frames = spectrogram.shape
        
        # Random frequency mask
        mask_width = random.randint(1, min(self.freq_mask_param, freq_bins // 4))
        mask_start = random.randint(0, freq_bins - mask_width)
        
        # Create mask
        mask = torch.ones_like(spectrogram)
        mask[mask_start:mask_start + mask_width, :] = 0
        
        return spectrogram * mask
    
    def time_masking(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking augmentation"""
        if not self.compute_on_device:
            return spectrogram
        
        freq_bins, time_frames = spectrogram.shape
        
        # Random time mask
        mask_width = random.randint(1, min(15, time_frames // 4))
        mask_start = random.randint(0, time_frames - mask_width)
        
        # Create mask
        mask = torch.ones_like(spectrogram)
        mask[:, mask_start:mask_start + mask_width] = 0
        
        return spectrogram * mask
    
    def amplitude_scaling(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply amplitude scaling augmentation
        A_amp(x_t)[n] = γ · x_t[n] with γ ~ Uniform(0.75, 1.25)
        """
        if not self.compute_on_device:
            return spectrogram
        
        # Random amplitude scaling
        scale = random.uniform(*self.amplitude_range)
        return spectrogram * scale
    
    def gaussian_noise(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Add small amount of Gaussian noise"""
        if not self.compute_on_device:
            return spectrogram
        
        noise = torch.randn_like(spectrogram) * self.noise_factor
        return spectrogram + noise
    
    def apply(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random combination of augmentations
        Optimized for edge device computation efficiency
        """
        augmented = spectrogram.clone()
        
        # Apply each augmentation with probability
        if random.random() < 0.8:  # 80% chance
            augmented = self.time_shift(augmented)
        
        if random.random() < 0.6:  # 60% chance
            augmented = self.frequency_masking(augmented)
        
        if random.random() < 0.4:  # 40% chance
            augmented = self.time_masking(augmented)
        
        if random.random() < 0.7:  # 70% chance
            augmented = self.amplitude_scaling(augmented)
        
        if random.random() < 0.3:  # 30% chance
            augmented = self.gaussian_noise(augmented)
        
        return augmented
    
    def create_positive_pair(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create positive pair for contrastive learning"""
        original = spectrogram
        augmented = self.apply(spectrogram)
        return original, augmented

class StreamingAudioBuffer:
    """
    Circular buffer for streaming audio processing
    Handles continuous audio streams with overlapping windows
    """
    
    def __init__(self, buffer_size: int = 16000 * 5,  # 5 seconds at 16kHz
                 overlap_samples: int = 1600):  # 10% overlap
        self.buffer_size = buffer_size
        self.overlap_samples = overlap_samples
        self.buffer = np.zeros(buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.samples_available = 0
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def push(self, audio_chunk: np.ndarray):
        """Add audio chunk to the buffer"""
        with self.lock:
            chunk_size = len(audio_chunk)
            
            # Handle wrap-around
            if self.write_pos + chunk_size <= self.buffer_size:
                self.buffer[self.write_pos:self.write_pos + chunk_size] = audio_chunk
            else:
                # Split across boundary
                first_part = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = audio_chunk[:first_part]
                self.buffer[:chunk_size - first_part] = audio_chunk[first_part:]
            
            self.write_pos = (self.write_pos + chunk_size) % self.buffer_size
            self.samples_available = min(self.samples_available + chunk_size, self.buffer_size)
    
    def pull(self, chunk_size: int) -> Optional[np.ndarray]:
        """Extract audio chunk from buffer"""
        with self.lock:
            if self.samples_available < chunk_size:
                return None
            
            # Extract chunk
            if self.read_pos + chunk_size <= self.buffer_size:
                chunk = self.buffer[self.read_pos:self.read_pos + chunk_size].copy()
            else:
                # Handle wrap-around
                first_part = self.buffer_size - self.read_pos
                chunk = np.concatenate([
                    self.buffer[self.read_pos:],
                    self.buffer[:chunk_size - first_part]
                ])
            
            # Update read position with overlap
            advance = chunk_size - self.overlap_samples
            self.read_pos = (self.read_pos + advance) % self.buffer_size
            self.samples_available -= advance
            
            return chunk
    
    def has_samples(self, required_samples: int) -> bool:
        """Check if buffer has enough samples"""
        with self.lock:
            return self.samples_available >= required_samples

class PerformanceMonitor:
    """Monitor audio processing performance metrics"""
    
    def __init__(self):
        self.processing_times = []
        self.memory_usage = []
        self.resolution_switches = {'full': 0, 'reduced': 0}
        self.total_samples_processed = 0
        self.start_time = time.time()
        
        self.logger = logging.getLogger(__name__)
    
    def log_processing_time(self, processing_time: float):
        """Log processing time for performance analysis"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 1000:  # Keep last 1000 measurements
            self.processing_times.pop(0)
    
    def log_resolution_switch(self, resolution: str):
        """Log resolution mode switches"""
        if resolution in self.resolution_switches:
            self.resolution_switches[resolution] += 1
    
    def log_samples_processed(self, num_samples: int):
        """Log number of samples processed"""
        self.total_samples_processed += num_samples
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        metrics = {
            'runtime_seconds': runtime,
            'total_samples_processed': self.total_samples_processed,
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0,
            'std_processing_time_ms': np.std(self.processing_times) * 1000 if self.processing_times else 0,
            'samples_per_second': self.total_samples_processed / runtime if runtime > 0 else 0,
            'resolution_switches': self.resolution_switches.copy(),
            'full_resolution_ratio': (
                self.resolution_switches['full'] / 
                max(1, sum(self.resolution_switches.values()))
            )
        }
        
        return metrics

# Factory functions for easy instantiation

def create_audio_processor(sample_rate: int = 16000,
                          n_mels: int = 128,
                          n_fft: int = 512,
                          **kwargs) -> AudioProcessor:
    """Create audio processor with default configuration"""
    config = AudioConfig(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        **kwargs
    )
    return AudioProcessor(config)

def create_edge_augmentations(sample_rate: int = 16000) -> AudioAugmentations:
    """Create edge-optimized augmentations"""
    return AudioAugmentations(sample_rate=sample_rate, compute_on_device=True)

# Example usage and testing functions

def test_audio_processing():
    """Test audio processing functionality"""
    # Create test audio signal
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz sine wave
    
    # Initialize processor
    processor = create_audio_processor(sample_rate=sample_rate)
    augmentations = create_edge_augmentations(sample_rate=sample_rate)
    
    # Process audio
    spectrogram = processor.extract_spectrogram(audio)
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    # Test reduced resolution
    reduced_spec = processor.extract_spectrogram(audio, reduced_resolution=True)
    print(f"Reduced spectrogram shape: {reduced_spec.shape}")
    
    # Test augmentations
    original, augmented = augmentations.create_positive_pair(spectrogram)
    print(f"Augmented spectrogram shape: {augmented.shape}")
    
    print("Audio processing test completed successfully!")

if __name__ == "__main__":
    test_audio_processing()