"""
On-Device Dataset Preparation for StreamSplit Framework
Creates and manages on-device audio recordings for evaluation
Based on experiments/on-device configuration and requirements
"""

import os
import sys
import time
import logging
import json
import yaml
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# Audio recording dependencies (optional)
try:
    import pyaudio
    import wave
    HAS_AUDIO_RECORDING = True
except ImportError:
    HAS_AUDIO_RECORDING = False
    logging.warning("PyAudio not available. Audio recording functionality disabled.")

# Additional dependencies
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_PROCESSING = True
except ImportError:
    HAS_AUDIO_PROCESSING = False
    logging.error("Required audio processing libraries (librosa, soundfile) not found.")

@dataclass
class OnDeviceConfig:
    """Configuration for on-device dataset preparation"""
    # Dataset parameters
    output_dir: str = "/data/ondevice"
    num_classes: int = 7
    classes: List[str] = None
    duration: float = 10.0  # seconds per recording
    sample_rate: int = 16000
    
    # Recording parameters
    recordings_per_class: int = 100
    recordings_per_location: int = 20
    buffer_size: int = 1024
    channels: int = 1
    
    # Processing parameters
    normalize_audio: bool = True
    remove_silence: bool = True
    apply_noise_gate: bool = True
    noise_gate_threshold: float = -40.0  # dB
    min_duration: float = 9.0
    max_duration: float = 11.0
    
    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Augmentation parameters
    apply_augmentations: bool = True
    augmentation_factor: int = 3  # Create N augmented versions per original
    
    # Recording locations and devices
    locations_config: str = "experiments/on-device/locations.yaml"
    devices_config: str = "experiments/on-device/devices_profile.yaml"
    
    # Quality control
    min_snr_db: float = 10.0  # Minimum signal-to-noise ratio
    max_silence_ratio: float = 0.3  # Maximum allowed silence in recording
    
    # Metadata
    include_metadata: bool = True
    metadata_format: str = "json"
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = [
                "Conversation", "Kitchen Activities", "Door Operations",
                "Footsteps", "Traffic Noise", "Background Music", "Ambient Silence"
            ]
        
        # Validate split ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'OnDeviceConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract dataset config
        dataset_config = config_dict.get('dataset', {})
        
        # Map YAML keys to config attributes
        config_kwargs = {
            'output_dir': dataset_config.get('path', '/data/ondevice'),
            'num_classes': dataset_config.get('num_classes', 7),
            'classes': dataset_config.get('classes', None),
            'duration': dataset_config.get('duration', 10.0),
            'sample_rate': dataset_config.get('sample_rate', 16000),
        }
        
        # Add other sections if they exist
        if 'recording' in config_dict:
            recording_config = config_dict['recording']
            config_kwargs.update({
                'recordings_per_class': recording_config.get('recordings_per_class', 100),
                'channels': recording_config.get('channels', 1),
                'normalize_audio': recording_config.get('normalize_audio', True),
                'remove_silence': recording_config.get('remove_silence', True),
            })
        
        return cls(**config_kwargs)

class LocationManager:
    """Manages recording locations and their characteristics"""
    
    def __init__(self, locations_config_path: str):
        self.logger = logging.getLogger(__name__)
        self.locations = self._load_locations(locations_config_path)
        
    def _load_locations(self, config_path: str) -> Dict[str, Any]:
        """Load locations configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('recording_locations', {})
        except FileNotFoundError:
            self.logger.warning(f"Locations config not found: {config_path}")
            return self._default_locations()
    
    def _default_locations(self) -> Dict[str, Any]:
        """Provide default locations if config file is missing"""
        return {
            'indoor': [
                {
                    'name': 'quiet_office',
                    'characteristics': 'low_ambient_noise',
                    'reverb_time': 0.3
                },
                {
                    'name': 'living_room',
                    'characteristics': 'moderate_activity',
                    'reverb_time': 0.5
                }
            ],
            'outdoor': [
                {
                    'name': 'street_sidewalk',
                    'characteristics': 'traffic_noise',
                    'noise_level_db': 65
                }
            ]
        }
    
    def get_all_locations(self) -> List[Dict[str, Any]]:
        """Get all available locations"""
        all_locations = []
        for location_type, locations in self.locations.items():
            for location in locations:
                location['type'] = location_type
                all_locations.append(location)
        return all_locations
    
    def get_location(self, location_name: str) -> Optional[Dict[str, Any]]:
        """Get specific location by name"""
        for location_type, locations in self.locations.items():
            for location in locations:
                if location['name'] == location_name:
                    location['type'] = location_type
                    return location
        return None

class DeviceManager:
    """Manages device profiles and capabilities"""
    
    def __init__(self, devices_config_path: str):
        self.logger = logging.getLogger(__name__)
        self.devices = self._load_devices(devices_config_path)
        
    def _load_devices(self, config_path: str) -> Dict[str, Any]:
        """Load device configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('devices', {})
        except FileNotFoundError:
            self.logger.warning(f"Devices config not found: {config_path}")
            return self._default_devices()
    
    def _default_devices(self) -> Dict[str, Any]:
        """Provide default device profiles"""
        return {
            'raspberry_pi_4b': {
                'cpu': 'cortex_a72',
                'cores': 4,
                'base_freq': 1.5,
                'memory': 4096,
                'power_budget': 3.1
            }
        }
    
    def get_device_profile(self, device_name: str) -> Optional[Dict[str, Any]]:
        """Get device profile by name"""
        return self.devices.get(device_name)

class AudioRecorder:
    """Handles audio recording functionality"""
    
    def __init__(self, config: OnDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not HAS_AUDIO_RECORDING:
            self.logger.error("Audio recording not available. Install pyaudio for recording functionality.")
            return
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.recording = False
        
    def record_audio(self, duration: float, device_index: Optional[int] = None) -> np.ndarray:
        """Record audio for specified duration"""
        if not HAS_AUDIO_RECORDING:
            raise RuntimeError("Audio recording not available")
        
        chunk = self.config.buffer_size
        format = pyaudio.paInt16
        channels = self.config.channels
        rate = self.config.sample_rate
        
        # Open stream
        stream = self.p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk
        )
        
        self.logger.info(f"Recording audio for {duration} seconds...")
        
        frames = []
        num_chunks = int(rate / chunk * duration)
        
        for _ in range(num_chunks):
            data = stream.read(chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        return audio_np
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices"""
        if not HAS_AUDIO_RECORDING:
            return []
        
        devices = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
        return devices
    
    def __del__(self):
        """Clean up PyAudio"""
        if hasattr(self, 'p'):
            self.p.terminate()

class AudioProcessor:
    """Processes recorded audio with noise reduction and quality checks"""
    
    def __init__(self, config: OnDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not HAS_AUDIO_PROCESSING:
            raise RuntimeError("Audio processing libraries not available")
    
    def process_audio(self, audio: np.ndarray, 
                     metadata: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process raw recorded audio"""
        processed_audio = audio.copy()
        processing_metadata = metadata.copy() if metadata else {}
        
        # Remove DC bias
        processed_audio = processed_audio - np.mean(processed_audio)
        
        # Apply noise gate if enabled
        if self.config.apply_noise_gate:
            processed_audio = self._apply_noise_gate(processed_audio)
            processing_metadata['noise_gate_applied'] = True
        
        # Remove silence if enabled
        if self.config.remove_silence:
            processed_audio, trim_info = self._remove_silence(processed_audio)
            processing_metadata['silence_removed'] = trim_info
        
        # Normalize audio if enabled
        if self.config.normalize_audio:
            processed_audio = self._normalize_audio(processed_audio)
            processing_metadata['normalized'] = True
        
        # Validate duration
        duration = len(processed_audio) / self.config.sample_rate
        if duration < self.config.min_duration or duration > self.config.max_duration:
            processing_metadata['duration_warning'] = f"Duration {duration:.2f}s outside range"
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(processed_audio)
        processing_metadata['quality_metrics'] = quality_metrics
        
        return processed_audio, processing_metadata
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to reduce background noise"""
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
        
        # Create gate mask
        gate_mask = audio_db > self.config.noise_gate_threshold
        
        # Apply gate with smooth transitions
        gate_kernel = np.ones(int(0.01 * self.config.sample_rate))  # 10ms smoothing
        gate_mask_smooth = np.convolve(gate_mask.astype(float), 
                                      gate_kernel / len(gate_kernel), 
                                      mode='same')
        
        return audio * gate_mask_smooth
    
    def _remove_silence(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove leading and trailing silence"""
        # Use librosa's trim function
        trimmed_audio, (start_frame, end_frame) = librosa.effects.trim(
            audio, 
            top_db=30,  # Silence threshold
            frame_length=1024,
            hop_length=256
        )
        
        trim_info = {
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'original_length': len(audio),
            'trimmed_length': len(trimmed_audio),
            'trim_ratio': len(trimmed_audio) / len(audio)
        }
        
        return trimmed_audio, trim_info
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _calculate_quality_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate audio quality metrics"""
        # Signal-to-noise ratio estimation
        # Simple method: compare signal power to estimated noise floor
        signal_power = np.mean(audio ** 2)
        
        # Estimate noise floor from quietest 10% of samples
        sorted_squared = np.sort(audio ** 2)
        noise_floor = np.mean(sorted_squared[:len(sorted_squared) // 10])
        
        snr_db = 10 * np.log10((signal_power / (noise_floor + 1e-8)))
        
        # Silence ratio
        silence_threshold = 0.01
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        
        # Dynamic range
        dynamic_range_db = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-8))
        
        return {
            'snr_db': float(snr_db),
            'silence_ratio': float(silence_ratio),
            'dynamic_range_db': float(dynamic_range_db),
            'peak_amplitude': float(np.max(np.abs(audio))),
            'rms_level': float(np.sqrt(np.mean(audio ** 2)))
        }

class DatasetAugmenter:
    """Applies audio augmentations to create diverse training data"""
    
    def __init__(self, config: OnDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def augment_audio(self, audio: np.ndarray, 
                     augmentation_type: str) -> np.ndarray:
        """Apply specified augmentation to audio"""
        if augmentation_type == 'time_stretch':
            return self._time_stretch(audio)
        elif augmentation_type == 'pitch_shift':
            return self._pitch_shift(audio)
        elif augmentation_type == 'add_noise':
            return self._add_noise(audio)
        elif augmentation_type == 'volume_change':
            return self._volume_change(audio)
        elif augmentation_type == 'reverb':
            return self._add_reverb(audio)
        else:
            self.logger.warning(f"Unknown augmentation type: {augmentation_type}")
            return audio
    
    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Apply time stretching (0.8x to 1.2x speed)"""
        stretch_factor = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    def _pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shifting (±2 semitones)"""
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(
            audio, 
            sr=self.config.sample_rate, 
            n_steps=n_steps
        )
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add white noise (SNR between 30-60 dB)"""
        snr_db = random.uniform(30, 60)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def _volume_change(self, audio: np.ndarray) -> np.ndarray:
        """Change volume (0.5x to 2.0x)"""
        volume_factor = random.uniform(0.5, 2.0)
        return audio * volume_factor
    
    def _add_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Add simple reverb effect"""
        # Simple reverb using delayed copies
        delay_samples = int(0.03 * self.config.sample_rate)  # 30ms delay
        decay_factor = 0.3
        
        reverb_audio = audio.copy()
        if len(audio) > delay_samples:
            reverb_audio[delay_samples:] += audio[:-delay_samples] * decay_factor
        
        return reverb_audio
    
    def create_augmented_versions(self, audio: np.ndarray, 
                                 num_augmentations: int) -> List[np.ndarray]:
        """Create multiple augmented versions of audio"""
        augmentations = [
            'time_stretch', 'pitch_shift', 'add_noise', 
            'volume_change', 'reverb'
        ]
        
        augmented_audios = []
        for i in range(num_augmentations):
            augmented = audio.copy()
            
            # Apply 1-3 random augmentations
            num_to_apply = random.randint(1, 3)
            selected_augs = random.sample(augmentations, num_to_apply)
            
            for aug_type in selected_augs:
                augmented = self.augment_audio(augmented, aug_type)
            
            # Ensure output is same length as input
            if len(augmented) != len(audio):
                if len(augmented) > len(audio):
                    # Trim
                    start_idx = (len(augmented) - len(audio)) // 2
                    augmented = augmented[start_idx:start_idx + len(audio)]
                else:
                    # Pad
                    pad_length = len(audio) - len(augmented)
                    augmented = np.pad(augmented, (0, pad_length), mode='constant')
            
            augmented_audios.append(augmented)
        
        return augmented_audios

class OnDeviceDatasetPreparer:
    """Main class for preparing on-device datasets"""
    
    def __init__(self, config: OnDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.location_manager = LocationManager(config.locations_config)
        self.device_manager = DeviceManager(config.devices_config)
        
        if HAS_AUDIO_RECORDING:
            self.audio_recorder = AudioRecorder(config)
        else:
            self.audio_recorder = None
            
        if HAS_AUDIO_PROCESSING:
            self.audio_processor = AudioProcessor(config)
            self.augmenter = DatasetAugmenter(config)
        else:
            self.audio_processor = None
            self.augmenter = None
        
        # Dataset statistics
        self.dataset_stats = {
            'total_recordings': 0,
            'recordings_per_class': defaultdict(int),
            'recordings_per_location': defaultdict(int),
            'augmented_recordings': 0,
            'failed_recordings': 0,
            'processing_time': 0.0
        }
        
        # Create output directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create output directory structure"""
        base_dir = Path(self.config.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        dirs_to_create = [
            'audio/train', 'audio/val', 'audio/test',
            'audio/augmented',
            'metadata',
            'labels',
            'quality_control'
        ]
        
        for dir_name in dirs_to_create:
            (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created directory structure in {base_dir}")
    
    def record_class_samples(self, class_name: str, 
                           num_recordings: int,
                           location_name: Optional[str] = None) -> List[str]:
        """Record audio samples for a specific class"""
        if not self.audio_recorder:
            raise RuntimeError("Audio recording not available")
        
        recorded_files = []
        location = self.location_manager.get_location(location_name) if location_name else None
        
        self.logger.info(f"Recording {num_recordings} samples for class '{class_name}'")
        
        for i in range(num_recordings):
            try:
                # Record audio
                audio_data = self.audio_recorder.record_audio(self.config.duration)
                
                # Create metadata
                metadata = {
                    'class': class_name,
                    'recording_index': i,
                    'location': location_name,
                    'location_info': location,
                    'duration': self.config.duration,
                    'sample_rate': self.config.sample_rate,
                    'recorded_at': datetime.now().isoformat(),
                    'raw_audio_stats': {
                        'length': len(audio_data),
                        'peak_amplitude': float(np.max(np.abs(audio_data))),
                        'rms_level': float(np.sqrt(np.mean(audio_data ** 2)))
                    }
                }
                
                # Process audio
                if self.audio_processor:
                    processed_audio, processing_metadata = self.audio_processor.process_audio(
                        audio_data, metadata
                    )
                    metadata.update(processing_metadata)
                else:
                    processed_audio = audio_data
                
                # Check quality
                if self._check_audio_quality(processed_audio, metadata):
                    # Save audio file
                    filename = f"{class_name}_{i:04d}_{location_name or 'default'}.wav"
                    filepath = Path(self.config.output_dir) / 'audio' / 'raw' / filename
                    filepath.parent.mkdir(exist_ok=True)
                    
                    sf.write(str(filepath), processed_audio, self.config.sample_rate)
                    
                    # Save metadata
                    metadata_filepath = Path(self.config.output_dir) / 'metadata' / f"{filename}.json"
                    with open(metadata_filepath, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    recorded_files.append(str(filepath))
                    self.dataset_stats['recordings_per_class'][class_name] += 1
                    if location_name:
                        self.dataset_stats['recordings_per_location'][location_name] += 1
                    
                    self.logger.info(f"Successfully recorded {filename}")
                else:
                    self.dataset_stats['failed_recordings'] += 1
                    self.logger.warning(f"Recording {i} for class {class_name} failed quality check")
                    
            except Exception as e:
                self.logger.error(f"Error recording sample {i} for class {class_name}: {e}")
                self.dataset_stats['failed_recordings'] += 1
        
        return recorded_files
    
    def _check_audio_quality(self, audio: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Check if recorded audio meets quality requirements"""
        quality_metrics = metadata.get('quality_metrics', {})
        
        # Check SNR
        if quality_metrics.get('snr_db', 0) < self.config.min_snr_db:
            return False
        
        # Check silence ratio
        if quality_metrics.get('silence_ratio', 1) > self.config.max_silence_ratio:
            return False
        
        # Check duration
        duration = len(audio) / self.config.sample_rate
        if duration < self.config.min_duration or duration > self.config.max_duration:
            return False
        
        return True
    
    def process_existing_audio(self, audio_dir: str) -> int:
        """Process existing audio files instead of recording"""
        if not self.audio_processor:
            raise RuntimeError("Audio processing not available")
        
        audio_files = list(Path(audio_dir).glob("*.wav"))
        processed_count = 0
        
        self.logger.info(f"Processing {len(audio_files)} existing audio files")
        
        for audio_file in audio_files:
            try:
                # Load audio
                audio_data, sr = librosa.load(str(audio_file), sr=self.config.sample_rate)
                
                # Extract class from filename or directory structure
                class_name = self._extract_class_from_path(audio_file)
                
                # Create metadata
                metadata = {
                    'class': class_name,
                    'source_file': str(audio_file),
                    'duration': len(audio_data) / sr,
                    'sample_rate': sr,
                    'processed_at': datetime.now().isoformat()
                }
                
                # Process audio
                processed_audio, processing_metadata = self.audio_processor.process_audio(
                    audio_data, metadata
                )
                metadata.update(processing_metadata)
                
                # Check quality
                if self._check_audio_quality(processed_audio, metadata):
                    # Save processed audio
                    output_filename = f"processed_{audio_file.stem}.wav"
                    output_path = Path(self.config.output_dir) / 'audio' / 'raw' / output_filename
                    output_path.parent.mkdir(exist_ok=True)
                    
                    sf.write(str(output_path), processed_audio, self.config.sample_rate)
                    
                    # Save metadata
                    metadata_path = Path(self.config.output_dir) / 'metadata' / f"{output_filename}.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    processed_count += 1
                    self.dataset_stats['recordings_per_class'][class_name] += 1
                    
                else:
                    self.dataset_stats['failed_recordings'] += 1
                    self.logger.warning(f"File {audio_file} failed quality check")
                    
            except Exception as e:
                self.logger.error(f"Error processing {audio_file}: {e}")
                self.dataset_stats['failed_recordings'] += 1
        
        return processed_count
    
    def _extract_class_from_path(self, audio_file: Path) -> str:
        """Extract class name from file path or name"""
        # Try to match with known classes
        for class_name in self.config.classes:
            if class_name.lower() in str(audio_file).lower():
                return class_name
        
        # Default to parent directory name
        return audio_file.parent.name
    
    def create_augmented_dataset(self, source_dir: Optional[str] = None) -> int:
        """Create augmented versions of the dataset"""
        if not self.augmenter:
            raise RuntimeError("Audio augmentation not available")
        
        # Use raw audio directory if not specified
        if source_dir is None:
            source_dir = Path(self.config.output_dir) / 'audio' / 'raw'
        
        source_path = Path(source_dir)
        if not source_path.exists():
            self.logger.error(f"Source directory not found: {source_dir}")
            return 0
        
        audio_files = list(source_path.glob("*.wav"))
        augmented_count = 0
        
        self.logger.info(f"Creating augmented dataset from {len(audio_files)} files")
        
        # Create augmentations for each file
        for audio_file in audio_files:
            try:
                # Load original audio
                audio_data, sr = librosa.load(str(audio_file), sr=self.config.sample_rate)
                
                # Load metadata if available
                metadata_file = Path(self.config.output_dir) / 'metadata' / f"{audio_file.name}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {'class': self._extract_class_from_path(audio_file)}
                
                # Create augmented versions
                augmented_audios = self.augmenter.create_augmented_versions(
                    audio_data, self.config.augmentation_factor
                )
                
                # Save augmented versions
                for i, augmented_audio in enumerate(augmented_audios):
                    aug_filename = f"aug_{i}_{audio_file.stem}.wav"
                    aug_path = Path(self.config.output_dir) / 'audio' / 'augmented' / aug_filename
                    
                    sf.write(str(aug_path), augmented_audio, self.config.sample_rate)
                    
                    # Create augmented metadata
                    aug_metadata = metadata.copy()
                    aug_metadata.update({
                        'original_file': str(audio_file),
                        'augmentation_index': i,
                        'augmented_at': datetime.now().isoformat(),
                        'is_augmented': True
                    })
                    
                    # Save augmented metadata
                    aug_metadata_path = Path(self.config.output_dir) / 'metadata' / f"{aug_filename}.json"
                    with open(aug_metadata_path, 'w') as f:
                        json.dump(aug_metadata, f, indent=2)
                    
                    augmented_count += 1
                    self.dataset_stats['augmented_recordings'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error augmenting {audio_file}: {e}")
        
        return augmented_count
    
    def split_dataset(self, include_augmented: bool = True) -> Dict[str, List[str]]:
        """Split dataset into train/val/test sets"""
        # Collect all audio files
        audio_dirs = [Path(self.config.output_dir) / 'audio' / 'raw']
        if include_augmented:
            audio_dirs.append(Path(self.config.output_dir) / 'audio' / 'augmented')
        
        files_by_class = defaultdict(list)
        
        for audio_dir in audio_dirs:
            if audio_dir.exists():
                for audio_file in audio_dir.glob("*.wav"):
                    # Get class from metadata
                    metadata_file = Path(self.config.output_dir) / 'metadata' / f"{audio_file.name}.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        class_name = metadata.get('class', 'unknown')
                    else:
                        class_name = self._extract_class_from_path(audio_file)
                    
                    files_by_class[class_name].append(str(audio_file))
        
        # Split each class proportionally
        splits = {'train': [], 'val': [], 'test': []}
        
        for class_name, files in files_by_class.items():
            # Shuffle files
            random.shuffle(files)
            
            n_files = len(files)
            n_train = int(n_files * self.config.train_ratio)
            n_val = int(n_files * self.config.val_ratio)
            n_test = n_files - n_train - n_val
            
            splits['train'].extend(files[:n_train])
            splits['val'].extend(files[n_train:n_train + n_val])
            splits['test'].extend(files[n_train + n_val:])
            
            self.logger.info(f"Class {class_name}: {n_train} train, {n_val} val, {n_test} test")
        
        # Copy files to split directories
        for split_name, files in splits.items():
            split_dir = Path(self.config.output_dir) / 'audio' / split_name
            split_dir.mkdir(exist_ok=True)
            
            for file_path in files:
                src_file = Path(file_path)
                dst_file = split_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                
                # Copy corresponding metadata
                metadata_src = Path(self.config.output_dir) / 'metadata' / f"{src_file.name}.json"
                metadata_dst = Path(self.config.output_dir) / 'metadata' / split_name / f"{src_file.name}.json"
                metadata_dst.parent.mkdir(exist_ok=True)
                if metadata_src.exists():
                    shutil.copy2(metadata_src, metadata_dst)
        
        return splits
    
    def create_class_labels(self):
        """Create class label mappings and files"""
        # Create class index mapping
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.config.classes)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        
        labels_dir = Path(self.config.output_dir) / 'labels'
        
        # Save class mappings
        with open(labels_dir / 'class_to_idx.json', 'w') as f:
            json.dump(class_to_idx, f, indent=2)
        
        with open(labels_dir / 'idx_to_class.json', 'w') as f:
            json.dump(idx_to_class, f, indent=2)
        
        # Create label files for each split
        for split in ['train', 'val', 'test']:
            split_dir = Path(self.config.output_dir) / 'audio' / split
            if split_dir.exists():
                label_file = labels_dir / f"{split}_labels.txt"
                with open(label_file, 'w') as f:
                    for audio_file in sorted(split_dir.glob("*.wav")):
                        # Get class from metadata
                        metadata_file = Path(self.config.output_dir) / 'metadata' / split / f"{audio_file.name}.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            class_name = metadata.get('class', 'unknown')
                        else:
                            class_name = self._extract_class_from_path(audio_file)
                        
                        class_idx = class_to_idx.get(class_name, -1)
                        f.write(f"{audio_file.name},{class_name},{class_idx}\n")
        
        self.logger.info(f"Created class label files for {len(self.config.classes)} classes")
    
    def generate_dataset_statistics(self):
        """Generate comprehensive dataset statistics"""
        stats = {
            'dataset_info': {
                'name': 'StreamSplit On-Device Dataset',
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'total_classes': len(self.config.classes),
                'classes': list(self.config.classes),
                'sample_rate': self.config.sample_rate,
                'target_duration': self.config.duration
            },
            'recording_statistics': self.dataset_stats,
            'split_statistics': {},
            'class_distribution': {},
            'audio_quality': {},
            'locations': {
                'total_locations': len(self.location_manager.get_all_locations()),
                'locations': [loc['name'] for loc in self.location_manager.get_all_locations()]
            }
        }
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            split_dir = Path(self.config.output_dir) / 'audio' / split
            if split_dir.exists():
                audio_files = list(split_dir.glob("*.wav"))
                durations = []
                class_counts = defaultdict(int)
                quality_metrics = {
                    'snr_db': [],
                    'silence_ratio': [],
                    'dynamic_range_db': []
                }
                
                for audio_file in audio_files:
                    # Load metadata
                    metadata_file = Path(self.config.output_dir) / 'metadata' / split / f"{audio_file.name}.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Collect statistics
                        duration = metadata.get('duration', 0)
                        durations.append(duration)
                        
                        class_name = metadata.get('class', 'unknown')
                        class_counts[class_name] += 1
                        
                        # Quality metrics
                        quality = metadata.get('quality_metrics', {})
                        for metric in quality_metrics:
                            if metric in quality:
                                quality_metrics[metric].append(quality[metric])
                
                # Calculate statistics
                split_stats = {
                    'total_files': len(audio_files),
                    'total_duration_hours': sum(durations) / 3600,
                    'average_duration': np.mean(durations) if durations else 0,
                    'duration_std': np.std(durations) if durations else 0,
                    'class_distribution': dict(class_counts)
                }
                
                # Quality statistics
                for metric, values in quality_metrics.items():
                    if values:
                        split_stats[f'{metric}_mean'] = np.mean(values)
                        split_stats[f'{metric}_std'] = np.std(values)
                        split_stats[f'{metric}_min'] = np.min(values)
                        split_stats[f'{metric}_max'] = np.max(values)
                
                stats['split_statistics'][split] = split_stats
        
        # Save statistics
        stats_file = Path(self.config.output_dir) / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report(stats)
        
        return stats
    
    def _generate_summary_report(self, stats: Dict):
        """Generate a human-readable summary report"""
        report_path = Path(self.config.output_dir) / 'dataset_summary.md'
        
        with open(report_path, 'w') as f:
            f.write("# StreamSplit On-Device Dataset Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset overview
            f.write("## Dataset Overview\n\n")
            f.write(f"- **Total Classes**: {stats['dataset_info']['total_classes']}\n")
            f.write(f"- **Classes**: {', '.join(stats['dataset_info']['classes'])}\n")
            f.write(f"- **Sample Rate**: {stats['dataset_info']['sample_rate']} Hz\n")
            f.write(f"- **Target Duration**: {stats['dataset_info']['target_duration']} seconds\n\n")
            
            # Recording statistics
            f.write("## Recording Statistics\n\n")
            rec_stats = stats['recording_statistics']
            f.write(f"- **Total Recordings**: {rec_stats['total_recordings']}\n")
            f.write(f"- **Augmented Recordings**: {rec_stats['augmented_recordings']}\n")
            f.write(f"- **Failed Recordings**: {rec_stats['failed_recordings']}\n\n")
            
            # Split statistics
            f.write("## Dataset Splits\n\n")
            for split_name, split_stats in stats['split_statistics'].items():
                f.write(f"### {split_name.capitalize()} Set\n\n")
                f.write(f"- **Files**: {split_stats['total_files']}\n")
                f.write(f"- **Duration**: {split_stats['total_duration_hours']:.2f} hours\n")
                f.write(f"- **Average Duration**: {split_stats['average_duration']:.2f}s\n\n")
                
                # Class distribution
                f.write("**Class Distribution:**\n\n")
                for class_name, count in split_stats['class_distribution'].items():
                    f.write(f"- {class_name}: {count} files\n")
                f.write("\n")
                
                # Quality metrics
                if 'snr_db_mean' in split_stats:
                    f.write("**Audio Quality:**\n\n")
                    f.write(f"- SNR: {split_stats['snr_db_mean']:.1f} ± {split_stats['snr_db_std']:.1f} dB\n")
                    f.write(f"- Silence Ratio: {split_stats['silence_ratio_mean']:.3f} ± {split_stats['silence_ratio_std']:.3f}\n")
                    f.write(f"- Dynamic Range: {split_stats['dynamic_range_db_mean']:.1f} ± {split_stats['dynamic_range_db_std']:.1f} dB\n\n")
            
            # Locations
            f.write("## Recording Locations\n\n")
            f.write(f"- **Total Locations**: {stats['locations']['total_locations']}\n")
            f.write(f"- **Locations Used**: {', '.join(stats['locations']['locations'])}\n\n")
        
        self.logger.info(f"Summary report saved to {report_path}")
    
    def prepare_dataset(self, record_new: bool = False, 
                       existing_audio_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to prepare the complete on-device dataset
        
        Args:
            record_new: Whether to record new audio samples
            existing_audio_dir: Directory containing existing audio files to process
            
        Returns:
            Dictionary containing preparation results
        """
        start_time = time.time()
        self.logger.info("Starting on-device dataset preparation...")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'recorded_files': 0,
            'processed_files': 0,
            'augmented_files': 0,
            'splits_created': False,
            'labels_created': False,
            'statistics_generated': False
        }
        
        try:
            # Step 1: Record new audio or process existing files
            if record_new and self.audio_recorder:
                # Record samples for each class
                for class_name in self.config.classes:
                    locations = self.location_manager.get_all_locations()
                    recordings_per_location = self.config.recordings_per_class // max(1, len(locations))
                    
                    for location in locations:
                        recorded_files = self.record_class_samples(
                            class_name, 
                            recordings_per_location,
                            location['name']
                        )
                        results['recorded_files'] += len(recorded_files)
                
            elif existing_audio_dir:
                # Process existing audio files
                results['processed_files'] = self.process_existing_audio(existing_audio_dir)
            
            # Step 2: Create augmented dataset
            if self.config.apply_augmentations and self.augmenter:
                results['augmented_files'] = self.create_augmented_dataset()
            
            # Step 3: Split dataset
            splits = self.split_dataset(include_augmented=self.config.apply_augmentations)
            results['splits_created'] = True
            results['splits'] = {k: len(v) for k, v in splits.items()}
            
            # Step 4: Create class labels
            self.create_class_labels()
            results['labels_created'] = True
            
            # Step 5: Generate statistics
            stats = self.generate_dataset_statistics()
            results['statistics_generated'] = True
            results['dataset_statistics'] = stats
            
            # Update final results
            self.dataset_stats['processing_time'] = time.time() - start_time
            results['end_time'] = datetime.now().isoformat()
            results['total_processing_time'] = self.dataset_stats['processing_time']
            results['final_statistics'] = self.dataset_stats
            
            self.logger.info(f"Dataset preparation completed in {self.dataset_stats['processing_time']:.2f} seconds")
            self.logger.info(f"Total files: {results['recorded_files'] + results['processed_files'] + results['augmented_files']}")
            
        except Exception as e:
            self.logger.error(f"Error during dataset preparation: {e}")
            results['error'] = str(e)
            raise
        
        return results
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the prepared dataset for completeness and quality"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'class_validation': {},
            'split_validation': {},
            'quality_validation': {}
        }
        
        base_dir = Path(self.config.output_dir)
        
        # Check directory structure
        required_dirs = [
            'audio/train', 'audio/val', 'audio/test',
            'metadata', 'labels'
        ]
        
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            if not dir_path.exists():
                validation_results['errors'].append(f"Missing directory: {dir_name}")
                validation_results['is_valid'] = False
        
        # Validate class distribution
        for class_name in self.config.classes:
            class_files = {
                'train': len(list((base_dir / 'audio/train').glob(f"*{class_name}*.wav"))),
                'val': len(list((base_dir / 'audio/val').glob(f"*{class_name}*.wav"))),
                'test': len(list((base_dir / 'audio/test').glob(f"*{class_name}*.wav")))
            }
            
            total_files = sum(class_files.values())
            if total_files == 0:
                validation_results['errors'].append(f"No files found for class: {class_name}")
                validation_results['is_valid'] = False
            elif total_files < 10:
                validation_results['warnings'].append(f"Class {class_name} has only {total_files} files")
            
            validation_results['class_validation'][class_name] = class_files
        
        # Validate splits
        for split in ['train', 'val', 'test']:
            split_dir = base_dir / 'audio' / split
            if split_dir.exists():
                files = list(split_dir.glob("*.wav"))
                split_results = {
                    'total_files': len(files),
                    'has_metadata': 0,
                    'has_labels': False
                }
                
                # Check metadata existence
                for audio_file in files:
                    metadata_file = base_dir / 'metadata' / split / f"{audio_file.name}.json"
                    if metadata_file.exists():
                        split_results['has_metadata'] += 1
                
                # Check label file
                label_file = base_dir / 'labels' / f"{split}_labels.txt"
                split_results['has_labels'] = label_file.exists()
                
                validation_results['split_validation'][split] = split_results
                
                if split_results['total_files'] == 0:
                    validation_results['errors'].append(f"No files in {split} split")
                    validation_results['is_valid'] = False
        
        # Quality validation
        quality_issues = 0
        total_files = 0
        
        for split in ['train', 'val', 'test']:
            metadata_dir = base_dir / 'metadata' / split
            if metadata_dir.exists():
                for metadata_file in metadata_dir.glob("*.json"):
                    total_files += 1
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    quality_metrics = metadata.get('quality_metrics', {})
                    
                    # Check SNR
                    if quality_metrics.get('snr_db', 0) < self.config.min_snr_db:
                        quality_issues += 1
                    
                    # Check silence ratio
                    if quality_metrics.get('silence_ratio', 1) > self.config.max_silence_ratio:
                        quality_issues += 1
        
        quality_ratio = (total_files - quality_issues) / max(1, total_files)
        validation_results['quality_validation'] = {
            'total_files_checked': total_files,
            'quality_issues': quality_issues,
            'quality_ratio': quality_ratio
        }
        
        if quality_ratio < 0.8:
            validation_results['warnings'].append(f"Quality issues in {quality_issues}/{total_files} files")
        
        # Save validation report
        validation_file = base_dir / 'validation_report.json'
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return validation_results


def create_ondevice_dataset(config: OnDeviceConfig, 
                          record_new: bool = False,
                          existing_audio_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Factory function to create on-device dataset with configuration
    
    Args:
        config: OnDeviceConfig instance
        record_new: Whether to record new audio samples
        existing_audio_dir: Directory with existing audio files to process
        
    Returns:
        Dictionary containing preparation results
    """
    preparer = OnDeviceDatasetPreparer(config)
    return preparer.prepare_dataset(record_new=record_new, existing_audio_dir=existing_audio_dir)


def main():
    """CLI interface for on-device dataset preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare StreamSplit On-Device Dataset')
    parser.add_argument('--config', type=str, default='experiments/on-device/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, help='Output directory for dataset')
    parser.add_argument('--record-new', action='store_true',
                       help='Record new audio samples')
    parser.add_argument('--existing-audio', type=str,
                       help='Directory containing existing audio files to process')
    parser.add_argument('--no-augment', action='store_true',
                       help='Skip augmentation step')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset after preparation')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config = OnDeviceConfig.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = OnDeviceConfig()
            logger.info("Using default configuration")
        
        # Override output directory if specified
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # Override augmentation setting
        if args.no_augment:
            config.apply_augmentations = False
        
        # Check dependencies
        if args.record_new and not HAS_AUDIO_RECORDING:
            logger.error("Audio recording requested but PyAudio not available. Install with: pip install pyaudio")
            sys.exit(1)
        
        if not HAS_AUDIO_PROCESSING:
            logger.error("Audio processing libraries not available. Install with: pip install librosa soundfile")
            sys.exit(1)
        
        # Prepare dataset
        logger.info("Starting on-device dataset preparation...")
        
        results = create_ondevice_dataset(
            config,
            record_new=args.record_new,
            existing_audio_dir=args.existing_audio
        )
        
        # Print results summary
        print("\n=== Dataset Preparation Results ===")
        print(f"Output Directory: {config.output_dir}")
        print(f"Recorded Files: {results.get('recorded_files', 0)}")
        print(f"Processed Files: {results.get('processed_files', 0)}")
        print(f"Augmented Files: {results.get('augmented_files', 0)}")
        print(f"Processing Time: {results.get('total_processing_time', 0):.2f} seconds")
        
        if 'splits' in results:
            print("\nDataset Splits:")
            for split_name, count in results['splits'].items():
                print(f"  {split_name}: {count} files")
        
        # Validate dataset if requested
        if args.validate:
            logger.info("Validating prepared dataset...")
            preparer = OnDeviceDatasetPreparer(config)
            validation_results = preparer.validate_dataset()
            
            print(f"\n=== Validation Results ===")
            print(f"Dataset Valid: {validation_results['is_valid']}")
            print(f"Warnings: {len(validation_results['warnings'])}")
            print(f"Errors: {len(validation_results['errors'])}")
            
            if validation_results['warnings']:
                print("\nWarnings:")
                for warning in validation_results['warnings']:
                    print(f"  - {warning}")
            
            if validation_results['errors']:
                print("\nErrors:")
                for error in validation_results['errors']:
                    print(f"  - {error}")
        
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()