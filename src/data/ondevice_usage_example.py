"""
Example usage of On-Device Dataset Preparation Module
This script demonstrates how to prepare on-device audio datasets for StreamSplit
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ondevice_preparation import OnDeviceDatasetPreparer, OnDeviceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_recording():
    """Example of basic on-device recording and preparation"""
    print("=== Basic On-Device Recording Example ===")
    
    # Create configuration for basic recording
    config = OnDeviceConfig(
        output_dir="/data/ondevice_basic",
        num_classes=7,
        duration=10.0,
        sample_rate=16000,
        recordings_per_class=20,  # Small number for testing
        recordings_per_location=5
    )
    
    # Create preparer
    preparer = OnDeviceDatasetPreparer(config)
    
    # List available audio devices
    if preparer.audio_recorder:
        devices = preparer.audio_recorder.list_audio_devices()
        print("Available audio devices:")
        for device in devices:
            print(f"  {device['index']}: {device['name']} ({device['channels']} channels)")
    
    # Prepare dataset with recording
    results = preparer.prepare_dataset(record_new=True)
    
    print(f"Successfully recorded {results['recorded_files']} files")
    print(f"Created {results['augmented_files']} augmented files")
    print(f"Dataset splits: {results.get('splits', {})}")
    
    return results

def example_process_existing_audio():
    """Example of processing existing audio files"""
    print("\n=== Process Existing Audio Example ===")
    
    # Create configuration for processing existing files
    config = OnDeviceConfig(
        output_dir="/data/ondevice_existing",
        num_classes=7,
        duration=10.0,
        sample_rate=16000,
        normalize_audio=True,
        remove_silence=True,
        apply_augmentations=True,
        augmentation_factor=2
    )
    
    # Specify directory containing existing audio files
    existing_audio_dir = "/path/to/existing/audio"  # Update this path
    
    # Create preparer
    preparer = OnDeviceDatasetPreparer(config)
    
    # Check if the directory exists
    if os.path.exists(existing_audio_dir):
        # Process existing audio files
        results = preparer.prepare_dataset(
            record_new=False,
            existing_audio_dir=existing_audio_dir
        )
        
        print(f"Processed {results['processed_files']} existing files")
        print(f"Created {results['augmented_files']} augmented versions")
        print(f"Processing time: {results['total_processing_time']:.2f} seconds")
    else:
        print(f"Directory {existing_audio_dir} not found")
        print("Creating synthetic example instead...")
        
        # Create some synthetic audio files for demonstration
        import numpy as np
        import soundfile as sf
        
        synthetic_dir = "/tmp/synthetic_audio"
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Generate synthetic audio for each class
        for i, class_name in enumerate(config.classes):
            for j in range(3):  # 3 files per class
                # Generate synthetic audio (sine wave with noise)
                t = np.linspace(0, config.duration, int(config.sample_rate * config.duration))
                frequency = 220 * (2 ** (i / 12))  # Different frequency per class
                audio = np.sin(2 * np.pi * frequency * t) * 0.5
                audio += np.random.normal(0, 0.1, len(audio))  # Add noise
                
                # Save synthetic audio
                filename = f"{class_name.replace(' ', '_').lower()}_{j}.wav"
                filepath = os.path.join(synthetic_dir, filename)
                sf.write(filepath, audio, config.sample_rate)
        
        # Process synthetic audio
        results = preparer.prepare_dataset(
            record_new=False,
            existing_audio_dir=synthetic_dir
        )
        
        print(f"Processed {results['processed_files']} synthetic files")
        print(f"Created {results['augmented_files']} augmented versions")
    
    return results

def example_config_from_yaml():
    """Example of loading configuration from YAML file"""
    print("\n=== Configuration from YAML Example ===")
    
    # Load configuration from the StreamSplit config file
    config_path = "experiments/on-device/config.yaml"
    
    if os.path.exists(config_path):
        config = OnDeviceConfig.from_yaml(config_path)
        print(f"Loaded config from {config_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Number of classes: {config.num_classes}")
        print(f"Classes: {config.classes}")
        
        # Create preparer and run with synthetic data
        preparer = OnDeviceDatasetPreparer(config)
        
        # Since we may not have microphone access, create synthetic data
        synthetic_dir = "/tmp/yaml_synthetic_audio"
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Generate minimal synthetic dataset
        import numpy as np
        import soundfile as sf
        
        for i, class_name in enumerate(config.classes[:3]):  # Just first 3 classes
            # Create a simple audio pattern for each class
            t = np.linspace(0, config.duration, int(config.sample_rate * config.duration))
            pattern = np.sin(2 * np.pi * (100 + i * 50) * t)  # Different frequencies
            pattern += 0.2 * np.sin(2 * np.pi * (300 + i * 100) * t)  # Add harmonics
            pattern *= np.exp(-t / 5)  # Add decay
            
            filename = f"{class_name.replace(' ', '_').lower()}_example.wav"
            filepath = os.path.join(synthetic_dir, filename)
            sf.write(filepath, pattern, config.sample_rate)
        
        # Process with loaded config
        results = preparer.prepare_dataset(
            record_new=False,
            existing_audio_dir=synthetic_dir
        )
        
        print(f"Results with YAML config:")
        print(f"  Processed files: {results['processed_files']}")
        print(f"  Augmented files: {results['augmented_files']}")
        
        return results
    else:
        print(f"Config file not found: {config_path}")
        return None

def example_validation():
    """Example of dataset validation"""
    print("\n=== Dataset Validation Example ===")
    
    # Use one of the previously created datasets
    dataset_paths = [
        "/data/ondevice_basic",
        "/data/ondevice_existing", 
        "/tmp/ondevice"  # From YAML config
    ]
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"\nValidating dataset at {dataset_path}")
            
            # Create config for validation
            config = OnDeviceConfig(output_dir=dataset_path)
            preparer = OnDeviceDatasetPreparer(config)
            
            # Validate the dataset
            validation_results = preparer.validate_dataset()
            
            print(f"Dataset valid: {validation_results['is_valid']}")
            print(f"Warnings: {len(validation_results['warnings'])}")
            print(f"Errors: {len(validation_results['errors'])}")
            
            if validation_results['warnings']:
                print("Warnings:")
                for warning in validation_results['warnings']:
                    print(f"  - {warning}")
            
            if validation_results['errors']:
                print("Errors:")
                for error in validation_results['errors']:
                    print(f"  - {error}")
            
            # Show class validation details
            print("\nClass validation:")
            for class_name, class_data in validation_results['class_validation'].items():
                total = sum(class_data.values())
                print(f"  {class_name}: {total} files "
                      f"(train:{class_data['train']}, val:{class_data['val']}, test:{class_data['test']})")
            
            # Show quality metrics if available
            quality_metrics = validation_results.get('quality_validation', {})
            if quality_metrics:
                quality_ratio = quality_metrics.get('quality_ratio', 0)
                print(f"\nQuality ratio: {quality_ratio:.2%}")
            
            return validation_results
    
    print("No datasets found for validation")
    return None

def example_custom_configuration():
    """Example with custom configuration and advanced features"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration with specific settings
    config = OnDeviceConfig(
        output_dir="/data/ondevice_custom",
        
        # Dataset parameters
        num_classes=5,
        classes=["Speech", "Music", "Noise", "Silence", "Mixed"],
        duration=5.0,  # Shorter duration
        sample_rate=22050,  # Higher sample rate
        
        # Recording parameters
        recordings_per_class=10,
        recordings_per_location=5,
        
        # Processing parameters
        normalize_audio=True,
        remove_silence=True,
        apply_noise_gate=True,
        noise_gate_threshold=-35.0,  # dB
        min_duration=4.5,
        max_duration=5.5,
        
        # Augmentation parameters
        apply_augmentations=True,
        augmentation_factor=4,  # More augmentations
        
        # Quality control
        min_snr_db=15.0,  # Higher SNR requirement
        max_silence_ratio=0.2,  # Less silence allowed
        
        # Dataset splits
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Create preparer
    preparer = OnDeviceDatasetPreparer(config)
    
    # Create synthetic audio with more realistic characteristics
    synthetic_dir = "/tmp/custom_synthetic_audio"
    os.makedirs(synthetic_dir, exist_ok=True)
    
    import numpy as np
    import soundfile as sf
    
    # Generate more sophisticated synthetic audio
    for i, class_name in enumerate(config.classes):
        for j in range(5):  # 5 files per class
            t = np.linspace(0, config.duration, int(config.sample_rate * config.duration))
            
            if class_name == "Speech":
                # Simulate speech-like modulated signal
                carrier = np.sin(2 * np.pi * 150 * t)
                modulation = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
                audio = carrier * modulation
                # Add formants
                audio += 0.3 * np.sin(2 * np.pi * 800 * t) * modulation
                audio += 0.2 * np.sin(2 * np.pi * 1200 * t) * modulation
                
            elif class_name == "Music":
                # Simulate music with multiple harmonics
                fundamental = 220 * (2 ** (j / 12))  # Different notes
                audio = np.sin(2 * np.pi * fundamental * t)
                audio += 0.5 * np.sin(2 * np.pi * 2 * fundamental * t)
                audio += 0.3 * np.sin(2 * np.pi * 3 * fundamental * t)
                # Add vibrato
                vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
                audio *= vibrato
                
            elif class_name == "Noise":
                # Simulate colored noise
                audio = np.random.normal(0, 1, len(t))
                # Apply simple low-pass filter for colored noise
                for k in range(1, len(audio)):
                    audio[k] = 0.8 * audio[k-1] + 0.2 * audio[k]
                    
            elif class_name == "Silence":
                # Very low amplitude audio with minimal background
                audio = 0.01 * np.random.normal(0, 1, len(t))
                
            elif class_name == "Mixed":
                # Combination of speech and music
                speech_like = np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
                music_like = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
                audio = 0.6 * speech_like + 0.4 * music_like
            
            # Normalize and add realistic characteristics
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio *= 0.7  # Prevent clipping
            
            # Add slight background noise
            audio += 0.02 * np.random.normal(0, 1, len(audio))
            
            # Save audio file
            filename = f"{class_name.lower()}_{j:02d}.wav"
            filepath = os.path.join(synthetic_dir, filename)
            sf.write(filepath, audio, config.sample_rate)
    
    # Prepare dataset with custom configuration
    results = preparer.prepare_dataset(
        record_new=False,
        existing_audio_dir=synthetic_dir
    )
    
    print("Custom configuration results:")
    print(f"  Processed files: {results['processed_files']}")
    print(f"  Augmented files: {results['augmented_files']}")
    print(f"  Processing time: {results['total_processing_time']:.2f} seconds")
    
    # Show detailed statistics
    if results['statistics_generated']:
        stats = results['dataset_statistics']
        print(f"  Total duration: {stats['split_statistics']['train']['total_duration_hours']:.2f} hours")
        print(f"  Average SNR: {stats['split_statistics']['train'].get('snr_db_mean', 'N/A')}")
    
    return results

def example_streaming_simulation():
    """Example simulating streaming on-device recording"""
    print("\n=== Streaming Simulation Example ===")
    
    # Configuration for streaming
    config = OnDeviceConfig(
        output_dir="/data/ondevice_streaming",
        duration=2.0,  # Shorter chunks for streaming
        sample_rate=16000,
        classes=["Conversation", "Background"],
        apply_augmentations=False,  # Disable for real-time simulation
        recordings_per_class=1  # Just for setup
    )
    
    # Create preparer
    preparer = OnDeviceDatasetPreparer(config)
    
    # Simulate streaming recording
    print("Simulating streaming audio processing...")
    
    import numpy as np
    import time
    
    # Simulate real-time processing
    for chunk_id in range(5):
        print(f"Processing chunk {chunk_id + 1}/5")
        
        # Generate streaming audio chunk
        t = np.linspace(0, config.duration, int(config.sample_rate * config.duration))
        
        # Simulate conversation vs background
        if chunk_id % 2 == 0:
            # Conversation-like signal
            audio = np.sin(2 * np.pi * 200 * t) * (1 + 0.8 * np.sin(2 * np.pi * 5 * t))
            class_name = "Conversation"
        else:
            # Background noise
            audio = 0.3 * np.random.normal(0, 1, len(t))
            # Apply smoothing for realistic background
            for i in range(1, len(audio)):
                audio[i] = 0.9 * audio[i-1] + 0.1 * audio[i]
            class_name = "Background"
        
        # Simulate processing time
        start_time = time.time()
        
        # Process chunk (normally this would be a streaming buffer)
        metadata = {
            'class': class_name,
            'chunk_id': chunk_id,
            'timestamp': time.time(),
            'streaming': True
        }
        
        # In a real implementation, this would use the streaming buffer
        # For demo, we'll just save the chunk
        import soundfile as sf
        chunk_path = f"/tmp/streaming_chunk_{chunk_id}.wav"
        sf.write(chunk_path, audio, config.sample_rate)
        
        processing_time = time.time() - start_time
        print(f"  Processed {class_name} chunk in {processing_time*1000:.2f}ms")
        
        # Simulate real-time delay
        time.sleep(0.1)
    
    print("Streaming simulation completed")
    
    return True

def main():
    """Run all examples"""
    print("StreamSplit On-Device Dataset Preparation Examples\n")
    
    # Check dependencies
    try:
        import librosa
        import soundfile as sf
        print("✓ Audio processing libraries available")
    except ImportError as e:
        print(f"✗ Missing audio libraries: {e}")
        print("Install with: pip install librosa soundfile")
        return
    
    # Check for audio recording capability
    try:
        import pyaudio
        print("✓ PyAudio available for recording")
    except ImportError:
        print("✗ PyAudio not available (recording disabled)")
        print("For recording functionality, install with: pip install pyaudio")
    
    # Run examples based on command line arguments
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        examples = {
            'basic': example_basic_recording,
            'existing': example_process_existing_audio,
            'yaml': example_config_from_yaml,
            'validate': example_validation,
            'custom': example_custom_configuration,
            'streaming': example_streaming_simulation
        }
        
        if example_name in examples:
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {list(examples.keys())}")
    else:
        # Run all examples
        print("Running all examples...\n")
        
        # Start with processing existing/synthetic audio (safer than recording)
        example_config_from_yaml()
        example_process_existing_audio()
        example_custom_configuration()
        example_validation()
        example_streaming_simulation()
        
        # Only run recording example if explicitly requested
        print("\nTo run recording example, use: python ondevice_usage_example.py basic")
        print("To run specific example, use: python ondevice_usage_example.py <example_name>")
        print(f"Available examples: basic, existing, yaml, validate, custom, streaming")

if __name__ == "__main__":
    main()
