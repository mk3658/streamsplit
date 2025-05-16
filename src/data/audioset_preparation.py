"""
Example usage of AudioSet preparation module
This script demonstrates how to prepare AudioSet data for StreamSplit
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audioset_preparation import AudioSetPreparer, AudioSetConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_preparation():
    """Example of basic AudioSet preparation"""
    print("=== Basic AudioSet Preparation ===")
    
    # Create configuration
    config = AudioSetConfig(
        output_dir="/data/audioset_small",
        subset_size=100,  # Small subset for testing
        duration=10.0,
        sample_rate=16000,
        max_workers=4
    )
    
    # Create preparer
    preparer = AudioSetPreparer(config)
    
    # Prepare dataset
    processed_segments = preparer.prepare_dataset()
    
    print(f"Successfully processed {len(processed_segments)} segments")
    
    return processed_segments

def example_config_from_yaml():
    """Example of loading configuration from YAML"""
    print("\n=== Configuration from YAML ===")
    
    # Load configuration from the StreamSplit config file
    config_path = "experiments/audioset/config.yaml"
    
    if os.path.exists(config_path):
        config = AudioSetConfig.from_yaml(config_path)
        print(f"Loaded config from {config_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Subset size: {config.subset_size}")
        
        # Create preparer and run
        preparer = AudioSetPreparer(config)
        processed_segments = preparer.prepare_dataset()
        
        return processed_segments
    else:
        print(f"Config file not found: {config_path}")
        return None

def example_custom_processing():
    """Example with custom processing parameters"""
    print("\n=== Custom Processing Example ===")
    
    config = AudioSetConfig(
        output_dir="/data/audioset_custom",
        subset_size=500,
        duration=10.0,
        sample_rate=22050,  # Higher sample rate
        max_workers=8,
        normalize_audio=True,
        remove_silence=True,
        min_duration=9.5,
        max_duration=10.5,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    preparer = AudioSetPreparer(config)
    processed_segments = preparer.prepare_dataset()
    
    return processed_segments

def example_download_only():
    """Example of download-only phase"""
    print("\n=== Download Only Example ===")
    
    config = AudioSetConfig(
        output_dir="/data/audioset_raw",
        subset_size=50,
        max_workers=4,
        cache_dir="/tmp/audioset_download"
    )
    
    preparer = AudioSetPreparer(config)
    
    # Download metadata
    preparer.metadata.download_metadata()
    
    # Get segments
    segments = preparer.metadata.get_subset_segments(config.subset_size)
    print(f"Selected {len(segments)} segments")
    
    # Download audio (without processing)
    successful_segments = preparer._download_audio_parallel(segments)
    print(f"Downloaded {len(successful_segments)} audio files")
    
    return successful_segments

def example_verify_dataset():
    """Example of dataset verification"""
    print("\n=== Dataset Verification ===")
    
    dataset_path = "/data/audioset"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return False
    
    # Check directory structure
    expected_dirs = ['audio', 'labels', 'metadata']
    for dir_name in expected_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ Found {dir_name} directory")
            
            # Count files
            if dir_name == 'audio':
                audio_files = len([f for f in os.listdir(dir_path) 
                                 if f.endswith('.wav')])
                print(f"  - {audio_files} audio files")
            elif dir_name == 'labels':
                label_files = [f for f in os.listdir(dir_path) 
                              if f.endswith('.json')]
                print(f"  - Label files: {label_files}")
        else:
            print(f"✗ Missing {dir_name} directory")
    
    # Check statistics file
    stats_file = os.path.join(dataset_path, 'dataset_stats.json')
    if os.path.exists(stats_file):
        import json
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print("\nDataset Statistics:")
        print(f"  Total segments: {stats.get('total_segments', 'N/A')}")
        print(f"  Unique labels: {stats.get('unique_labels', 'N/A')}")
        print(f"  Splits: {stats.get('splits', 'N/A')}")
        
        return True
    else:
        print(f"✗ Missing statistics file")
        return False

def main():
    """Run all examples"""
    print("AudioSet Preparation Examples\n")
    
    # Check dependencies
    try:
        import yt_dlp
        print("✓ yt-dlp is available")
    except ImportError:
        try:
            import youtube_dl
            print("✓ youtube-dl is available")
        except ImportError:
            print("✗ No YouTube downloader found. Install yt-dlp or youtube-dl")
            print("Run: pip install yt-dlp")
            return
    
    # Run examples based on command line arguments
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        examples = {
            'basic': example_basic_preparation,
            'yaml': example_config_from_yaml,
            'custom': example_custom_processing,
            'download': example_download_only,
            'verify': example_verify_dataset
        }
        
        if example_name in examples:
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {list(examples.keys())}")
    else:
        # Run basic example by default
        example_basic_preparation()

if __name__ == "__main__":
    main()