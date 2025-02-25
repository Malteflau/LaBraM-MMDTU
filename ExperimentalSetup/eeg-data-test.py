#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for validating the EEG data prepared with LaBraM.

This script tests the integrity of the prepared EEG data for the Force Game 
social interaction experiment, ensuring correct dimensions, consistency 
between metadata and EEG data, and proper condition mappings.

Author: Magnus Evensen, Malte Færgemann Lau
Project: Bachelor's Project - EEG Social Interaction
"""

import os
import numpy as np
import pandas as pd
import pickle
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
import sys
from tqdm import tqdm

def test_metadata_structure(metadata_path):
    """Test basic structure and content of metadata DataFrame."""
    print("\n=== Testing Metadata Structure ===")
    
    # Load metadata
    metadata_df = pd.read_pickle(metadata_path)
    
    # Basic info
    print(f"Metadata shape: {metadata_df.shape}")
    print(f"Number of participants: {metadata_df['participant_id'].nunique()}")
    print(f"Number of epochs: {len(metadata_df)}")
    
    # Check for required columns
    required_columns = [
        'participant_id', 'triad_id', 'epoch_idx', 'condition', 
        'eeg_reference', 'friend_status', 'condition_type'
    ]
    
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        print(f"WARNING: Missing required columns: {missing_columns}")
    else:
        print("All required columns present: OK")
    
    # Check condition distribution
    print("\nCondition distribution:")
    condition_counts = metadata_df['condition'].value_counts()
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count}")
    
    # Check participant distribution
    print("\nEpochs per participant (sample):")
    epochs_per_participant = metadata_df.groupby('participant_id').size()
    for participant, count in epochs_per_participant.head().items():
        print(f"  {participant}: {count}")
    
    min_epochs = epochs_per_participant.min()
    max_epochs = epochs_per_participant.max()
    print(f"\nEpoch count range: {min_epochs} - {max_epochs}")
    
    return metadata_df


def test_h5_data_structure(h5_path):
    """Test basic structure and content of HDF5 EEG data."""
    print("\n=== Testing HDF5 Data Structure ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        # List participants
        participants = list(h5file.keys())
        print(f"Number of participants in H5 file: {len(participants)}")
        print(f"Participants (sample): {participants[:5]}")
        
        # Sample a few participants to check data structure
        for participant in participants[:3]:
            if 'epochs' in h5file[participant]:
                epochs = h5file[participant]['epochs']
                print(f"\nParticipant {participant}:")
                print(f"  Epochs shape: {epochs.shape}")
                print(f"  EEG channels: {epochs.shape[1]}")
                print(f"  Time points: {epochs.shape[2]}")
                
                # Basic statistics of first epoch
                first_epoch = epochs[0]
                print(f"  First epoch stats:")
                print(f"    Min: {np.min(first_epoch):.2f}")
                print(f"    Max: {np.max(first_epoch):.2f}")
                print(f"    Mean: {np.mean(first_epoch):.2f}")
                print(f"    Std: {np.std(first_epoch):.2f}")
            else:
                print(f"\nWARNING: Participant {participant} missing 'epochs' dataset")
        
        # Check consistency of epoch shapes
        shapes = []
        for participant in participants:
            if 'epochs' in h5file[participant]:
                shapes.append(h5file[participant]['epochs'].shape)
        
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            print("\nWARNING: Inconsistent epoch shapes across participants:")
            for shape in unique_shapes:
                shape_count = shapes.count(shape)
                print(f"  {shape}: {shape_count} participants")
        else:
            print("\nConsistent epoch shapes across participants: OK")
    
    return participants


def test_eeg_metadata_consistency(metadata_df, h5_path):
    """Test consistency between metadata and EEG data."""
    print("\n=== Testing Metadata-EEG Consistency ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        h5_participants = set(h5file.keys())
        metadata_participants = set(metadata_df['participant_id'].unique())
        
        # Check participant consistency
        print("Comparing participants in metadata vs HDF5...")
        participants_in_metadata_not_h5 = metadata_participants - h5_participants
        participants_in_h5_not_metadata = h5_participants - metadata_participants
        
        if participants_in_metadata_not_h5:
            print(f"WARNING: {len(participants_in_metadata_not_h5)} participants in metadata but not in HDF5")
            print(f"  Sample: {list(participants_in_metadata_not_h5)[:5]}")
        
        if participants_in_h5_not_metadata:
            print(f"WARNING: {len(participants_in_h5_not_metadata)} participants in HDF5 but not in metadata")
            print(f"  Sample: {list(participants_in_h5_not_metadata)[:5]}")
        
        if not participants_in_metadata_not_h5 and not participants_in_h5_not_metadata:
            print("Participant consistency: OK")
        
        # Check random epoch references
        print("\nTesting random epoch references...")
        success_count = 0
        test_count = min(20, len(metadata_df))
        
        samples = metadata_df.sample(test_count)
        for _, row in samples.iterrows():
            participant_id, epoch_idx = row['eeg_reference']
            
            if participant_id not in h5file:
                print(f"WARNING: Participant {participant_id} not found in HDF5")
                continue
            
            try:
                epoch_data = h5file[participant_id]['epochs'][epoch_idx]
                success_count += 1
            except Exception as e:
                print(f"ERROR: Could not access epoch {epoch_idx} for participant {participant_id}: {e}")
        
        print(f"Successfully accessed {success_count}/{test_count} sample epochs")
        
        # Check epoch count consistency
        print("\nChecking epoch count consistency...")
        mismatch_count = 0
        checked_count = 0
        
        for participant in h5_participants.intersection(metadata_participants):
            checked_count += 1
            h5_epoch_count = h5file[participant]['epochs'].shape[0]
            metadata_epoch_count = metadata_df[metadata_df['participant_id'] == participant].shape[0]
            
            if h5_epoch_count != metadata_epoch_count:
                print(f"WARNING: Epoch count mismatch for {participant}: HDF5={h5_epoch_count}, Metadata={metadata_epoch_count}")
                mismatch_count += 1
        
        if mismatch_count == 0:
            print(f"Epoch count consistency across {checked_count} participants: OK")
        else:
            print(f"WARNING: Found {mismatch_count}/{checked_count} participants with epoch count mismatches")


def test_mapping_files(mapping_dir, metadata_df):
    """Test condition mapping files."""
    print("\n=== Testing Condition Mapping Files ===")
    
    mapping_path = Path(mapping_dir)
    if not mapping_path.exists():
        print(f"WARNING: Mapping directory {mapping_path} does not exist")
        return
    
    mapping_files = list(mapping_path.glob("*.pkl"))
    print(f"Found {len(mapping_files)} mapping files")
    
    for mapping_file in mapping_files:
        mapping_name = mapping_file.stem
        print(f"\nTesting mapping: {mapping_name}")
        
        try:
            with open(mapping_file, 'rb') as f:
                mapping = pickle.load(f)
            
            train_indices = set(mapping.get('train_indices', []))
            test_indices = set(mapping.get('test_indices', []))
            
            # Check for index overlap
            overlap = train_indices.intersection(test_indices)
            if overlap:
                print(f"WARNING: Found {len(overlap)} overlapping indices between train and test")
            else:
                print("No train/test overlap: OK")
            
            # Check indices are valid for metadata
            max_index = len(metadata_df) - 1
            invalid_train = [idx for idx in train_indices if idx > max_index]
            invalid_test = [idx for idx in test_indices if idx > max_index]
            
            if invalid_train:
                print(f"WARNING: Found {len(invalid_train)} invalid train indices")
            
            if invalid_test:
                print(f"WARNING: Found {len(invalid_test)} invalid test indices")
            
            if not invalid_train and not invalid_test:
                print("All indices valid: OK")
            
            # Check class distribution
            print("Class distribution:")
            if 'y' in metadata_df.columns:
                if train_indices:
                    train_labels = metadata_df.iloc[list(train_indices)]['y'].value_counts()
                    print(f"  Train: {train_labels.to_dict()}")
                
                if test_indices:
                    test_labels = metadata_df.iloc[list(test_indices)]['y'].value_counts()
                    print(f"  Test: {test_labels.to_dict()}")
            else:
                print("  Cannot check class distribution ('y' column not in metadata)")
                
        except Exception as e:
            print(f"ERROR testing mapping file {mapping_file}: {e}")


def test_plot_sample_epochs(metadata_df, h5_path, output_dir, num_samples=3):
    """Plot sample epochs to visualize EEG data."""
    print("\n=== Creating Sample Epoch Plots ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with h5py.File(h5_path, 'r') as h5file:
        # Sample random epochs
        samples = metadata_df.sample(num_samples)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            participant_id, epoch_idx = row['eeg_reference']
            
            if participant_id not in h5file:
                print(f"WARNING: Participant {participant_id} not found in HDF5")
                continue
            
            try:
                epoch_data = h5file[participant_id]['epochs'][epoch_idx]
                
                # Create plot
                plt.figure(figsize=(12, 8))
                
                # Plot 6 random channels
                channels = random.sample(range(epoch_data.shape[0]), min(6, epoch_data.shape[0]))
                
                for j, channel in enumerate(channels):
                    plt.subplot(len(channels), 1, j+1)
                    
                    # Plot a 2-second segment
                    time_slice = slice(0, min(1000, epoch_data.shape[1]))  # 2 seconds at 500 Hz
                    plt.plot(epoch_data[channel, time_slice])
                    
                    plt.title(f"Channel {channel}")
                    if j == len(channels) - 1:
                        plt.xlabel("Time (samples)")
                
                plt.suptitle(f"Participant: {participant_id}, Epoch: {epoch_idx}, Condition: {row['condition']}")
                plt.tight_layout()
                
                # Save plot
                filename = f"sample_epoch_{participant_id}_{epoch_idx}.png"
                plt.savefig(output_dir / filename)
                plt.close()
                
                print(f"Saved sample plot: {filename}")
                
            except Exception as e:
                print(f"ERROR: Could not create plot for {participant_id}, epoch {epoch_idx}: {e}")


def test_labram_compatibility(h5_path):
    """Test compatibility with LaBraM format requirements."""
    print("\n=== Testing LaBraM Compatibility ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        participants = list(h5file.keys())
        if not participants:
            print("WARNING: No participants found in HDF5 file")
            return
        
        # Check a sample participant
        participant = participants[0]
        epochs = h5file[participant]['epochs']
        
        # Check dimensions
        print(f"EEG dimensions for {participant}:")
        print(f"  Shape: {epochs.shape}")
        print(f"  Number of channels: {epochs.shape[1]}")
        print(f"  Time points per epoch: {epochs.shape[2]}")
        
        # Check value range (LaBraM expects values normalized to 0.1 mV)
        flat_data = epochs[:10].reshape(-1)  # Use first 10 epochs for speed
        data_min, data_max = np.min(flat_data), np.max(flat_data)
        data_mean, data_std = np.mean(flat_data), np.std(flat_data)
        
        print("\nValue range statistics (first 10 epochs):")
        print(f"  Min: {data_min:.2f}")
        print(f"  Max: {data_max:.2f}")
        print(f"  Mean: {data_mean:.2f}")
        print(f"  Std: {data_std:.2f}")
        
        # Verify data is in expected range for LaBraM
        if data_min < -10000 or data_max > 10000:
            print("WARNING: Data values outside expected range for LaBraM (-10000 to 10000)")
        else:
            print("Data value range compatible with LaBraM: OK")


def find_epoch_duration(h5_path):
    """Calculate the duration of epochs in seconds."""
    print("\n=== Calculating Epoch Duration ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        participants = list(h5file.keys())
        if not participants:
            print("WARNING: No participants found in HDF5 file")
            return
        
        # Check a sample participant
        participant = participants[0]
        epochs = h5file[participant]['epochs']
        
        # Get time points per epoch
        time_points = epochs.shape[2]
        
        # Using known sampling rate (500 Hz) from Force Game Methods doc
        sampling_rate = 500  # Hz
        duration_seconds = time_points / sampling_rate
        
        print(f"Epoch information:")
        print(f"  Time points per epoch: {time_points}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration: {duration_seconds:.2f} seconds")
        
        # Check if duration is close to expected (6 seconds according to doc)
        expected_duration = 6.0  # seconds
        if abs(duration_seconds - expected_duration) > 0.1:
            print(f"WARNING: Epoch duration ({duration_seconds:.2f}s) differs from expected ({expected_duration:.2f}s)")
        else:
            print(f"Epoch duration matches expected value: OK")
            
        return duration_seconds, time_points, sampling_rate


def check_memory_requirements(h5_path):
    """Estimate memory requirements for loading a batch of epochs."""
    print("\n=== Estimating Memory Requirements ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        # Get total size of HDF5 file
        file_size_bytes = os.path.getsize(h5_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Count total epochs
        total_epochs = 0
        for participant in h5file.keys():
            if 'epochs' in h5file[participant]:
                total_epochs += h5file[participant]['epochs'].shape[0]
        
        # Get shape of a typical epoch
        participant = list(h5file.keys())[0]
        epoch_shape = h5file[participant]['epochs'][0].shape
        
        # Memory per epoch (assuming float32)
        bytes_per_element = 4  # float32
        memory_per_epoch = np.prod(epoch_shape) * bytes_per_element
        memory_per_epoch_mb = memory_per_epoch / (1024 * 1024)
        
        # Standard batch size for LaBraM
        batch_size = 64
        memory_per_batch_mb = memory_per_epoch_mb * batch_size
        
        print(f"HDF5 file size: {file_size_mb:.2f} MB")
        print(f"Total epochs: {total_epochs}")
        print(f"Typical epoch shape: {epoch_shape}")
        print(f"Memory per epoch: {memory_per_epoch_mb:.2f} MB")
        print(f"Memory for batch of {batch_size}: {memory_per_batch_mb:.2f} MB")
        
        # Warn if batch memory is high
        if memory_per_batch_mb > 4000:  # 4 GB
            print(f"WARNING: Large memory requirement for batch. Consider reducing batch size.")
        else:
            print(f"Memory requirements for standard batch size are reasonable: OK")


def compute_channel_statistics(h5_path, num_epochs_to_sample=50):
    """Compute channel-wise statistics across a sample of epochs."""
    print("\n=== Computing Channel Statistics ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        participants = list(h5file.keys())
        
        # Initialize
        channel_means = []
        channel_stds = []
        channel_mins = []
        channel_maxs = []
        
        # Sample random epochs
        epochs_sampled = 0
        
        for participant in tqdm(participants, desc="Processing participants"):
            if epochs_sampled >= num_epochs_to_sample:
                break
                
            if 'epochs' in h5file[participant]:
                epochs = h5file[participant]['epochs']
                num_available = min(epochs.shape[0], num_epochs_to_sample - epochs_sampled)
                
                indices = np.random.choice(epochs.shape[0], num_available, replace=False)
                
                for idx in indices:
                    epoch = epochs[idx]
                    
                    # Compute statistics per channel
                    means = np.mean(epoch, axis=1)
                    stds = np.std(epoch, axis=1)
                    mins = np.min(epoch, axis=1)
                    maxs = np.max(epoch, axis=1)
                    
                    channel_means.append(means)
                    channel_stds.append(stds)
                    channel_mins.append(mins)
                    channel_maxs.append(maxs)
                    
                    epochs_sampled += 1
        
        # Aggregate statistics
        if channel_means:
            channel_means = np.array(channel_means)
            channel_stds = np.array(channel_stds)
            channel_mins = np.array(channel_mins)
            channel_maxs = np.array(channel_maxs)
            
            # Get global statistics per channel
            global_means = np.mean(channel_means, axis=0)
            global_stds = np.mean(channel_stds, axis=0)
            global_mins = np.min(channel_mins, axis=0)
            global_maxs = np.max(channel_maxs, axis=0)
            
            print(f"Statistics computed across {epochs_sampled} sampled epochs")
            
            # Print summary
            print("\nChannel-wise statistics summary:")
            print(f"  Mean value range: {np.min(global_means):.2f} to {np.max(global_means):.2f}")
            print(f"  Std value range: {np.min(global_stds):.2f} to {np.max(global_stds):.2f}")
            print(f"  Min value: {np.min(global_mins):.2f}")
            print(f"  Max value: {np.max(global_maxs):.2f}")
            
            # Identify potentially problematic channels
            extreme_channels = np.where(np.abs(global_means) > 100)[0]
            if len(extreme_channels) > 0:
                print(f"\nWARNING: {len(extreme_channels)} channels have extreme mean values:")
                for ch in extreme_channels[:10]:  # Show first 10
                    print(f"  Channel {ch}: mean={global_means[ch]:.2f}")
                if len(extreme_channels) > 10:
                    print(f"  ... and {len(extreme_channels) - 10} more")
            
            noisy_channels = np.where(global_stds > 200)[0]
            if len(noisy_channels) > 0:
                print(f"\nWARNING: {len(noisy_channels)} channels may be noisy (high std):")
                for ch in noisy_channels[:10]:  # Show first 10
                    print(f"  Channel {ch}: std={global_stds[ch]:.2f}")
                if len(noisy_channels) > 10:
                    print(f"  ... and {len(noisy_channels) - 10} more")
        else:
            print("No epochs were sampled for analysis")


def run_all_tests(metadata_path, h5_path, mapping_dir, output_dir):
    """Run all test functions."""
    metadata_df = test_metadata_structure(metadata_path)
    participants = test_h5_data_structure(h5_path)
    test_eeg_metadata_consistency(metadata_df, h5_path)
    test_mapping_files(mapping_dir, metadata_df)
    test_plot_sample_epochs(metadata_df, h5_path, output_dir)
    test_labram_compatibility(h5_path)
    epoch_duration, time_points, sampling_rate = find_epoch_duration(h5_path)
    check_memory_requirements(h5_path)
    compute_channel_statistics(h5_path)
    
    print("\n=== Test Summary ===")
    print(f"Metadata file: {metadata_path}")
    print(f"HDF5 file: {h5_path}")
    print(f"Mapping directory: {mapping_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nNumber of participants: {len(participants)}")
    print(f"Number of epochs: {len(metadata_df)}")
    print(f"Epoch duration: {epoch_duration:.2f} seconds ({time_points} time points @ {sampling_rate} Hz)")
    print("\nTest suite completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EEG data integrity")
    parser.add_argument("--metadata", type=str, default="./DataProcessed/metadata.pkl",
                        help="Path to metadata DataFrame")
    parser.add_argument("--h5", type=str, default="./DataProcessed/eeg_data.h5",
                        help="Path to HDF5 EEG data file")
    parser.add_argument("--mappings", type=str, default="./DataProcessed/condition_mappings",
                        help="Path to condition mappings directory")
    parser.add_argument("--output", type=str, default="./test_output",
                        help="Directory for saving output plots and reports")
    parser.add_argument("--all", action="store_true", 
                        help="Run all tests")
    
    # Individual test flags
    parser.add_argument("--test-metadata", action="store_true", 
                        help="Test metadata structure")
    parser.add_argument("--test-h5", action="store_true",
                        help="Test HDF5 structure")
    parser.add_argument("--test-consistency", action="store_true",
                        help="Test metadata-EEG consistency")
    parser.add_argument("--test-mappings", action="store_true",
                        help="Test condition mappings")
    parser.add_argument("--plot-samples", action="store_true",
                        help="Create sample epoch plots")
    parser.add_argument("--check-compatibility", action="store_true",
                        help="Check LaBraM compatibility")
    parser.add_argument("--check-duration", action="store_true",
                        help="Calculate epoch duration")
    parser.add_argument("--check-memory", action="store_true",
                        help="Estimate memory requirements")
    parser.add_argument("--compute-stats", action="store_true",
                        help="Compute channel statistics")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Flag to track if any individual test was selected
    any_test_selected = any([
        args.test_metadata, args.test_h5, args.test_consistency, args.test_mappings,
        args.plot_samples, args.check_compatibility, args.check_duration,
        args.check_memory, args.compute_stats
    ])
    
    # If no specific tests selected or --all flag is set, run all tests
    if args.all or not any_test_selected:
        run_all_tests(args.metadata, args.h5, args.mappings, args.output)
    else:
        # Load metadata for tests that need it
        metadata_df = None
        if args.test_metadata or args.test_consistency or args.test_mappings or args.plot_samples:
            metadata_df = pd.read_pickle(args.metadata)
        
        # Run selected tests
        if args.test_metadata:
            metadata_df = test_metadata_structure(args.metadata)
        
        if args.test_h5:
            test_h5_data_structure(args.h5)
        
        if args.test_consistency:
            if metadata_df is None:
                metadata_df = pd.read_pickle(args.metadata)
            test_eeg_metadata_consistency(metadata_df, args.h5)
        
        if args.test_mappings:
            if metadata_df is None:
                metadata_df = pd.read_pickle(args.metadata)
            test_mapping_files(args.mappings, metadata_df)
        
        if args.plot_samples:
            if metadata_df is None:
                metadata_df = pd.read_pickle(args.metadata)
            test_plot_sample_epochs(metadata_df, args.h5, args.output)
        
        if args.check_compatibility:
            test_labram_compatibility(args.h5)
        
        if args.check_duration:
            find_epoch_duration(args.h5)
        
        if args.check_memory:
            check_memory_requirements(args.h5)
        
        if args.compute_stats:
            compute_channel_statistics(args.h5)
        
        print("\nSelected tests completed successfully")
