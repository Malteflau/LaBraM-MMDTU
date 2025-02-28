#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataLoader for LaBraM fine-tuning on social interaction EEG data.

This module provides a PyTorch-compatible DataLoader for the social interaction
EEG dataset, properly formatted for LaBraM fine-tuning with the correct
sampling rate and patch size.

Author: Magnus Evensen, Malte Færgemann Lau
Project: Bachelor's Project - EEG Social Interaction
"""

import os
import numpy as np
import pandas as pd
import pickle
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import signal
from typing import Dict, List, Tuple, Union, Optional, Callable
import random


class SocialEEGDataset(Dataset):
    """Dataset for fine-tuning LaBraM on social interaction EEG data."""
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        h5_path: str,
        indices: Optional[List[int]] = None,
        target_key: str = 'has_feedback',
        resample_to_hz: int = 200,
        patch_size: int = 200,
        return_all_patches: bool = True
    ):
        """
        Initialize dataset for EEG analysis with LaBraM.
        
        Args:
            metadata_df: DataFrame with metadata
            h5_path: Path to HDF5 file with EEG data
            indices: Optional list of indices to include (otherwise uses all indices)
            target_key: Column name in metadata_df for the target variable
            resample_to_hz: Target sampling rate in Hz 
            patch_size: Number of time samples per patch
            return_all_patches: Whether to return all patches per epoch (True) or a single 
                              random patch (False)
        """
        self.metadata_df = metadata_df
        self.h5_path = h5_path
        self.indices = indices if indices is not None else metadata_df.index.tolist()
        self.target_key = target_key
        self.resample_to_hz = resample_to_hz
        self.patch_size = patch_size
        self.return_all_patches = return_all_patches
        
        # Original sampling rate based on Force Game Methods doc
        self.original_sampling_rate = 500  # Hz
        
        # Get basic info about data dimensions
        self._check_data_dimensions()
    
    def _check_data_dimensions(self):
        """Check the dimensions of the EEG data."""
        # Get a sample participant and epoch to determine dimensions
        first_idx = self.indices[0]
        row = self.metadata_df.loc[first_idx]
        participant_id, epoch_idx = row['eeg_reference']
        
        # Open file temporarily for dimension checking
        with h5py.File(self.h5_path, 'r') as h5file:
            sample_epoch = h5file[participant_id]['epochs'][epoch_idx]
            
            # Store dimensions
            self.n_channels = sample_epoch.shape[0]
            self.n_timepoints_orig = sample_epoch.shape[1]
        
        # Calculate number of timepoints after resampling
        resampling_factor = self.resample_to_hz / self.original_sampling_rate
        self.n_timepoints_resampled = int(self.n_timepoints_orig * resampling_factor)
        
        # Calculate number of patches
        self.n_patches_per_epoch = self.n_timepoints_resampled // self.patch_size
        
        print(f"Dataset initialized with:")
        print(f"  {len(self.indices)} samples")
        print(f"  {self.n_channels} channels")
        print(f"  Original: {self.n_timepoints_orig} timepoints @ {self.original_sampling_rate} Hz")
        print(f"  Resampled: {self.n_timepoints_resampled} timepoints @ {self.resample_to_hz} Hz")
        print(f"  {self.n_patches_per_epoch} complete patches of size {self.patch_size} per epoch")
    
    def __len__(self):
        return len(self.indices)
    
    def _resample_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Resample EEG data from original sampling rate to target rate.
        
        Args:
            eeg_data: EEG data array of shape (channels, timepoints)
            
        Returns:
            Resampled EEG data of shape (channels, resampled_timepoints)
        """
        # Calculate number of output points
        n_out = int(eeg_data.shape[1] * (self.resample_to_hz / self.original_sampling_rate))
        
        # Resample each channel
        resampled_data = np.zeros((eeg_data.shape[0], n_out))
        for ch in range(eeg_data.shape[0]):
            resampled_data[ch] = signal.resample(eeg_data[ch], n_out)
        
        return resampled_data
    
    def _extract_patches(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract fixed-size patches from resampled EEG data.
        
        Args:
            eeg_data: Resampled EEG data array of shape (channels, timepoints)
            
        Returns:
            Patches of shape (n_patches, channels, patch_size)
        """
        n_channels, n_timepoints = eeg_data.shape
        n_patches = n_timepoints // self.patch_size
        
        # Truncate data to fit complete patches
        truncated_timepoints = n_patches * self.patch_size
        truncated_data = eeg_data[:, :truncated_timepoints]
        
        # Reshape to extract patches
        patches = truncated_data.reshape(n_channels, n_patches, self.patch_size)
        patches = np.transpose(patches, (1, 0, 2))  # (n_patches, channels, patch_size)
        
        return patches
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.
        
        Args:
            idx: Index of item to get
            
        Returns:
            Tuple of (eeg_patches, label)
            - If return_all_patches is True: 
                - eeg_patches shape: (n_patches, channels, patch_size)
            - If return_all_patches is False:
                - eeg_patches shape: (channels, patch_size)
        """
        # Get metadata
        data_idx = self.indices[idx]
        row = self.metadata_df.loc[data_idx]
        
        # Get label
        if self.target_key in row:
            label = row[self.target_key]

        else:
            # Default to 0 if target key doesn't exist
            label = 0
        
        # Get EEG data - open and close file for each item to allow multiprocessing
        participant_id, epoch_idx = row['eeg_reference']
        
        with h5py.File(self.h5_path, 'r') as h5file:
            eeg_data = h5file[participant_id]['epochs'][epoch_idx][:]  # Copy data to memory
        
        # Resample
        resampled_data = self._resample_eeg(eeg_data)
        
        # Extract patches
        patches = self._extract_patches(resampled_data)
         
        # Convert to torch tensor
        patches_tensor = torch.from_numpy(patches).float()
        
        # Return all patches or a single random patch
        if self.return_all_patches:
            return patches_tensor, label
        else:
            # Select a random patch
            patch_idx = random.randint(0, patches.shape[0] - 1)
            return patches_tensor[patch_idx], label


class PatchCollator:
    """
    Custom collator for handling patches in batches.
    
    This collator can either:
    1. Return batches with all patches from each epoch concatenated
    2. Return batches with one random patch from each epoch
    """
    
    def __init__(self, all_patches=True, max_patches_per_sample=None):
        """
        Initialize the collator.
        
        Args:
            all_patches: Whether to include all patches (True) or a single 
                        random patch per epoch (False)
            max_patches_per_sample: Maximum number of patches to include per sample
                                  (None means include all patches)
        """
        self.all_patches = all_patches
        self.max_patches_per_sample = max_patches_per_sample
    
    def __call__(self, batch):
        """
        Collate function.
        
        Args:
            batch: List of (patches, label) tuples
            
        Returns:
            Tuple of (patches, labels)
        """
        if self.all_patches:
            # Extract all patches and labels
            all_patches = []
            all_labels = []
            
            for patches, label in batch:
                # Limit number of patches if specified
                if self.max_patches_per_sample is not None:
                    n_patches = min(patches.shape[0], self.max_patches_per_sample)
                    patches = patches[:n_patches]
                
                # Add patches and corresponding labels
                all_patches.append(patches)
                all_labels.extend([label] * patches.shape[0])
            
            # Stack patches and convert labels
            patches_batch = torch.cat(all_patches, dim=0)
            labels_batch = torch.tensor(all_labels)
            
            return patches_batch, labels_batch
        else:
            # Each item is already a single patch
            patches, labels = zip(*batch)
            return torch.stack(patches), torch.tensor(labels)


def create_dataloaders(
    metadata_path: str,
    h5_path: str,
    mapping_name: str = None,
    batch_size: int = 64,
    num_workers: int = 0,
    resample_to_hz: int = 200,
    patch_size: int = 200,
    all_patches: bool = True,
    max_patches_per_sample: Optional[int] = None,
    test_ratio: float = 0.2,
    random_seed: int = 42
):
    """
    Create train and validation data loaders.
    
    Args:
        metadata_path: Path to metadata DataFrame
        h5_path: Path to HDF5 file with EEG data
        mapping_name: Name of condition mapping to use (None to create a new split)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        resample_to_hz: Target sampling rate in Hz
        patch_size: Number of time samples per patch
        all_patches: Whether to include all patches or just one random patch per epoch
        max_patches_per_sample: Maximum number of patches to include per sample
        test_ratio: Ratio of data to use for testing (if creating a new split)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load metadata
    metadata_df = pd.read_pickle(metadata_path)
    
    # Get train/test indices
    if mapping_name:
        # Load existing mapping
        mapping_dir = Path(metadata_path).parent / 'condition_mappings'
        mapping_path = mapping_dir / f"{mapping_name}.pkl"
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file {mapping_path} not found")
        
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        
        train_indices = mapping.get('train_indices', [])
        test_indices = mapping.get('test_indices', [])
        
        print(f"Using mapping: {mapping_name}")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

    else:
        # Create a new random split
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Split by participants to avoid data leakage
        participants = metadata_df['participant_id'].unique()
        n_test = int(len(participants) * test_ratio)
        
        # Random shuffle
        np.random.shuffle(participants)
        test_participants = participants[:n_test]
        train_participants = participants[n_test:]
        
        # Get indices
        train_indices = metadata_df[metadata_df['participant_id'].isin(train_participants)].index.tolist()
        test_indices = metadata_df[metadata_df['participant_id'].isin(test_participants)].index.tolist()
        
        print(f"Created new train/test split:")
        print(f"  Train: {len(train_indices)} samples from {len(train_participants)} participants")
        print(f"  Test: {len(test_indices)} samples from {len(test_participants)} participants")
    
    # Create datasets
    train_dataset = SocialEEGDataset(
        metadata_df=metadata_df,
        h5_path=h5_path,
        indices=train_indices,
        resample_to_hz=resample_to_hz,
        patch_size=patch_size,
        return_all_patches=all_patches
    )
    
    test_dataset = SocialEEGDataset(
        metadata_df=metadata_df,
        h5_path=h5_path,
        indices=test_indices,
        resample_to_hz=resample_to_hz,
        patch_size=patch_size,
        return_all_patches=all_patches
    )
    
    # Create collator
    collator = PatchCollator(all_patches=all_patches, max_patches_per_sample=max_patches_per_sample)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DataLoader for social interaction EEG data")
    parser.add_argument("--metadata", type=str, default="./DataProcessed/metadata.pkl",
                        help="Path to metadata DataFrame")
    parser.add_argument("--h5", type=str, default="./DataProcessed/eeg_data.h5",
                        help="Path to HDF5 EEG data file")
    parser.add_argument("--mapping", type=str, default="feedback_prediction",
                        help="Name of condition mapping to use")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for DataLoader")
    parser.add_argument("--all-patches", action="store_true", default=True,
                        help="Include all patches per epoch")
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Maximum number of patches per sample")
    parser.add_argument("--target-key", type=str, default="has_feedback",)
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        metadata_path=args.metadata,
        h5_path=args.h5,
        mapping_name=args.mapping,
        batch_size=args.batch_size,
        all_patches=args.all_patches,
        max_patches_per_sample=args.max_patches
    )
    
    # Test data loaders
    print("\nTesting train loader...")
    for i, (patches, labels) in enumerate(train_loader):
        print(f"Batch {i+1}: Patches shape {patches.shape}, Labels shape {labels.shape}")
        print(f"  Patch value range: {patches.min().item():.4f} to {patches.max().item():.4f}")
        print(f"  Labels: {labels.unique()} unique labels")
        
        if i >= 2:  # Just check a few batches
            break
    
    print("\nTesting test loader...")
    for i, (patches, labels) in enumerate(test_loader):
        print(f"Batch {i+1}: Patches shape {patches.shape}, Labels shape {labels.shape}")
        print(f"  Labels: {labels.tolist()[:100]}...")
        if i >= 2:  # Just check a few batches
            break
    
    print("\nDataLoader test complete")