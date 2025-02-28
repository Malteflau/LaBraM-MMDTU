#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Efficient data preparation module for social interaction EEG data with LaBraM.

This module prepares EEG data from the Force Game experiment for use with the
LaBraM (Large Brain Model) framework, storing data efficiently in HDF5 format.

Author: Magnus Evensen, Malte Færgemann Lau
Project: Bachelor's Project - EEG Social Interaction
"""

import os
import mne
import numpy as np
import pandas as pd
import pickle
import h5py
import dill  # For serializing functions
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional, Union
import random
import gc


def create_combined_df(eeg_folder_path: str, overview_path: str, 
                       behavior_path: str = None) -> pd.DataFrame:
    """
    Create a combined DataFrame with EEG data and metadata.
    
    Args:
        eeg_folder_path: Path to folder containing preprocessed EEG .fif files
        overview_path: Path to the overview DataFrame pickle file
        behavior_path: Path to the behavioral features DataFrame pickle file
        
    Returns:
        Combined DataFrame with metadata (without EEG data)
    """
    # Load metadata dataframes
    with open(overview_path, "rb") as f:
        overview_df = pickle.load(f)
    
    if behavior_path:
        with open(behavior_path, "rb") as f:
            behavior_df = pickle.load(f)
    else:
        behavior_df = None

    # Initialize lists to store data
    all_data = []

    # Get list of all .fif files
    fif_files = [f for f in os.listdir(eeg_folder_path) if f.endswith('.fif')]
    
    # Create EEG data structure
    eeg_data_dict = {}
    fif_files = fif_files[:10]  # For testing purposes
    # Process each file
    for fif_file in fif_files:
        # Extract participant ID from filename
        participant_id = fif_file.split('_')[0]
        triad_id = int(participant_id[:3])
        
        # Get participant overview data
        if not overview_df[overview_df['Exp_id'] == participant_id].empty:
            participant_overview = overview_df[overview_df['Exp_id'] == participant_id].iloc[0]
            
            try:
                # Load epochs
                epochs = mne.read_epochs(os.path.join(eeg_folder_path, fif_file), preload=True)
                eeg_data_np = epochs.get_data()
                
                # Normalize EEG data to 0.1 mV as LaBraM expects
                eeg_data_np = eeg_data_np * 10000
                
                # Store EEG data for this participant
                eeg_data_dict[participant_id] = eeg_data_np
                
                # Get event IDs to condition mapping
                event_id_to_condition = {v: k for k, v in epochs.event_id.items()}
                
                # Extract participant id info
                subject_id = participant_overview['Subject_id']
                friend_status = participant_overview['Friend_status']
                participant_position = participant_id[-1]  # A, B, or C
                
                # For each epoch
                for epoch_idx in range(len(epochs)):
                    # Get condition for this epoch
                    event_id = epochs.events[epoch_idx][2]
                    condition = event_id_to_condition[event_id]
                    
                    # Create row for each epoch
                    row_data = {
                        # Participant information
                        'participant_id': participant_id,
                        'triad_id': triad_id,
                        'subject_id': subject_id,
                        'friend_status': friend_status,
                        'participant_position': participant_position,  # A, B, or C
                        'age': participant_overview['Age'],
                        'gender': participant_overview['Gender'],
                        'class_friends': participant_overview['Class_friends'],
                        'class_close_friends': participant_overview['Class_close_friends'],
                        'friends': participant_overview['Friends'],
                        'close_friends': participant_overview['Close_friends'],
                        
                        # Trial information
                        'epoch_idx': epoch_idx,
                        'condition': condition,
                        
                        # Device information
                        'eeg_device': participant_overview['EEG_device'],
                        'force_device': participant_overview['Force_device'],
                        'force_port': participant_overview['Force_port'],
                        
                        # EEG reference (tuple of participant_id and epoch_idx)
                        'eeg_reference': (participant_id, epoch_idx)
                    }
                    
                    # Add behavioral features if available
                    if behavior_df is not None:
                        # Find matching behavioral data based on triad_id, participant, and condition
                        beh_match = behavior_df[
                            (behavior_df['Triad_id'] == triad_id) & 
                            (behavior_df['Participant'] == f'P{participant_position}') &
                            (behavior_df['Condition'] == condition)
                        ]
                        
                        if not beh_match.empty:
                            # Extract behavioral metrics
                            for feature in ['Latency', 'Stability', 'Success', 'RelativeForce', 'cVariability']:
                                feature_rows = beh_match[beh_match['Feature'] == feature]
                                if not feature_rows.empty:
                                    row_data[f'beh_{feature.lower()}'] = feature_rows['Value'].values[0]
                    
                    all_data.append(row_data)
                    
                # Free up memory
                del epochs
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {fif_file}: {e}")
        else:
            print(f"Warning: No overview data found for participant {participant_id}")

    # Create DataFrame
    combined_df = pd.DataFrame(all_data)
    
    if not combined_df.empty:
        # Parse condition codes into meaningful components
        combined_df['condition_type'] = combined_df['condition'].apply(parse_condition_type)
        combined_df['has_feedback'] = combined_df['condition'].apply(lambda x: 'Pn' not in x)
        
        print("Metadata DataFrame shape:", combined_df.shape)
        print("\nColumns:", combined_df.columns.tolist())
        print("\nParticipants:", combined_df['participant_id'].nunique())
        print("Total epochs:", len(combined_df))
        print("\nConditions:", combined_df['condition'].unique())
    else:
        print("Warning: No data was processed successfully.")
    
    return combined_df, eeg_data_dict


def save_eeg_data_to_h5(eeg_data_dict: Dict[str, np.ndarray], output_file: Path):
    """
    Save EEG data to HDF5 format.
    
    Args:
        eeg_data_dict: Dictionary of participant_id -> EEG data array
        output_file: Path to save the HDF5 file
    """
    with h5py.File(output_file, 'w') as h5file:
        for participant_id, eeg_data in eeg_data_dict.items():
            # Create group for this participant
            participant_group = h5file.create_group(participant_id)
            
            # Save EEG data
            participant_group.create_dataset('epochs', data=eeg_data, 
                                           chunks=(1, eeg_data.shape[1], min(1000, eeg_data.shape[2])),
                                           compression='gzip', compression_opts=4)
            
    print(f"Saved EEG data to {output_file}")


def parse_condition_type(condition):
    """Parse condition string to extract condition type."""
    # Examples of conditions: T1P, T1Pn, T3P, T3Pn, T12P, T12Pn, T13P, T13Pn, T23P, T23Pn
    # T1 = solo participant 1
    # T3 = solo participant 3
    # T12 = duo participants 1 and 2
    # T13 = duo participants 1 and 3
    # T23 = duo participants 2 and 3
    # 'P' = with continuous feedback
    # 'Pn' = without continuous feedback
    
    # Extract condition type (note: we need to handle both "P" and "Pn" variants)
    condition_base = condition.split('P')[0]  # Get the part before P/Pn
    
    if condition_base == 'T1':
        return 'solo_p1'
    elif condition_base == 'T3':
        return 'solo_p3'
    elif condition_base == 'T12':
        return 'duo_p1p2'
    elif condition_base == 'T13':
        return 'duo_p1p3'
    elif condition_base == 'T23':
        return 'duo_p2p3'
    else:
        return 'unknown'


def create_prediction_mapping(
    metadata_df: pd.DataFrame, 
    condition_filter: Callable, 
    output_dir: Path, 
    name: str
) -> Tuple[List[int], List[int]]:
    """
    Creates train/test mappings based on conditions.
    
    Args:
        metadata_df: DataFrame with metadata
        condition_filter: Function that selects data and assigns labels
        output_dir: Directory to save files
        name: Name of this prediction setup
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    # Apply the condition filter to create a temporary DataFrame with labels
    temp_df = condition_filter(metadata_df.copy())
    
    # Remove rows where label is -1 (not applicable for this classification)
    temp_df = temp_df[temp_df['y'] != -1]
    if temp_df.empty:
        print(f"Warning: No samples for prediction setup {name}")
        return [], []
    
    # Split by participants to avoid data leakage
    all_participants = temp_df['participant_id'].unique()
    # Set random seed for reproducibility
    np.random.seed(42)
    train_participants = np.random.choice(
        all_participants, 
        size=int(len(all_participants) * 0.8), 
        replace=False
    )
    
    # Get indices for train and test sets
    train_indices = temp_df[temp_df['participant_id'].isin(train_participants)].index.tolist()
    test_indices = temp_df[~temp_df['participant_id'].isin(train_participants)].index.tolist()
    
    # Create mapping dictionary
    mapping = {
        'name': name,
        'train_indices': train_indices,
        'test_indices': test_indices,
        # We can't directly serialize the function, but we can use dill to serialize a reference
        'label_function_name': condition_filter.__name__
    }
    
    # Save mapping
    mapping_dir = output_dir / 'condition_mappings'
    mapping_dir.mkdir(exist_ok=True, parents=True)
    
    with open(mapping_dir / f"{name}.pkl", 'wb') as f:
        pickle.dump(mapping, f)
    
    print(f"Created {name} prediction mapping:")
    print(f"  Train: {len(train_indices)} samples from {len(train_participants)} participants")
    print(f"  Test: {len(test_indices)} samples from {len(all_participants) - len(train_participants)} participants")
    print(f"  Class distribution in train: {pd.Series(temp_df.iloc[train_indices]['y']).value_counts().to_dict()}")
    
    return train_indices, test_indices


# Define label mapping functions for different prediction setups

def friend_vs_nonfriend_label(row):
    """Convert friend status to binary label."""
    return 1 if row['friend_status'] == 'Yes' else 0

def solo_vs_duo_label(row):
    """Convert condition to solo vs duo binary label."""
    # Extract condition type 
    condition_type = row['condition_type']
    
    if condition_type.startswith('solo'):
        return 0  # Solo condition
    elif condition_type.startswith('duo'):
        return 1  # Duo condition
    else:
        return -1  # Not applicable

def feedback_vs_nofeedback_label(row):
    """Convert condition to feedback vs no feedback binary label."""
    # In your data, conditions with 'Pn' at the end are no feedback
    # Conditions with just 'P' at the end have feedback
    return 1 if row['has_feedback'] else 0

def participant_position_label(row):
    """Create label based on participant position in the triad."""
    participant_pos = row['participant_position']
    
    if participant_pos == 'A':
        return 0
    elif participant_pos == 'B':
        return 1
    elif participant_pos == 'C':
        return 2
    else:
        return -1

def success_high_low_label(row):
    """Create binary label for high/low success."""
    if 'beh_success' in row and not pd.isna(row['beh_success']):
        return 1 if row['beh_success'] > 0.5 else 0
    return -1

def relative_force_high_low_label(row, threshold=None):
    """Create binary label for high/low relative force."""
    if 'beh_relativeforce' in row and not pd.isna(row['beh_relativeforce']):
        if threshold is None:
            # This will be replaced with the actual median in the filter function
            return row['beh_relativeforce']
        return 1 if row['beh_relativeforce'] > threshold else 0
    return -1

# Define condition filter functions for different prediction setups

def friend_vs_nonfriend_filter(df):
    """Filter for friend vs non-friend prediction."""
    df['y'] = df.apply(friend_vs_nonfriend_label, axis=1)
    return df

def solo_vs_duo_filter(df):
    """Filter for solo vs duo condition prediction."""
    df['y'] = df.apply(solo_vs_duo_label, axis=1)
    # Keep only rows where label is not -1
    return df

def feedback_filter(df):
    """Filter for feedback vs no feedback prediction."""
    df['y'] = df.apply(feedback_vs_nofeedback_label, axis=1)
    return df

def participant_position_filter(df):
    """Filter for predicting participant position in the triad."""
    df['y'] = df.apply(participant_position_label, axis=1)
    return df

def success_high_low_filter(df):
    """Filter for predicting high vs low success based on behavioral data."""
    # Only keep rows that have the success behavioral feature
    filtered_df = df[df['beh_success'].notna()].copy()
    # Binary classification: 1 for high success (>0.5), 0 for low success (<=0.5)
    filtered_df['y'] = filtered_df.apply(success_high_low_label, axis=1)
    return filtered_df

def relative_force_high_low_filter(df):
    """Filter for predicting high vs low relative force based on behavioral data."""
    # Only keep rows that have the relative force behavioral feature
    filtered_df = df[df['beh_relativeforce'].notna()].copy()
    # Use median as threshold for high vs low relative force
    median_value = filtered_df['beh_relativeforce'].median()
    filtered_df['y'] = filtered_df.apply(
        lambda row: relative_force_high_low_label(row, threshold=median_value), 
        axis=1
    )
    return filtered_df


def test_data_integration(metadata_df, eeg_h5_path):
    """
    Test function to verify data integration.
    
    Args:
        metadata_df: DataFrame with metadata
        eeg_h5_path: Path to the HDF5 file with EEG data
    """
    print("\n--- Data Integration Test ---")
    
    # 1. Basic DataFrame info
    print(f"Total epochs: {len(metadata_df)}")
    print(f"Total participants: {metadata_df['participant_id'].nunique()}")
    print(f"Participants: {metadata_df['participant_id'].unique()[:5]}...")
    
    # 2. Condition distribution
    print("\nCondition distribution:")
    condition_counts = metadata_df['condition'].value_counts()
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count}")
    
    # 3. Friend status distribution
    print("\nFriend status distribution:")
    friend_counts = metadata_df['friend_status'].value_counts()
    for status, count in friend_counts.items():
        print(f"  {status}: {count}")
    
    # 4. Check behavioral feature integration
    beh_columns = [col for col in metadata_df.columns if col.startswith('beh_')]
    if beh_columns:
        print("\nBehavioral features:")
        for col in beh_columns:
            non_null_count = metadata_df[col].notna().sum()
            print(f"  {col}: {non_null_count} non-null values ({non_null_count/len(metadata_df)*100:.1f}%)")
    
    # 5. Check EEG data
    print("\nEEG data check:")
    with h5py.File(eeg_h5_path, 'r') as h5file:
        # Get a random participant
        participants = list(h5file.keys())
        if participants:
            participant_id = participants[0]
            print(f"  Participant: {participant_id}")
            
            # Get epochs dataset
            epochs = h5file[participant_id]['epochs']
            print(f"  EEG data shape: {epochs.shape}")
            print(f"  First epoch shape: {epochs[0].shape}")
            
            # Get a sample from the first epoch
            sample = epochs[0, :, :100]  # First 100 time points of first epoch
            print(f"  Min value: {np.min(sample)}")
            print(f"  Max value: {np.max(sample)}")
        else:
            print("  No participants found in HDF5 file")
    
    # 6. Verify condition parsing
    print("\nCondition type parsing:")
    condition_type_counts = metadata_df['condition_type'].value_counts()
    for condition_type, count in condition_type_counts.items():
        print(f"  {condition_type}: {count}")
    
    # 7. Sample a few rows to verify mapping
    print("\nSample mapping check:")
    sample_idx = np.random.randint(0, len(metadata_df), size=3)
    for idx in sample_idx:
        row = metadata_df.iloc[idx]
        print(f"\n  Sample {idx}:")
        print(f"    Participant: {row['participant_id']}, Friend status: {row['friend_status']}")
        print(f"    Condition: {row['condition']}, Type: {row['condition_type']}, Feedback: {row['has_feedback']}")
        print(f"    EEG reference: {row['eeg_reference']}")
        
        # Check if behavioral features exist
        beh_features = {col: row[col] for col in beh_columns if col in row and not pd.isna(row[col])}
        if beh_features:
            print("    Behavioral features:")
            for feature, value in beh_features.items():
                print(f"      {feature}: {value}")
        else:
            print("    No behavioral features found for this sample")
            
        # Check that we can access EEG data using the reference
        with h5py.File(eeg_h5_path, 'r') as h5file:
            participant_id, epoch_idx = row['eeg_reference']
            if participant_id in h5file:
                eeg_data = h5file[participant_id]['epochs'][epoch_idx]
                print(f"    Successfully accessed EEG data with shape {eeg_data.shape}")
            else:
                print(f"    Could not find participant {participant_id} in HDF5 file")


def save_label_functions(output_dir):
    """
    Save label functions to allow loading of serialized functions.
    
    Args:
        output_dir: Output directory
    """
    func_dir = output_dir / 'functions'
    func_dir.mkdir(exist_ok=True, parents=True)
    
    # Save each label function
    functions = {
        'friend_vs_nonfriend_label': friend_vs_nonfriend_label,
        'solo_vs_duo_label': solo_vs_duo_label,
        'feedback_vs_nofeedback_label': feedback_vs_nofeedback_label,
        'participant_position_label': participant_position_label,
        'success_high_low_label': success_high_low_label,
        'relative_force_high_low_label': relative_force_high_low_label
    }
    
    for name, func in functions.items():
        with open(func_dir / f"{name}.dill", 'wb') as f:
            dill.dump(func, f)
    
    print(f"Saved label functions to {func_dir}")


if __name__ == "__main__":
    # Define paths
    eeg_folder_path = os.path.abspath('./FG_Data/PreprocessedEEGData')
    overview_path = os.path.abspath('./FG_Data/FG_overview_df_v2.pkl')
    behavior_path = "./FG_Data/Beh_feat_df_v2.pkl"
    
    output_dir = Path("./DataProcessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create combined DataFrame and process EEG data
    metadata_df, eeg_data_dict = create_combined_df(
        eeg_folder_path, 
        overview_path, 
        behavior_path=behavior_path
    )
    
    # 2. Save metadata DataFrame (without EEG data)
    metadata_df.to_pickle(output_dir / "metadata.pkl")
    print(f"\nSaved metadata DataFrame to {output_dir / 'metadata.pkl'}")
    
    # 3. Save EEG data to HDF5
    eeg_h5_path = output_dir / "eeg_data.h5"
    save_eeg_data_to_h5(eeg_data_dict, eeg_h5_path)
    
    # 4. Create prediction mappings
    create_prediction_mapping(
        metadata_df, friend_vs_nonfriend_filter, output_dir, "friend_prediction")
    
    create_prediction_mapping(
        metadata_df, solo_vs_duo_filter, output_dir, "solo_vs_duo")
    
    create_prediction_mapping(
        metadata_df, feedback_filter, output_dir, "feedback_prediction")
    
    create_prediction_mapping(
        metadata_df, participant_position_filter, output_dir, "participant_position")
    
    # Additional behavioral prediction mappings
    if 'beh_success' in metadata_df.columns:
        create_prediction_mapping(
            metadata_df, success_high_low_filter, output_dir, "success_high_low")
    
    if 'beh_relativeforce' in metadata_df.columns:
        create_prediction_mapping(
            metadata_df, relative_force_high_low_filter, output_dir, "relativeforce_high_low")
    
    # 5. Save label functions
    save_label_functions(output_dir)
    
    # 6. Test data integration
    test_data_integration(metadata_df, eeg_h5_path)
    
    print("\nEfficient data preparation complete!")
