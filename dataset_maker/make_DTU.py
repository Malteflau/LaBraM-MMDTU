# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Applied to the DTU Force Game Dataset
# --------------------------------------------------------
import mne
import numpy as np
import os
import pickle
import gc
import pandas as pd
from pathlib import Path

# Define standard channels to ensure data format consistency
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 
                    'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 
                    'EEG T1-REF', 'EEG T2-REF']

# Channel mapping from your dataset to standard format
channel_mapping = {
    'Fp1': 'EEG FP1-REF', 'AF7': 'EEG AF7-REF', 'AF3': 'EEG AF3-REF', 'F1': 'EEG F1-REF',
    'F3': 'EEG F3-REF', 'F5': 'EEG F5-REF', 'F7': 'EEG F7-REF', 'FT7': 'EEG FT7-REF',
    'FC5': 'EEG FC5-REF', 'FC3': 'EEG FC3-REF', 'FC1': 'EEG FC1-REF', 'C1': 'EEG C1-REF',
    'C3': 'EEG C3-REF', 'C5': 'EEG C5-REF', 'T7': 'EEG T7-REF', 'TP7': 'EEG TP7-REF',
    'CP5': 'EEG CP5-REF', 'CP3': 'EEG CP3-REF', 'CP1': 'EEG CP1-REF', 'P1': 'EEG P1-REF',
    'P3': 'EEG P3-REF', 'P5': 'EEG P5-REF', 'P7': 'EEG P7-REF', 'P9': 'EEG P9-REF',
    'PO7': 'EEG PO7-REF', 'PO3': 'EEG PO3-REF', 'O1': 'EEG O1-REF', 'Iz': 'EEG Iz-REF',
    'Oz': 'EEG Oz-REF', 'POz': 'EEG POz-REF', 'Pz': 'EEG Pz-REF', 'CPz': 'EEG CPz-REF',
    'Fpz': 'EEG Fpz-REF', 'Fp2': 'EEG FP2-REF', 'AF8': 'EEG AF8-REF', 'AF4': 'EEG AF4-REF',
    'AFz': 'EEG AFz-REF', 'Fz': 'EEG Fz-REF', 'F2': 'EEG F2-REF', 'F4': 'EEG F4-REF',
    'F6': 'EEG F6-REF', 'F8': 'EEG F8-REF', 'FT8': 'EEG FT8-REF', 'FC6': 'EEG FC6-REF',
    'FC4': 'EEG FC4-REF', 'FC2': 'EEG FC2-REF', 'FCz': 'EEG FCz-REF', 'Cz': 'EEG Cz-REF',
    'C2': 'EEG C2-REF', 'C4': 'EEG C4-REF', 'C6': 'EEG C6-REF', 'T8': 'EEG T8-REF',
    'TP8': 'EEG TP8-REF', 'CP6': 'EEG CP6-REF', 'CP4': 'EEG CP4-REF', 'CP2': 'EEG CP2-REF',
    'P2': 'EEG P2-REF', 'P4': 'EEG P4-REF', 'P6': 'EEG P6-REF', 'P8': 'EEG P8-REF',
    'P10': 'EEG P10-REF', 'PO8': 'EEG PO8-REF', 'PO4': 'EEG PO4-REF', 'O2': 'EEG O2-REF'
}

def process_file(fif_file, participant_data, output_dir):
    """
    Process a single EEG file and create samples.
    
    Args:
        fif_file: Path to the EEG file
        participant_data: List of dictionaries with participant metadata
        output_dir: Directory to save processed data
    
    Returns:
        int: Number of epochs processed
    """
    try:
        # Extract participant ID from filename
        participant_id = os.path.basename(fif_file).split('_')[0]
        triad_id = int(participant_id[:3])
        participant_position = participant_id[-1]  # A, B, or C
        
        # Map participant position to P1, P2, P3
        position_map = {'A': 'P1', 'B': 'P2', 'C': 'P3'}
        participant_num = position_map.get(participant_position, 'P1')
        
        # Get metadata from participant_data if available
        subject_info = None
        for row in participant_data:
            if row['Exp_id'] == participant_id:
                subject_info = row
                break
        
        if subject_info is None:
            print(f"Warning: No metadata found for {participant_id}")
            return 0
        
        # Load epochs - this is memory intensive
        print(f"Processing {fif_file}")
        epochs = mne.read_epochs(fif_file, preload=True)
        
        # Get event IDs to condition mapping
        event_id_to_condition = {v: k for k, v in epochs.event_id.items()}
        
        # Process each epoch
        epochs_processed = 0
        for epoch_idx, event in enumerate(epochs.events):
            event_id = event[2]
            condition = event_id_to_condition[event_id]
            
            # Extract parameters from condition
            has_feedback = 'Pn' not in condition
            
            # Get the trial data for this epoch
            eeg_data = epochs[epoch_idx].get_data()[0]
            
            # Normalize data to 0.1mV scale for LaBraM
            eeg_data = eeg_data * 10000
            
            # Determine if this condition involves this participant
            participant_involved = True
            if condition.startswith('T1') and participant_num not in ['P1']:
                participant_involved = False
            elif condition.startswith('T3') and participant_num not in ['P3']:
                participant_involved = False
            elif condition.startswith('T12') and participant_num not in ['P1', 'P2']:
                participant_involved = False
            elif condition.startswith('T13') and participant_num not in ['P1', 'P3']:
                participant_involved = False
            elif condition.startswith('T23') and participant_num not in ['P2', 'P3']:
                participant_involved = False
            
            # Create sample dictionary
            sample = {
                'X': eeg_data,
                'y': 1 if subject_info['Friend_status'] == 'Yes' else 0,
                'participant_id': participant_id,
                'triad_id': triad_id,
                'epoch_idx': epoch_idx,
                'condition': condition,
                'has_feedback': has_feedback,
                'participant_num': participant_num,
                'participant_involved': participant_involved,
                'subject_id': subject_info['Subject_id'],
                'friend_status': subject_info['Friend_status'],
                'age': subject_info['Age'],
                'gender': subject_info['Gender'],
                'class_friends': subject_info['Class_friends'],
                'class_close_friends': subject_info['Class_close_friends']
            }
            
            # Save the sample to disk immediately to reduce memory usage
            output_file = os.path.join(output_dir, f"{participant_id}_{epoch_idx}_{condition}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(sample, f)
            
            epochs_processed += 1
        
        # Free memory
        del epochs
        gc.collect()
        
        print(f"Processed {epochs_processed} epochs for {participant_id}")
        return epochs_processed
        
    except Exception as e:
        print(f"Error processing {fif_file}: {e}")
        return 0


def parse_condition(condition):
    """Parse condition code to determine trial type"""
    # Whether this has continuous feedback
    has_feedback = 'Pn' not in condition
    
    # Extract numbers to determine who's involved
    if condition.startswith('T1'):
        return 'solo_p1', has_feedback
    elif condition.startswith('T3'):
        return 'solo_p3', has_feedback
    elif condition.startswith('T12'):
        return 'duo_p1p2', has_feedback
    elif condition.startswith('T13'):
        return 'duo_p1p3', has_feedback
    elif condition.startswith('T23'):
        return 'duo_p2p3', has_feedback
    else:
        return 'unknown', has_feedback
    

def split_data_by_participant(processed_dir, output_dir):
    """
    Split processed data into train, validation, and test sets by participant.
    
    Args:
        processed_dir: Directory containing processed data files
        output_dir: Directory to save train/val/test files
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all pickle files
    all_files = [f for f in os.listdir(processed_dir) if f.endswith('.pkl')]
    
    # Extract unique participant IDs
    participant_ids = set()
    for file in all_files:
        participant_id = file.split('_')[0]
        participant_ids.add(participant_id)
    
    # Sort for reproducibility
    participant_ids = sorted(list(participant_ids))
    
    # Set seed for reproducibility
    np.random.seed(42)
    # Shuffle but in a reproducible way
    indices = np.arange(len(participant_ids))
    np.random.shuffle(indices)
    shuffled_participants = [participant_ids[i] for i in indices]
    
    # Split participants into train/val/test
    train_participants = shuffled_participants[:int(len(shuffled_participants) * 0.7)]
    val_participants = shuffled_participants[int(len(shuffled_participants) * 0.7):int(len(shuffled_participants) * 0.85)]
    test_participants = shuffled_participants[int(len(shuffled_participants) * 0.85):]
    
    print(f"Train participants: {len(train_participants)} participants")
    print(f"Val participants: {len(val_participants)} participants")
    print(f"Test participants: {len(test_participants)} participants")
    
    # Group files by participant for more efficient processing
    files_by_participant = {}
    for file in all_files:
        participant_id = file.split('_')[0]
        if participant_id not in files_by_participant:
            files_by_participant[participant_id] = []
        files_by_participant[participant_id].append(file)
    
    # Process each participant separately to reduce memory usage
    train_count, val_count, test_count = 0, 0, 0
    
    for participant_id, files in files_by_participant.items():
        # Determine which set this participant belongs to
        if participant_id in train_participants:
            target_dir = train_dir
            train_count += len(files)
        elif participant_id in val_participants:
            target_dir = val_dir
            val_count += len(files)
        else:
            target_dir = test_dir
            test_count += len(files)
        
        # Copy each file for this participant
        for file in files:
            src_path = os.path.join(processed_dir, file)
            dst_path = os.path.join(target_dir, file)
            
            # Copy the pickle file to the appropriate directory
            with open(src_path, "rb") as f_src:
                data = pickle.load(f_src)
                with open(dst_path, "wb") as f_dst:
                    pickle.dump(data, f_dst)
            
            # Delete source file after successful copy to save disk space
            os.remove(src_path)
    
    # Print summary
    print(f"Split data into:")
    print(f"  Train: {train_count} samples from {len(train_participants)} participants")
    print(f"  Validation: {val_count} samples from {len(val_participants)} participants")
    print(f"  Test: {test_count} samples from {len(test_participants)} participants")


if __name__ == "__main__":
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to project root
    root_dir = os.path.dirname(script_dir)  # Assuming script is one level below project root
    
    # Path to data directory
    data_dir = os.path.join(root_dir, "DTUDATA", "FG_Data")
    
    # Specific paths
    eeg_folder_path = os.path.join(data_dir, "PreprocessedEEGData")
    overview_path = os.path.join(data_dir, "FG_overview_df_v2.pkl")
    
    # Output paths
    processed_dir = os.path.join(data_dir, "processed")
    output_dir = os.path.join(data_dir, "LaBraM_data")
    
    # Create output directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load overview data
    with open(overview_path, "rb") as f:
        overview_df = pickle.load(f)
    
    # Convert DataFrame to list of dictionaries for easier processing
    participant_data = overview_df.to_dict('records')
    
    # Get list of all fif files
    fif_files = [os.path.join(eeg_folder_path, f) for f in os.listdir(eeg_folder_path) 
                if f.endswith('.fif') and 'preprocessed' in f]
    
    fif_files= fif_files[:3]  # For testing
    # Process files one by one to reduce memory usage
    total_epochs = 0
    for i, fif_file in enumerate(fif_files):
        print(f"Processing file {i+1}/{len(fif_files)}: {os.path.basename(fif_file)}")
        epochs_processed = process_file(fif_file, participant_data, processed_dir)
        total_epochs += epochs_processed
        
        # Force garbage collection after each file
        gc.collect()
    
    print(f"Total epochs processed: {total_epochs}")
    
    # Split data into train/val/test by participant
    split_data_by_participant(processed_dir, output_dir)
    
    print("DTU data processing complete!")