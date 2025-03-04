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

def get_behavioral_data(behavior_path, triad_id, participant_num, condition):
    """
    Get behavioral data for a specific trial if available.
    
    Args:
        behavior_path: Path to behavior data file
        triad_id: ID of the triad
        participant_num: Participant number (P1, P2, P3)
        condition: Condition name
    
    Returns:
        dict: Dictionary of behavioral features or None if not found
    """
    try:
        with open(behavior_path, "rb") as f:
            behavior_df = pickle.load(f)
        
        # Find matching behavioral data
        beh_match = behavior_df[
            (behavior_df['Triad_id'] == triad_id) & 
            (behavior_df['Participant'] == participant_num) &
            (behavior_df['Condition'] == condition)
        ]
        
        if beh_match.empty:
            return None
        
        # Extract behavioral features
        beh_features = {}
        for feature in ['Latency', 'Stability', 'Success', 'RelativeForce', 'cVariability']:
            feature_rows = beh_match[beh_match['Feature'] == feature]
            if not feature_rows.empty:
                beh_features[feature.lower()] = feature_rows['Value'].values[0]
        
        return beh_features
    except Exception as e:
        print(f"Error getting behavioral data: {e}")
        return None

def process_file(fif_file, participant_data, behavior_path, output_dir):
    """
    Process a single EEG file and create samples that are compatible with LaBraM.
    
    Args:
        fif_file: Path to the EEG file
        participant_data: DataFrame with participant metadata
        behavior_path: Path to behavioral data file
        output_dir: Directory to save processed data
    
    Returns:
        int: Number of epochs processed
    """
    try:
        # Extract participant ID from filename
        participant_id = os.path.basename(fif_file).split('_')[0]
        
        if len(participant_id) < 4 or not participant_id[-1].isalpha():
            print(f"Warning: Invalid participant ID format for {participant_id}")
            return 0
            
        triad_id = int(participant_id[:3])
        participant_position = participant_id[-1]  # A, B, or C
        
        # Map participant position to P1, P2, P3
        position_map = {'A': 'P1', 'B': 'P2', 'C': 'P3'}
        if participant_position not in position_map:
            print(f"Warning: Unknown participant position {participant_position} for {participant_id}")
            return 0
            
        participant_num = position_map[participant_position]
        
        # Get metadata from participant_data if available
        participant_rows = participant_data[participant_data['Exp_id'] == participant_id]
        if participant_rows.empty:
            print(f"Warning: No metadata found for {participant_id}")
            return 0
            
        subject_info = participant_rows.iloc[0].to_dict()
        
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
            
            # Get timing information from epochs
            sfreq = epochs.info['sfreq']  # Sampling frequency (should be 500 Hz)
            times = epochs.times  # Time points array
            
            # Get the trial data for this epoch
            eeg_data = epochs[epoch_idx].get_data()[0]  # Shape: [channels, time_points]
            
            # Find indices corresponding to t=0s and t=4s
            # Find indices corresponding to t=0s and t=4s
            t0_idx = np.where(times >= 0)[0][0]  # First index where time >= 0
            t4_idx = np.where(times <= 4)[0][-1]  # Last index where time <= 4

            # Calculate exactly how many points we should have after resampling
            orig_fs = sfreq  # Original frequency (should be 500 Hz)
            target_fs = 200  # Target frequency for LaBraM
            time_duration = 4.0  # Exactly 4 seconds

            # Extract only the time window from t=0s to t=4s
            eeg_data = eeg_data[:, t0_idx:t4_idx+1]  # +1 to include the t4_idx

            # For more accurate control, let's compute exact indices after resampling
            desired_samples = int(time_duration * target_fs)  # Should be 800
            current_samples = eeg_data.shape[1]

            # Resample more precisely to get exactly 800 samples
            resampled_data = np.zeros((eeg_data.shape[0], desired_samples))
            for ch_idx in range(eeg_data.shape[0]):
                # Use linear interpolation to get exactly 800 samples
                x_orig = np.linspace(0, time_duration, current_samples)
                x_new = np.linspace(0, time_duration, desired_samples)
                resampled_data[ch_idx] = np.interp(x_new, x_orig, eeg_data[ch_idx])

            # Replace our data with the precisely resampled data
            eeg_data = resampled_data

            # Normalize data to μV scale for LaBraM
            eeg_data = eeg_data * 10000

            # Now we should have exactly 800 time points
            time_points = eeg_data.shape[1]
            
            # Calculate how many complete patches we can fit
            num_patches = time_points // 200
            
            # Make sure we have at least one patch
            if num_patches == 0:
                print(f"Warning: Epoch {epoch_idx} too short after resampling - skipping")
                continue
            
            # Truncate to the nearest multiple of 200
            new_length = num_patches * 200
            eeg_data = eeg_data[:, :new_length]
            
            # Reshape the data to match LaBraM's expectations
            # Shape should be [channels, num_patches, patch_size]
            eeg_data_reshaped = eeg_data.reshape(eeg_data.shape[0], num_patches, 200)
            
            # Parse condition to determine trial type and participants involved
            condition_type, has_feedback_check = parse_condition(condition)
            
            # Double-check has_feedback from parse_condition matches our direct check
            if has_feedback != has_feedback_check:
                print(f"Warning: Feedback status mismatch for {condition}")
            
            # Determine if this participant is involved in this condition
            participant_involved = is_participant_involved(condition_type, participant_num)
            
            # Get behavioral data if available
            beh_features = get_behavioral_data(behavior_path, triad_id, participant_num, condition)
            
            # Create sample dictionary - using has_feedback as the primary classification target (y)
            sample = {
                'X': eeg_data_reshaped,  # Now shaped as [channels, num_patches, patch_size]
                'y': int(has_feedback),  # Binary classification: 1 for feedback, 0 for no feedback
                'participant_id': participant_id,
                'triad_id': triad_id,
                'epoch_idx': epoch_idx,
                'condition': condition,
                'condition_type': condition_type,
                'has_feedback': has_feedback,
                'participant_position': participant_position,
                'participant_num': participant_num,
                'participant_involved': participant_involved,
                'subject_id': subject_info.get('Subject_id'),
                'friend_status': subject_info.get('Friend_status'),
                'age': subject_info.get('Age'),
                'gender': subject_info.get('Gender'),
                'class_friends': subject_info.get('Class_friends'),
                'class_close_friends': subject_info.get('Class_close_friends')
            }
            
            # Add behavioral features if available
            if beh_features:
                for key, value in beh_features.items():
                    sample[f'beh_{key}'] = value
            
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

def is_participant_involved(condition_type, participant_num):
    """
    Determine if a participant is involved in a specific condition.
    
    Args:
        condition_type: Type of condition (solo_p1, duo_p1p2, etc.)
        participant_num: Participant number (P1, P2, P3)
    
    Returns:
        bool: True if participant is involved, False otherwise
    """
    if condition_type == 'solo_p1' and participant_num == 'P1':
        return True
    elif condition_type == 'solo_p3' and participant_num == 'P3':
        return True
    elif condition_type == 'duo_p1p2' and participant_num in ['P1', 'P2']:
        return True
    elif condition_type == 'duo_p1p3' and participant_num in ['P1', 'P3']:
        return True
    elif condition_type == 'duo_p2p3' and participant_num in ['P2', 'P3']:
        return True
    return False

def parse_condition(condition):
    """
    Parse condition code to determine trial type and feedback status.
    
    Args:
        condition: Condition string from the EEG file
    
    Returns:
        tuple: (condition_type, has_feedback)
    """
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
            
            # # Delete source file after successful copy to save disk space
            # os.remove(src_path)
    
    # Print summary
    print(f"Split data into:")
    print(f"  Train: {train_count} samples from {len(train_participants)} participants")
    print(f"  Validation: {val_count} samples from {len(val_participants)} participants")
    print(f"  Test: {test_count} samples from {len(test_participants)} participants")
    
    # Generate class distribution statistics
    train_statistics = get_class_distribution(train_dir)
    val_statistics = get_class_distribution(val_dir)
    test_statistics = get_class_distribution(test_dir)
    
    print("\nClass distribution (feedback vs. no feedback):")
    print(f"  Train: {train_statistics}")
    print(f"  Validation: {val_statistics}")
    print(f"  Test: {test_statistics}")

def get_class_distribution(directory):
    """
    Calculate class distribution in a directory.
    
    Args:
        directory: Directory containing data files
    
    Returns:
        dict: Class distribution
    """
    class_counts = {0: 0, 1: 0}
    
    for file in os.listdir(directory):
        if file.endswith('.pkl'):
            try:
                with open(os.path.join(directory, file), 'rb') as f:
                    data = pickle.load(f)
                    y_value = data.get('y', None)
                    if y_value is not None and y_value in class_counts:
                        class_counts[y_value] += 1
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    total = sum(class_counts.values())
    if total > 0:
        percentages = {k: f"{v} ({v/total:.1%})" for k, v in class_counts.items()}
        return percentages
    return class_counts

if __name__ == "__main__":
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to project root
    root_dir = os.path.dirname(script_dir)  # Assuming script is one level below project root
    
    # Path to data directory
    data_dir = os.path.join(root_dir, "DTUDATA", "FG_Data")
    
    # Specific paths
    eeg_folder_path = "/work3/s224183/PreprocessedEEGData/"
    overview_path = os.path.join(data_dir, "FG_overview_df_v2.pkl")
    behavior_path = os.path.join(data_dir, "Beh_feat_df_v2.pkl")
    
    # Output paths
    processed_dir = "/work3/s224183/processed"
    output_dir = "/work3/s224183/LaBraM_data"
    
    # Create output directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load overview data
    try:
        with open(overview_path, "rb") as f:
            overview_df = pickle.load(f)
        print(f"Loaded overview data with {len(overview_df)} rows")
    except Exception as e:
        print(f"Error loading overview data: {e}")
        overview_df = pd.DataFrame()
    
    # Get list of all fif files
    fif_files = [os.path.join(eeg_folder_path, f) for f in os.listdir(eeg_folder_path) 
                if f.endswith('.fif') and 'preprocessed' in f]
    # Process files one by one to reduce memory usage
    total_epochs = 0
    for i, fif_file in enumerate(fif_files):
        print(f"Processing file {i+1}/{len(fif_files)}: {os.path.basename(fif_file)}")
        epochs_processed = process_file(fif_file, overview_df, behavior_path, processed_dir)
        total_epochs += epochs_processed
        
        # Force garbage collection after each file
        gc.collect()
    
    print(f"Total epochs processed: {total_epochs}")
    
    # Split data into train/val/test by participant
    if total_epochs > 0:
        split_data_by_participant(processed_dir, output_dir)
    
    print("DTU data processing complete!")
