import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class DTUAdapter:
    """
    Adapter class to load DTU data and convert it to the format expected by PBT.
    This bridges the gap between the DTULoader and the PBT model requirements.
    """
    def __init__(self, root, condition=["feedback"], filter_feedback_only=None, 
                 filter_non_feedback_only=None, filter_non_participant=True):
        """
        Initialize the adapter to load and prepare DTU data for PBT.
        
        Args:
            root: Root directory containing train and test folders
            condition: Label type to extract (feedback, friendship, sologroup, gender)
            filter_feedback_only: If True, only include samples with feedback
            filter_non_feedback_only: If True, only include samples without feedback
            filter_non_participant: Filter out non-participant conditions
        """
        self.root = root
        self.condition = condition
        # FIX: Don't override the parameter values with hardcoded False
        self.filter_feedback_only = filter_feedback_only
        self.filter_non_feedback_only = filter_non_feedback_only
        self.filter_non_participant = filter_non_participant
        
    def get_data(self, subjects=None):
        """
        Load data and convert to the format expected by PBT:
        - data: numpy array of shape (num_trials, num_channels, num_timepoints)
        - labels: numpy array of shape (num_trials,) containing class labels
        - meta: pandas DataFrame with subject and session information
        - channels: list of channel names
        
        Args:
            subjects: List of subject IDs to include (None for all)
            
        Returns:
            data, labels, meta, channels
        """
        # Get all files from train and test directories
        train_files = [f for f in os.listdir(os.path.join(self.root, "train")) if f.endswith('.pkl')]
        test_files = [f for f in os.listdir(os.path.join(self.root, "test")) if f.endswith('.pkl')]
        
        # Extract valid files based on filtering criteria
        train_valid_indices = self._get_valid_indices(os.path.join(self.root, "train"), train_files)
        test_valid_indices = self._get_valid_indices(os.path.join(self.root, "test"), test_files)
        
        # Filter by subject if specified
        if subjects is not None:
            train_valid_indices = self._filter_by_subjects(os.path.join(self.root, "train"), train_files, train_valid_indices, subjects)
            test_valid_indices = self._filter_by_subjects(os.path.join(self.root, "test"), test_files, test_valid_indices, subjects)
        
        # Load and combine all data
        all_data = []
        all_labels = []
        all_meta_info = []
        
        # Process training files
        for idx in train_valid_indices:
            file = train_files[idx]
            data, label, meta_dict = self._load_and_process_file(
                os.path.join(self.root, "train"), file, "train"
            )
            all_data.append(data)
            all_labels.append(label)
            meta_dict["session"] = "session_train"
            all_meta_info.append(meta_dict)
            
        # Process test files
        for idx in test_valid_indices:
            file = test_files[idx]
            data, label, meta_dict = self._load_and_process_file(
                os.path.join(self.root, "test"), file, "test"
            )
            all_data.append(data)
            all_labels.append(label)
            meta_dict["session"] = "session_test"
            all_meta_info.append(meta_dict)
        
        if not all_data:
            raise ValueError("No valid data found after filtering")
        
        # Combine all data
        combined_data = np.stack(all_data)
        combined_labels = np.array(all_labels)
        
        # Create metadata DataFrame
        meta_df = pd.DataFrame(all_meta_info)
        
        # Get list of channel names (based on first sample as they should be consistent)
        first_file_path = os.path.join(self.root, "train", train_files[train_valid_indices[0]])
        with open(first_file_path, 'rb') as f:
            sample = pickle.load(f)
            # Replace with actual channel names if available, otherwise generate them
            num_channels = sample["X"].shape[0]
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
            channels = [name.upper() for name in channel_mapping.keys()]
        
        return combined_data, combined_labels, meta_df, channels
    
    def _filter_by_subjects(self, directory, files, valid_indices, subjects):
        """
        Filter valid indices to only include specified subjects.
        
        Args:
            directory: Directory containing files
            files: List of filenames
            valid_indices: List of currently valid indices
            subjects: List of subject IDs to include
            
        Returns:
            List of filtered valid indices
        """
        filtered_indices = []
        
        for idx in valid_indices:
            file = files[idx]
            try:
                with open(os.path.join(directory, file), 'rb') as f:
                    sample = pickle.load(f)
                    participant_num = sample.get("participant_num", "")
                    
                    if participant_num in subjects:
                        filtered_indices.append(idx)
                        
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        
        return filtered_indices
    
    def _get_valid_indices(self, directory, files):
        """
        Get indices of valid files based on filtering criteria.
        
        Args:
            directory: Directory containing files
            files: List of filenames
            
        Returns:
            List of valid file indices
        """
        valid_indices = []
        
        for i, file in enumerate(files):
            try:
                with open(os.path.join(directory, file), 'rb') as f:
                    sample = pickle.load(f)
                    
                    # Check if the sample meets our filtering criteria
                    keep_sample = True
                    
                    # Apply feedback filters if requested
                    if self.filter_feedback_only is not None:
                        has_feedback = sample.get("has_feedback", True)
                        if self.filter_feedback_only and not has_feedback:
                            # We only want samples WITH feedback but this one doesn't have it
                            keep_sample = False
                        elif self.filter_non_feedback_only and has_feedback:
                            # We only want samples WITHOUT feedback but this one has it
                            keep_sample = False
                    
                    # Apply non-participant filter if requested
                    if self.filter_non_participant and keep_sample:
                        condition_str = sample.get("condition", "")
                        participant_num = sample.get("participant_num", "")
                        
                        # FIX: More robust non-participant detection
                        if self._is_non_participant(condition_str, participant_num):
                            keep_sample = False
                            #print(f"Filtering out non-participant: {condition_str}, {participant_num}")
                    
                    if keep_sample:
                        valid_indices.append(i)
                        
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        
        if not valid_indices:
            print(f"Warning: No valid samples found after filtering in {directory}")
        
        return valid_indices
    
    def _is_non_participant(self, condition_str, participant_num):
        """
        Helper method to determine if a participant is NOT involved in a given condition.
        
        Args:
            condition_str: Condition string (e.g., "T23", "T13", "T12")  
            participant_num: Participant number (e.g., "P1", "P2", "P3")
            
        Returns:
            True if participant is NOT involved (should be filtered out), False otherwise
        """
        # Define the conditions where specific participants are NOT involved
        non_participant_conditions = {
            "T23": ["P1"],  # In T23 (P2-P3 interaction), P1 is not involved
            "T13": ["P2"],  # In T13 (P1-P3 interaction), P2 is not involved  
            "T12": ["P3"]   # In T12 (P1-P2 interaction), P3 is not involved
        }
        
        # Check if this condition-participant combination should be filtered
        for condition_prefix, excluded_participants in non_participant_conditions.items():
            if condition_str.startswith(condition_prefix) and participant_num in excluded_participants:
                return True
        
        return False
    
    def _load_and_process_file(self, directory, file, file_type):
        """
        Load and process a single file.
        
        Args:
            directory: Directory containing the file
            file: Filename
            file_type: 'train' or 'test'
            
        Returns:
            processed_data: numpy array of shape (num_channels, num_timepoints)
            label: class label
            meta_dict: dictionary with metadata
        """
        with open(os.path.join(directory, file), 'rb') as f:
            sample = pickle.load(f)
        
        # Extract data
        X = sample["X"]  # shape: (channels, patches, time_per_patch)
        channels, patches, time_per_patch = X.shape
        
        # Reshape to (channels, timepoints) by flattening patches and time_per_patch
        processed_data = X.reshape(channels, patches * time_per_patch)
        
        # Determine label based on condition
        if self.condition[0] == "feedback":
            label = sample["y"]
        elif self.condition[0] == "friendship":
            label = 1 if sample["friend_status"] == "Yes" else 0
        elif self.condition[0] == "sologroup":
            condition_str = sample.get("condition", "")
            participant_num = sample.get("participant_num", "")
            label = self._is_solo_condition(condition_str, participant_num)
        elif self.condition[0] == "gender":
            label = 1 if sample["gender"] == "M" else 0
        else:
            label = sample["y"]
        
        # Collect metadata
        meta_dict = {
            "subject": sample.get("participant_num", "unknown"),
            "file": file,
            "condition": sample.get("condition", ""),
            "participant_num": sample.get("participant_num", "")
        }
        
        return processed_data, label, meta_dict
    
    def _is_solo_condition(self, condition_str, participant_num):
        """
        Helper method to determine if a trial is solo for this participant.
        
        Args:
            condition_str: Condition string
            participant_num: Participant number
            
        Returns:
            True if solo condition, False otherwise
        """
        # FIX: Clearer logic for solo conditions
        # T1 conditions are always solo (but exclude T12, T13 which are group conditions)
        if condition_str.startswith("T1") and not condition_str.startswith("T12") and not condition_str.startswith("T13"):
            return True
        
        # For group conditions, check if this participant is actually involved
        # If they're not involved, it's effectively a "solo" condition for them
        if self._is_non_participant(condition_str, participant_num):
            return True
        
        # Otherwise, it's a group condition where this participant is involved
        return False


class DTUDataset(Dataset):
    """
    PyTorch Dataset for DTU EEG data.
    """
    def __init__(self, root, files, valid_indices=None, condition=["feedback"], type="train"):
        """
        Initialize dataset.
        
        Args:
            root: Root directory containing data files
            files: List of data files
            valid_indices: List of valid indices (optional, will use all if None)
            condition: Label type to extract
            type: Dataset type ('train' or 'test')
        """
        self.root = root
        self.files = files
        self.condition = condition
        self.type = type
        
        # If valid_indices not provided, use all file indices
        if valid_indices is None:
            valid_indices = list(range(len(files)))
        
        # FIX: Remove redundant filtering since it should be done by DTUAdapter
        # Just use the provided valid_indices directly
        self.valid_indices = valid_indices
        
        if len(self.valid_indices) == 0:
            raise ValueError("No valid indices found after filtering")
    
    def __len__(self):
        """
        Return the number of valid samples in the dataset.
        """
        return len(self.valid_indices)
    
    def _is_solo_condition(self, condition_str, participant_num):
        """
        Helper method to determine if a trial is solo for this participant.
        """
        # T1 conditions are always solo (but exclude T12, T13 which are group conditions)
        if condition_str.startswith("T1") and not condition_str.startswith("T12") and not condition_str.startswith("T13"):
            return True
        
        # Define the conditions where specific participants are NOT involved
        non_participant_conditions = {
            "T23": ["P1"],  # In T23 (P2-P3 interaction), P1 is not involved
            "T13": ["P2"],  # In T13 (P1-P3 interaction), P2 is not involved  
            "T12": ["P3"]   # In T12 (P1-P2 interaction), P3 is not involved
        }
        
        # If participant is not involved in the group condition, it's solo for them
        for condition_prefix, excluded_participants in non_participant_conditions.items():
            if condition_str.startswith(condition_prefix) and participant_num in excluded_participants:
                return True
        
        return False
        
    def __getitem__(self, index):
        """
        Get item at index. Since valid_indices already contains only valid trials,
        we don't need to do additional filtering here.
        """
        # Get the actual file index from our pre-filtered valid indices
        file_index = self.valid_indices[index]
        file = self.files[file_index]
        
        # Load the sample
        with open(os.path.join(self.root, file), 'rb') as f:
            sample = pickle.load(f)
        
        X = sample["X"]
        condition_type = sample.get("condition", "")
        participant_num = sample.get("participant_num", "")
        
        # Determine label based on condition
        if self.condition[0] == "feedback":
            y = sample["y"]
        elif self.condition[0] == "friendship":
            y = 1 if sample["friend_status"] == "Yes" else 0
        elif self.condition[0] == "sologroup":
            y = 1 if self._is_solo_condition(condition_type, participant_num) else 0
        elif self.condition[0] == "gender":
            y = 1 if sample["gender"] == "M" else 0
        else:
            # Default fallback to feedback label
            y = sample["y"]

        channels, patches, time_per_patch = X.shape
        
        # Prepare metadata
        gender = 1 if sample.get("gender", "M") == "M" else 0
        feedback = sample.get("y", 0)
        friendship = 1 if sample.get("friend_status", "No") == "Yes" else 0
        solo_group = 1 if self._is_solo_condition(condition_type, participant_num) else 0
        
        metadata = {
            "gender": torch.LongTensor([gender]),
            "feedback": torch.LongTensor([feedback]),
            "friendship": torch.LongTensor([solo_group])  # FIX: This should be friendship, not solo_group
        }

        # Prepare tensors
        X_tensor = torch.FloatTensor(X.reshape(channels, patches * time_per_patch))
        y_tensor = torch.FloatTensor([y]).squeeze()

        return X_tensor, y_tensor, metadata

import numpy as np
from scipy import signal

def get_DTU_data(subject=None, freq_min=8, freq_max=45, resample=200, channels=None, 
                 condition=["feedback"], root=None, filter_feedback_only=None, 
                 filter_non_feedback_only=None, filter_non_participant=True,
                 compute_power_spectrum=False):
    """
    Load DTU data in a format compatible with PBT.
    This function follows the pattern of the other get_X functions in the PBT codebase.
    
    Args:
        subject: List of subject IDs to include (default: None for all)
        freq_min: Minimum frequency for bandpass filter (default: 8)
        freq_max: Maximum frequency for bandpass filter (default: 45)  
        resample: Sampling rate to resample data to (default: 200)
        channels: List of channels to include (default: None for all)
        condition: Label type to extract (default: ["feedback"])
        root: Root directory containing train and test folders
        filter_feedback_only: If True, only include samples with feedback
        filter_non_feedback_only: If True, only include samples without feedback
        filter_non_participant: Filter out non-participant conditions
        compute_power_spectrum: If True, compute power spectrum instead of time domain data
        
    Returns:
        data, labels, meta, channels
    """
    if root is None:
        raise ValueError("Root directory must be specified")
    
    # Create adapter instance
    adapter = DTUAdapter(
        root=root,
        condition=condition,
        filter_feedback_only=filter_feedback_only,
        filter_non_feedback_only=filter_non_feedback_only,
        filter_non_participant=filter_non_participant
    )
    
    # Get data
    data, labels, meta, channel_names = adapter.get_data(subjects=subject)
    
    # Filter channels if specified
    if channels is not None:
        # Find indices of specified channels
        channel_indices = [i for i, ch in enumerate(channel_names) if ch in channels]
        if not channel_indices:
            raise ValueError(f"None of the specified channels {channels} were found in the data")
        
        # Filter data to only include specified channels
        data = data[:, channel_indices, :]
        channel_names = [channel_names[i] for i in channel_indices]
    
    # Compute power spectrum if requested
    if compute_power_spectrum:
        data = calculate_power_spectrum(data, fs=resample, freq_min=freq_min, freq_max=freq_max)
    
    return data, labels, meta, channel_names


def calculate_power_spectrum(eeg_data, fs=200, nperseg=256, noverlap=128, freq_min=8, freq_max=45):
    """
    Calculate power spectrum for EEG data.
    
    Args:
        eeg_data: EEG data with shape (trials, channels, time_points)
        fs: Sampling frequency in Hz
        nperseg: Length of each segment for Welch's method
        noverlap: Overlap between segments
        freq_min: Minimum frequency to include
        freq_max: Maximum frequency to include
        
    Returns:
        power_spectra: Power spectra with shape (trials, channels, frequency_bins)
    """
    # Check input data shape
    if eeg_data.ndim != 3:
        raise ValueError(f"Expected 3D data (trials, channels, time_points), got shape {eeg_data.shape}")
    
    num_trials, num_channels, num_samples = eeg_data.shape
    
    # Compute Welch's periodogram for one channel to get frequency array
    f_temp, _ = signal.welch(eeg_data[0, 0, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Find frequency indices within the specified range
    freq_indices = np.where((f_temp >= freq_min) & (f_temp <= freq_max))[0]
    if len(freq_indices) == 0:
        raise ValueError(f"No frequencies found within range {freq_min}-{freq_max} Hz")
    
    # Create output array for power spectra (only including specified frequency range)
    power_spectra = np.zeros((num_trials, num_channels, len(freq_indices)))
    
    # Calculate power spectrum for each trial and channel
    for trial_idx in range(num_trials):
        for ch_idx in range(num_channels):
            f, Pxx = signal.welch(eeg_data[trial_idx, ch_idx, :], 
                                 fs=fs, 
                                 nperseg=nperseg, 
                                 noverlap=noverlap)
            
            # Extract power values for the specified frequency range
            power_spectra[trial_idx, ch_idx, :] = Pxx[freq_indices]
    
    return power_spectra


# Alternative implementation using multidimensional FFT for faster computation
def calculate_power_spectrum_fft(eeg_data, fs=200, freq_min=13, freq_max=40):
    """
    Calculate power spectrum using FFT for EEG data.
    This implementation is faster than Welch's method but doesn't perform averaging.
    
    Args:
        eeg_data: EEG data with shape (trials, channels, time_points)
        fs: Sampling frequency in Hz
        freq_min: Minimum frequency to include
        freq_max: Maximum frequency to include
        
    Returns:
        power_spectra: Power spectra with shape (trials, channels, frequency_bins)
    """
    # Compute FFT
    fft_vals = np.fft.rfft(eeg_data, axis=2)
    
    # Calculate power (squared magnitude)
    power = np.abs(fft_vals)**2
    
    # Get frequency values
    freq_resolution = fs / eeg_data.shape[2]
    freqs = np.fft.rfftfreq(eeg_data.shape[2], d=1/fs)
    
    # Find frequency indices within the specified range
    freq_indices = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    if len(freq_indices) == 0:
        raise ValueError(f"No frequencies found within range {freq_min}-{freq_max} Hz")
    
    # Extract power values for the specified frequency range
    power_spectra = power[:, :, freq_indices]
    
    return power_spectra


def train_test_split_dtu(data, labels, meta, test_size=0.2, random_state=None):
    """
    Split data into train and test sets.
    This is an alternative to using the predefined train/test split in the DTU dataset.
    
    Args:
        data: Numpy array of shape (num_trials, num_channels, num_timepoints)
        labels: Numpy array of shape (num_trials,)
        meta: Pandas DataFrame with metadata
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_data, train_labels, train_meta, test_data, test_labels, test_meta
    """
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    train_meta = meta.iloc[train_indices].reset_index(drop=True)
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    test_meta = meta.iloc[test_indices].reset_index(drop=True)
    
    return train_data, train_labels, train_meta, test_data, test_labels, test_meta