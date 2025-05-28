import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

class DTUConcatenatedAdapter:
    """
    Adapter class to load DTU data, concatenate epochs with identical conditions,
    and convert to the format expected by PBT.
    """
    def __init__(self, root, condition=["sologroup"], filter_feedback_only=None,
                 filter_non_feedback_only=None, filter_non_participant=True,
                 concat_same_conditions=True, max_epochs_to_concat=30):
        """
        Initialize the adapter to load and prepare DTU data for PBT.
        
        Args:
            root: Root directory containing train and test folders
            condition: Label type to extract (feedback, friendship, sologroup, gender)
            filter_feedback_only: If True, only include samples with feedback
            filter_non_feedback_only: If True, only include samples without feedback
            filter_non_participant: Filter out non-participant conditions
            concat_same_conditions: Whether to concatenate epochs from the same condition
            max_epochs_to_concat: Maximum number of epochs to concatenate into a single sample
        """
        self.root = root
        self.condition = condition
        self.filter_feedback_only = filter_feedback_only
        self.filter_non_feedback_only = filter_non_feedback_only
        self.filter_non_participant = filter_non_participant
        self.concat_same_conditions = concat_same_conditions
        self.max_epochs_to_concat = max_epochs_to_concat
        
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
            # Additional filtering by subject would go here
            pass
        
        # Get channel names from first valid file
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
        
        if self.concat_same_conditions:
            # Process and concatenate train and test data separately
            train_data, train_labels, train_meta = self._process_and_concatenate_files(
                os.path.join(self.root, "train"), train_files, train_valid_indices, "session_train"
            )
            
            test_data, test_labels, test_meta = self._process_and_concatenate_files(
                os.path.join(self.root, "test"), test_files, test_valid_indices, "session_test"
            )
            
            # Combine train and test data
            all_data = np.concatenate([train_data, test_data])
            all_labels = np.concatenate([train_labels, test_labels])
            all_meta = pd.concat([train_meta, test_meta], ignore_index=True)
            
        else:
            # Original processing without concatenation
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
            
            # Combine all data
            all_data = np.stack(all_data)
            all_labels = np.array(all_labels)
            all_meta = pd.DataFrame(all_meta_info)
        
        return all_data, all_labels, all_meta, channels
    

    def _process_and_concatenate_files(self, directory, files, valid_indices, session):
        """
        Process files and concatenate epochs with identical conditions.
        
        Args:
            directory: Directory containing files
            files: List of filenames
            valid_indices: Indices of valid files
            session: Session name
            
        Returns:
            concatenated_data, concatenated_labels, concatenated_meta
        """
        # Collect data grouped by participant and condition
        data_groups = defaultdict(list)
        label_groups = defaultdict(list)
        meta_groups = defaultdict(list)
        
        # Process each file
        for idx in valid_indices:
            file = files[idx]
            data, label, meta_dict = self._load_and_process_file(directory, file, session)
            
            # Create a key combining participant and condition
            participant = meta_dict["participant_num"]
            condition_str = meta_dict["condition"]
            if self.condition[0] == "sologroup":
                is_solo = self._is_solo_condition(condition_str, participant)
                group_key = f"{participant}_{int(is_solo)}"
            else:
                # For other condition types, group by participant and condition value
                group_key = f"{participant}_{label}"
            
            # Add data, label, and metadata to the appropriate group
            data_groups[group_key].append(data)
            label_groups[group_key].append(label)
            meta_dict["session"] = session
            meta_groups[group_key].append(meta_dict)
        
        # Concatenate data within each group
        concatenated_data = []
        concatenated_labels = []
        concatenated_meta = []
        
        # Find the maximum time dimension across all data
        max_time_dim = 0
        for group_key in data_groups:
            for data_array in data_groups[group_key]:
                _, time_dim = data_array.shape
                max_time_dim = max(max_time_dim, time_dim)
        
        print(f"Maximum time dimension found: {max_time_dim}")
        
        for group_key in data_groups:
            group_data = data_groups[group_key]
            group_labels = label_groups[group_key]
            group_meta = meta_groups[group_key]
            
            # Ensure all samples in the group have the same label
            assert len(set(group_labels)) == 1, f"Group {group_key} has mixed labels: {group_labels}"
            label = group_labels[0]
            
            # Split into chunks of max_epochs_to_concat
            for i in range(0, len(group_data), self.max_epochs_to_concat):
                chunk_data = group_data[i:i + self.max_epochs_to_concat]
                chunk_meta = group_meta[i:i + self.max_epochs_to_concat]
                
                if len(chunk_data) >= 3:  # Only use chunks with at least 3 epochs
                    # Instead of concatenating directly, we'll pad each sample to a fixed length
                    # and then use a subset of the concatenated data
                    
                    # Get number of channels from first sample
                    num_channels = chunk_data[0].shape[0]
                    
                    # Calculate total time dimension for this chunk
                    chunk_time_dim = sum(data.shape[1] for data in chunk_data)
                    
                    # Create a new array for the concatenated data
                    concat_data = np.zeros((num_channels, chunk_time_dim), dtype=np.float32)
                    
                    # Fill the array with data from each sample
                    time_idx = 0
                    for data in chunk_data:
                        time_len = data.shape[1]
                        concat_data[:, time_idx:time_idx + time_len] = data
                        time_idx += time_len
                    
                    concatenated_data.append(concat_data)
                    concatenated_labels.append(label)
                    
                    # Combine metadata
                    meta_entry = chunk_meta[0].copy()
                    meta_entry["num_concatenated"] = len(chunk_data)
                    meta_entry["original_files"] = [m["file"] for m in chunk_meta]
                    meta_entry["concat_time_dim"] = chunk_time_dim
                    concatenated_meta.append(meta_entry)
        
        print(f"Created {len(concatenated_data)} concatenated samples")
        
        if len(concatenated_data) == 0:
            raise ValueError("No valid concatenated samples were created!")
        
        # Check if all concatenated data have the same shape
        shapes = [data.shape for data in concatenated_data]
        print(f"Concatenated data shapes: {shapes[:5]} {'...' if len(shapes) > 5 else ''}")
        
        if len(set(tuple(shape) for shape in shapes)) > 1:
            print("Warning: Concatenated data have different shapes. Padding to uniform shape...")
            
            # Find the maximum dimensions
            max_channels = max(shape[0] for shape in shapes)
            max_time = max(shape[1] for shape in shapes)
            
            # Pad all arrays to the maximum dimensions
            padded_data = []
            for data in concatenated_data:
                channels, time = data.shape
                
                # Create a new array with maximum dimensions
                padded = np.zeros((max_channels, max_time), dtype=np.float32)
                
                # Copy the original data into the padded array
                padded[:channels, :time] = data
                
                padded_data.append(padded)
            
            concatenated_data = np.array(padded_data)
        else:
            # All arrays have the same shape, can directly convert to numpy array
            concatenated_data = np.array(concatenated_data)
        
        concatenated_labels = np.array(concatenated_labels)
        concatenated_meta = pd.DataFrame(concatenated_meta)
        
        return concatenated_data, concatenated_labels, concatenated_meta
        
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
                        
                        # Check specific conditions where participant is not involved
                        if condition_str.startswith("T23") and participant_num == "P1":
                            keep_sample = False
                        elif condition_str.startswith("T13") and participant_num == "P2":
                            keep_sample = False
                        elif condition_str.startswith("T12") and participant_num == "P3":
                            keep_sample = False
                    
                    if keep_sample:
                        valid_indices.append(i)
                        
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        
        if not valid_indices:
            print(f"Warning: No valid samples found after filtering in {directory}")
        
        return valid_indices
    
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
            label = 1 if self._is_solo_condition(condition_str, participant_num) else 0
        elif self.condition[0] == "gender":
            label = 1 if sample["gender"] == "M" else 0
        else:
            label = sample["y"]
        
        # Collect metadata
        meta_dict = {
            "subject": sample.get("participant_num", "unknown"),
            "file": file,
            "condition": sample.get("condition", ""),
            "participant_num": sample.get("participant_num", ""),
            "original_condition_label": sample.get("y", None)
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
        if condition_str.startswith("T1") and not condition_str.startswith("T12") and not condition_str.startswith("T13"):
            return True  # Direct solo condition
        elif condition_str.startswith("T23") and participant_num == "P1":
            return True  # Participant 1 not involved in T23
        elif condition_str.startswith("T13") and participant_num == "P2":
            return True  # Participant 2 not involved in T13
        elif condition_str.startswith("T12") and participant_num == "P3":
            return True  # Participant 3 not involved in T12
        else:
            return False  # Group condition for this participant


def get_DTU_concatenated_data(subject=None, freq_min=8, freq_max=45, resample=200, channels=None, 
                             condition=["sologroup"], root=None, filter_feedback_only=None, 
                             filter_non_feedback_only=None, filter_non_participant=True,
                             concat_same_conditions=True, max_epochs_to_concat=30):
    """
    Load DTU data in a format compatible with PBT, concatenating epochs with identical conditions.
    
    Args:
        subject: List of subject IDs to include (default: None for all)
        freq_min: Minimum frequency for bandpass filter (default: 8)
        freq_max: Maximum frequency for bandpass filter (default: 45)  
        resample: Sampling rate to resample data to (default: 200)
        channels: List of channels to include (default: None for all)
        condition: Label type to extract (default: ["sologroup"])
        root: Root directory containing train and test folders
        filter_feedback_only: If True, only include samples with feedback
        filter_non_feedback_only: If True, only include samples without feedback
        filter_non_participant: Filter out non-participant conditions
        concat_same_conditions: Whether to concatenate epochs from the same condition
        max_epochs_to_concat: Maximum number of epochs to concatenate into a single sample
        
    Returns:
        data, labels, meta, channels
    """
    if root is None:
        raise ValueError("Root directory must be specified")
    
    # Create adapter instance
    adapter = DTUConcatenatedAdapter(
        root=root,
        condition=condition,
        filter_feedback_only=filter_feedback_only,
        filter_non_feedback_only=filter_non_feedback_only,
        filter_non_participant=filter_non_participant,
        concat_same_conditions=concat_same_conditions,
        max_epochs_to_concat=max_epochs_to_concat
    )
    
    # Get data
    data, labels, meta, channel_names = adapter.get_data(subjects=subject)
    
    # Apply frequency filtering if needed (would need to implement)
    # This could be implemented similar to the bandpass filter in DTULoader
    
    # Filter channels if specified
    if channels is not None:
        # Implementation for channel filtering would go here
        pass
    
    return data, labels, meta, channel_names