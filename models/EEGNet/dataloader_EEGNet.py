import os
import pickle
import numpy as np

def load_eeg_data_from_pkl(directory, target_conditions=('T1P', 'T1Pn')):
    """
    Simplified loader: Loads 4-second EEG trials from individual .pkl files (as written by make_DTU.py),
    filters for T1P and T1Pn conditions, flattens patches to [channels, 800], and reshapes for TensorFlow.

    Returns:
    - X: np.ndarray, shape [samples, channels, 800, 1]
    - y: np.ndarray, binary labels (0 = T1Pn, 1 = T1P)
    """
    X, y = [], []

    for file in os.listdir(directory):
        if not file.endswith('.pkl'):
            continue

        with open(os.path.join(directory, file), 'rb') as f:
            sample = pickle.load(f)

        condition = sample.get('condition', '')
        if not any(condition.startswith(c) for c in target_conditions):
            continue

        data = sample['X']  # shape: [channels, 4, 200]
        if data.shape[1:] != (4, 200):
            print('Skipped malformed sample')
            continue  # skip malformed samples

        # Reshape to [channels, 800]
        reshaped = data.reshape(data.shape[0], -1)  # [channels, 800]
        reshaped = reshaped[:, :, np.newaxis]       # [channels, 800, 1]

        X.append(reshaped)
        y.append(1 if condition.startswith('T1P') and 'Pn' not in condition else 0)

    X = np.stack(X)
    y = np.array(y)
    return X, y