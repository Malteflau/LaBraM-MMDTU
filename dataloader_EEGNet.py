import os
import pickle
import numpy as np

def load_eeg_data_from_pkl(directory, label_mode='feedback'):
    """
    Generalized EEG data loader with label modes:
    - 'feedback': has_feedback (1) vs no_feedback (0)
    - 'gender': female (0), male (1), using only feedback or only no-feedback trials
    - 'solo_vs_group': T1P (0) vs T3P (1), using only feedback trials
    - 'friend_status': non-friend (0), friend (1), using only feedback trials

    Returns:
    - X: np.ndarray, shape [samples, channels, 800, 1]
    - y: np.ndarray, labels according to the selected mode
    """
    X, y = [], []

    for file in os.listdir(directory):
        if not file.endswith('.pkl'):
            continue

        with open(os.path.join(directory, file), 'rb') as f:
            sample = pickle.load(f)

        condition = sample.get('condition', '')
        has_feedback = sample.get('has_feedback', False)

        # Filtering logic based on label_mode
        if label_mode == 'feedback':
            print('Data labeled by feedback')
            label = int(has_feedback)

        elif label_mode == 'gender':
            print('Data labeled by gender')
            if 'gender' not in sample or sample['gender'] not in ['M', 'F']:
                continue
            label = 1 if sample['gender'] == 'M' else 0

        elif label_mode == 'solo_vs_group':
            print('Data labeled by solo vs group')
            if condition != 'T1P' and condition != 'T3P':
                continue
            if not has_feedback:
                continue
            label = 0 if condition == 'T1P' else 1

        elif label_mode == 'friend_status':
            print('Data labeled by friend status')
            if 'friend_status' not in sample:
                continue
            if not has_feedback:
                continue
            label = 1 if sample['friend_status'] == 'Yes' else 0

        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        data = sample['X']
        if data.shape[1:] != (4, 200):
            continue

        reshaped = data.reshape(data.shape[0], -1)  # [channels, 800]
        reshaped = reshaped[:, :, np.newaxis]       # [channels, 800, 1]
        X.append(reshaped)
        y.append(label)

    X = np.stack(X)
    y = np.array(y)
    print(X.shape, y.shape)
    return X, y