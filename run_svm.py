# --------------------------------------------------------
# SVM implementation for EEG classification with LaBraM data loaders
# --------------------------------------------------------

import argparse
import numpy as np
import time
import os
import json
import torch
from pathlib import Path
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, mean_squared_error, r2_score, cohen_kappa_score
import utils
from einops import rearrange

def get_args():
    parser = argparse.ArgumentParser('SVM for EEG classification with LaBraM data loaders', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--output_dir', default='./svm_results',
                       help='path where to save results')
    parser.add_argument('--device', default='cuda',
                       help='device to use for data loading (not used for SVM computation)')
    parser.add_argument('--seed', default=0, type=int)
    
    # SVM specific parameters
    parser.add_argument('--kernel', default='rbf', type=str,
                       help='kernel type for SVM (linear, poly, rbf, sigmoid)')
    parser.add_argument('--C', default=1.0, type=float,
                       help='regularization parameter')
    parser.add_argument('--gamma', default='scale', type=str,
                       help='kernel coefficient (scale, auto, or float value)')
    parser.add_argument('--probability', action='store_true', default=True,
                       help='enable probability estimates')
    parser.add_argument('--max_samples', default=5000, type=int,
                       help='maximum number of samples to use for training (memory constraint)')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='DTU', type=str,
                       help='dataset: TUAB | TUEV | DTU')
    parser.add_argument('--condition', default='feedback', type=str,
                       help='Condition to classify: feedback, friendship, sologroup, gender')
    parser.add_argument('--filter_feedback', type=str, default=None,
                       help='Filter by feedback: feedback (with feedback only), nofeedback (without feedback only)')
    parser.add_argument('--filter_non_participant', action='store_true', default=True,
                       help='Filter out trials where participant is not involved')
    parser.add_argument('--regression', action='store_true', default=False,
                       help='Treat the task as a regression problem instead of classification')
    
    return parser.parse_args()

def get_dataset(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("path/to/TUAB")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("path/to/TUEV")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        
    elif args.dataset == 'DTU':
        # Set default condition if not specified
        condition = [args.condition]
        
        filter_feedback_only = None  # Default to no filtering
        filter_non_feedback_only = None  # Default to no filtering
        
        if args.filter_feedback == 'feedback':
            filter_feedback_only = True
            filter_non_feedback_only = False
        elif args.filter_feedback == 'nofeedback':
            filter_feedback_only = False
            filter_non_feedback_only = True
            
        train_dataset, test_dataset = utils.prepare_DTU_data(
            "/work3/s224183/LaBraM_data", 
            condition=condition,
            filter_feedback_only=filter_feedback_only, 
            filter_non_feedback_only=filter_non_feedback_only,
            filter_non_participant=args.filter_non_participant
        )
        
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
        ch_names = [name.upper() for name in channel_mapping.keys()]
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        
        # If it's a regression task, override metrics
        if args.regression:
            metrics = ["mse", "r2"]

    return train_dataset, test_dataset, ch_names, metrics

def extract_features_from_dataloader(dataloader, max_samples=None):
    """Extract features and labels from a dataloader."""
    all_features = []
    all_labels = []
    count = 0
    
    for batch in dataloader:
        # Get features and labels from batch
        features, labels = batch
        
        # Process features (reshape to flatten or extract relevant features)
        if isinstance(features, torch.Tensor):
            if features.ndim == 4:  # [batch, channels, patches, time_per_patch]
                features = rearrange(features, 'b c p t -> b (c p t)')
            elif features.ndim == 3:  # [batch, channels, time_points]
                features = rearrange(features, 'b c t -> b (c t)')
            
            features = features.cpu().numpy()
        
        # Process labels
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            # Ensure labels are 1D
            labels = labels.ravel()
        
        all_features.append(features)
        all_labels.append(labels)
        
        count += features.shape[0]
        if max_samples is not None and count >= max_samples:
            break
    
    # Concatenate all batches
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    # If max_samples specified, limit the dataset size
    if max_samples is not None and X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    return X, y

def compute_metrics(y_true, y_pred, y_prob=None, is_regression=False):
    """Compute evaluation metrics."""
    results = {}
    
    if is_regression:
        # Regression metrics
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['r2'] = r2_score(y_true, y_pred)
    else:
        # Classification metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # For binary classification
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_prob)
            results['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # For multi-class classification
        if len(np.unique(y_true)) > 2:
            results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    return results

def main():
    args = get_args()
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Fix the seed for reproducibility
    np.random.seed(args.seed)
    
    # Get datasets and metrics
    train_dataset, test_dataset, ch_names, metrics = get_dataset(args)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,  # No need to shuffle for feature extraction
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    print("Extracting features from training set...")
    X_train, y_train = extract_features_from_dataloader(train_loader, args.max_samples)
    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    
    print("Extracting features from test set...")
    X_test, y_test = extract_features_from_dataloader(test_loader)
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")
    
    # Create and train SVM model
    print("Creating SVM model...")
    if args.regression:
        svm_model = SVR(
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma
        )
    else:
        svm_model = SVC(
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma,
            probability=args.probability
        )
    
    # Create pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', svm_model)
    ])
    
    print("Training SVM model...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    y_pred = pipeline.predict(X_test)
    
    # Get probabilities for relevant metrics if available
    y_prob = None
    if not args.regression and args.probability:
        if len(np.unique(y_train)) == 2:  # Binary classification
            y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    results = compute_metrics(y_test, y_pred, y_prob, is_regression=args.regression)
    
    # Print results
    print("\nResults:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    if args.output_dir:
        # Save model
        from joblib import dump
        dump(pipeline, os.path.join(args.output_dir, "svm_model.joblib"))
        
        # Save results
        result_dict = {
            'args': vars(args),
            'metrics': results,
            'training_time': training_time,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'feature_dim': X_train.shape[1]
        }
        
        with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
            json.dump(result_dict, f, indent=4)
        
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()