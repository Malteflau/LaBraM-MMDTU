import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12})

def load_log_data(log_path):
    """Load log data from a LaBraM log.txt file"""
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {log_path}")
    return data

def extract_metrics(data):
    """Extract the relevant metrics from the log data"""
    epochs = []
    train_losses = []
    mlm_acc = []
    
    for entry in data:
        epoch = entry.get('epoch', None)
        if epoch is not None:
            epochs.append(epoch)
            
            # Extract training loss
            train_loss = entry.get('train_loss', None)
            if train_loss is not None:
                train_losses.append(train_loss)
                
            # Extract masked EEG modeling accuracy
            accuracy = entry.get('train_mlm_acc', None)
            if accuracy is None:
                # Try alternate keys that might hold accuracy
                accuracy = entry.get('train_class_acc', None)
            
            if accuracy is not None:
                mlm_acc.append(accuracy)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'mlm_acc': mlm_acc
    }

def plot_training_loss(metrics_dict1, metrics_dict2, metrics_dict3, exp_names):
    """Plot the training loss from three different experiments"""
    plt.figure(figsize=(10, 6))
    
    # Define colors: Hybrid (green), Scratch (orange), Beta band (purple)
    colors = ['#2E8B57', '#FF8C00', '#570157']
    
    plt.plot(metrics_dict1['epochs'], metrics_dict1['train_losses'], 'o-', label=exp_names[0], color=colors[0])
    plt.plot(metrics_dict2['epochs'], metrics_dict2['train_losses'], 's-', label=exp_names[1], color=colors[1])
    plt.plot(metrics_dict3['epochs'], metrics_dict3['train_losses'], '^-', label=exp_names[2], color=colors[2])
    
    plt.title('Pre-training Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_mlm_accuracy(metrics_dict1, metrics_dict2, metrics_dict3, exp_names):
    """Plot the masked EEG modeling accuracy from three different experiments"""
    plt.figure(figsize=(10, 6))
    
    # Define colors: Hybrid (green), Scratch (orange), Beta band (purple)
    colors = ['#2E8B57', '#FF8C00', '#570157']
    
    # Only plot for experiments that have MLM accuracy
    if metrics_dict1['mlm_acc']:
        plt.plot(metrics_dict1['epochs'][:len(metrics_dict1['mlm_acc'])], 
                 metrics_dict1['mlm_acc'], 'o-', label=exp_names[0], color=colors[0])
    
    if metrics_dict2['mlm_acc']:
        plt.plot(metrics_dict2['epochs'][:len(metrics_dict2['mlm_acc'])], 
                 metrics_dict2['mlm_acc'], 's-', label=exp_names[1], color=colors[1])
    
    if metrics_dict3['mlm_acc']:
        plt.plot(metrics_dict3['epochs'][:len(metrics_dict3['mlm_acc'])], 
                 metrics_dict3['mlm_acc'], '^-', label=exp_names[2], color=colors[2])
    
    plt.title('Masked EEG Modeling Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/mlm_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Paths to log files
    path1 = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/finetune_original_vqnsp/log.txt"
    path2 = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/finetune_dtu_labram/log.txt"
    path3 = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/pretrain_betaband/log.txt"
    
    # Names for the experiments (for plot legends)
    exp_names = ["Hybrid setup", "Scratch setup", "Beta band setup"]
    
    # Load the data
    data1 = load_log_data(path1)
    data2 = load_log_data(path2)
    data3 = load_log_data(path3)
    
    # Extract metrics
    metrics1 = extract_metrics(data1)
    metrics2 = extract_metrics(data2)
    metrics3 = extract_metrics(data3)
    
    # Print how many data points we have for each metric
    print(f"Experiment 1 ({exp_names[0]}):")
    print(f"  Epochs: {len(metrics1['epochs'])}")
    print(f"  Train Losses: {len(metrics1['train_losses'])}")
    print(f"  MLM Accuracy: {len(metrics1['mlm_acc'])}")
    
    print(f"\nExperiment 2 ({exp_names[1]}):")
    print(f"  Epochs: {len(metrics2['epochs'])}")
    print(f"  Train Losses: {len(metrics2['train_losses'])}")
    print(f"  MLM Accuracy: {len(metrics2['mlm_acc'])}")
    
    print(f"\nExperiment 3 ({exp_names[2]}):")
    print(f"  Epochs: {len(metrics3['epochs'])}")
    print(f"  Train Losses: {len(metrics3['train_losses'])}")
    print(f"  MLM Accuracy: {len(metrics3['mlm_acc'])}")
    
    # Plot the metrics
    plot_training_loss(metrics1, metrics2, metrics3, exp_names)
    plot_mlm_accuracy(metrics1, metrics2, metrics3, exp_names)

if __name__ == "__main__":
    main()