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
    rec_losses = []
    angle_losses = []
    total_losses = []
    unused_codes = []
    
    for entry in data:
        epoch = entry.get('epoch', None)
        if epoch is not None:
            epochs.append(epoch)
            
            # For training data
            rec_loss = entry.get('train_rec_loss', None)
            angle_loss = entry.get('train_rec_angle_loss', None)
            total_loss = entry.get('train_loss', None)
            
            if rec_loss is not None:
                rec_losses.append(rec_loss)
            if angle_loss is not None:
                angle_losses.append(angle_loss)
            if total_loss is not None:
                total_losses.append(total_loss)
            
            # For test data (unused codebook)
            unused_code = entry.get('test_unused_code', entry.get('test_Unused_code', None))
            if unused_code is not None:
                unused_codes.append(unused_code)
    
    return {
        'epochs': epochs,
        'rec_losses': rec_losses,
        'angle_losses': angle_losses,
        'total_losses': total_losses,
        'unused_codes': unused_codes
    }

def plot_metrics(metrics_dict1, metrics_dict2, exp_names):
    """Plot the metrics from two different experiments"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of Training Metrics in VQNSP', fontsize=16)
    
    # Define colors: orange for Scratch setup, green for Beta band setup
    colors = ['#FF8C00', "#570157"]  # Orange, Forest Green
    markers = ['o', 's']
    
    # Plot reconstruction loss
    ax = axes[0, 0]
    ax.plot(metrics_dict1['epochs'], metrics_dict1['rec_losses'], 
            marker=markers[0], color=colors[0], linestyle='-', label=f"{exp_names[0]}")
    ax.plot(metrics_dict2['epochs'], metrics_dict2['rec_losses'], 
            marker=markers[1], color=colors[1], linestyle='-', label=f"{exp_names[1]}")
    ax.set_title('Amplitude Reconstruction Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.text(0.05, 0.95, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot angle loss
    ax = axes[0, 1]
    ax.plot(metrics_dict1['epochs'], metrics_dict1['angle_losses'], 
            marker=markers[0], color=colors[0], linestyle='-', label=f"{exp_names[0]}")
    ax.plot(metrics_dict2['epochs'], metrics_dict2['angle_losses'], 
            marker=markers[1], color=colors[1], linestyle='-', label=f"{exp_names[1]}")
    ax.set_title('Angle Reconstruction Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.text(0.05, 0.95, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot total loss
    ax = axes[1, 0]
    ax.plot(metrics_dict1['epochs'], metrics_dict1['total_losses'], 
            marker=markers[0], color=colors[0], linestyle='-', label=f"{exp_names[0]}")
    ax.plot(metrics_dict2['epochs'], metrics_dict2['total_losses'], 
            marker=markers[1], color=colors[1], linestyle='-', label=f"{exp_names[1]}")
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.text(0.05, 0.95, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot unused codebook
    ax = axes[1, 1]
    ax.plot(metrics_dict1['epochs'], metrics_dict1['unused_codes'], 
            marker=markers[0], color=colors[0], linestyle='-', label=f"{exp_names[0]}")
    ax.plot(metrics_dict2['epochs'], metrics_dict2['unused_codes'], 
            marker=markers[1], color=colors[1], linestyle='-', label=f"{exp_names[1]}")
    ax.set_title('Unused Codebook Entries')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Count')
    ax.legend()
    ax.text(0.05, 0.95, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Paths to log files
    path1 = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/finetune_dtu_vqnsp/log.txt"
    path2 = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/dtu_vqnsp_betaband/log.txt"
    
    # Names for the experiments (for plot legends)
    exp_names = ["Scratch setup", "Beta band setup"]
    
    # Load the data
    data1 = load_log_data(path1)
    data2 = load_log_data(path2)
    
    # Extract metrics
    metrics1 = extract_metrics(data1)
    metrics2 = extract_metrics(data2)
    
    # Plot the metrics
    plot_metrics(metrics1, metrics2, exp_names)
    
    # Print some statistics
    print(f"Final metrics for {exp_names[0]}:")
    print(f"  Reconstruction Loss: {metrics1['rec_losses'][-1]:.4f}")
    print(f"  Angle Loss: {metrics1['angle_losses'][-1]:.4f}")
    print(f"  Total Loss: {metrics1['total_losses'][-1]:.4f}")
    print(f"  Unused Codebook Entries: {metrics1['unused_codes'][-1]}")
    
    print(f"\nFinal metrics for {exp_names[1]}:")
    print(f"  Reconstruction Loss: {metrics2['rec_losses'][-1]:.4f}")
    print(f"  Angle Loss: {metrics2['angle_losses'][-1]:.4f}")
    print(f"  Total Loss: {metrics2['total_losses'][-1]:.4f}")
    print(f"  Unused Codebook Entries: {metrics2['unused_codes'][-1]}")

if __name__ == "__main__":
    main()