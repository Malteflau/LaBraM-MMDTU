#!/usr/bin/env python
# --------------------------------------------------------
# Plotting script for LaBraM experiments on betaband classification
# --------------------------------------------------------

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_log_data(log_file_path):
    """
    Load data from a log.txt file.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        Dictionary containing training and testing metrics across epochs
    """
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_class_acc': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract metrics if they exist
                    metrics['epoch'].append(data.get('epoch', None))
                    metrics['train_loss'].append(data.get('train_loss', None))
                    metrics['train_class_acc'].append(data.get('train_class_acc', None))
                    metrics['test_loss'].append(data.get('test_loss', None))
                    metrics['test_accuracy'].append(data.get('test_accuracy', None))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {log_file_path}")
                    continue
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    
    # Convert lists to numpy arrays for easier manipulation
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics

def plot_metrics(conditions, base_dir, output_dir=None):
    """
    Plot metrics for multiple conditions.
    
    Args:
        conditions: List of condition names
        base_dir: Base directory containing condition folders
        output_dir: Directory to save output plots (defaults to base_dir)
    """
    if output_dir is None:
        output_dir = base_dir
    
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up the figure for comparing all conditions
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training process for Betaband classification', fontsize=16)
    
    # Define consistent colors for each condition
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # For storing data to be used in combined plots
    all_data = {}
    
    for i, condition in enumerate(conditions):
        log_path = os.path.join(base_dir, condition, 'log.txt')
        metrics = load_log_data(log_path)
        
        if metrics is None:
            print(f"Skipping condition {condition} due to missing or invalid log file")
            continue
        
        all_data[condition] = metrics
        
        # Also create individual plots for each condition
        create_individual_plot(condition, metrics, output_dir, colors[i])
    
    # Create combined plots
    # Top-left: Training Loss
    ax = axes[0, 0]
    for i, (condition, metrics) in enumerate(all_data.items()):
        ax.plot(metrics['epoch'], metrics['train_loss'], 
                label=condition, color=colors[i])
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Top-right: Test Loss
    ax = axes[0, 1]
    for i, (condition, metrics) in enumerate(all_data.items()):
        ax.plot(metrics['epoch'], metrics['test_loss'], 
                label=condition, color=colors[i])
    ax.set_title('Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom-left: Training Accuracy
    ax = axes[1, 0]
    for i, (condition, metrics) in enumerate(all_data.items()):
        ax.plot(metrics['epoch'], metrics['train_class_acc'], 
                label=condition, color=colors[i])
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom-right: Test Accuracy
    ax = axes[1, 1]
    for i, (condition, metrics) in enumerate(all_data.items()):
        ax.plot(metrics['epoch'], metrics['test_accuracy'], 
                label=condition, color=colors[i])
    ax.set_title('Test Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    combined_plot_path = os.path.join(output_dir, 'betaband_all_conditions.png')
    plt.savefig(combined_plot_path, dpi=300)
    print(f"Combined plot saved to {combined_plot_path}")
    plt.close(fig)

def create_individual_plot(condition, metrics, output_dir, color):
    """
    Create individual plots for a specific condition.
    
    Args:
        condition: Condition name
        metrics: Dictionary of metrics for this condition
        output_dir: Directory to save output plots
        color: Color to use for plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f'BetaBand Classification Metrics - {condition}', fontsize=16)
    
    # Training Loss
    ax = axes[0, 0]
    ax.plot(metrics['epoch'], metrics['train_loss'], 
            label='Train Loss', color=color)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Test Loss
    ax = axes[0, 1]
    ax.plot(metrics['epoch'], metrics['test_loss'], 
            label='Test Loss', color=color, marker='o', markersize=4)
    ax.set_title('Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Training Accuracy
    ax = axes[1, 0]
    ax.plot(metrics['epoch'], metrics['train_class_acc'], 
            label='Train Accuracy', color=color, marker='o', markersize=4)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Test Accuracy
    ax = axes[1, 1]
    ax.plot(metrics['epoch'], metrics['test_accuracy'], 
            label='Test Accuracy', color=color, marker='o', markersize=4)
    ax.set_title('Test Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    individual_plot_path = os.path.join(output_dir, f'betaband_{condition}.png')
    plt.savefig(individual_plot_path, dpi=300)
    print(f"Individual plot for {condition} saved to {individual_plot_path}")
    plt.close(fig)

def main():
    # Base directory containing all condition folders
    base_dir = "checkpoints/Final_models/betaband_on_betaband"
    
    # List of conditions to plot
    conditions = ["feedback", "friendship", "gender", "sologroup"]
    
    # Output directory for plots
    output_dir = os.path.join(base_dir, "plots")
    
    # Create plots
    plot_metrics(conditions, base_dir, output_dir)
    
    print("Plotting complete!")

if __name__ == "__main__":
    main()