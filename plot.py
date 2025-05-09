#!/usr/bin/env python3
# Script to plot LaBraM training results for different models and conditions

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Define base directory
base_dir = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models"

# Model names and conditions (reordered)
#model_names = ["Base setup", "Scratch setup", "Hybrid setup", "Majority Class Baseline"]
model_names = ["Base setup Time Shifts","Hybrid setup Time Shifts"]
conditions = ["sologroup", "friendship", "feedback", "gender"]

# Function to extract metrics from log files
def extract_metrics(log_file_path):
    data = []
    
    try:
        with open(log_file_path, 'r') as file:
            log_content = file.read()
            
            # Find all JSON objects in the log file
            json_pattern = r'\{.*?\}'
            matches = re.findall(json_pattern, log_content)
            
            for json_str in matches:
                try:
                    entry = json.loads(json_str)
                    data.append(entry)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {log_file_path}")
                    continue
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
        return pd.DataFrame()
    
    if not data:
        print(f"No valid data found in {log_file_path}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Collect all results
all_results = []

for model_name in model_names:
    for condition in conditions:
        # Check both possible log file locations
        log_path = os.path.join(base_dir, model_name, condition, "log.txt")
        
        if not os.path.exists(log_path):
            # Try alternate path structure
            log_path = os.path.join(base_dir, f"{model_name}_{condition}", "log.txt")
            if not os.path.exists(log_path):
                print(f"Log file not found for {model_name}, {condition}")
                continue
        
        df = extract_metrics(log_path)
        if not df.empty:
            df['model'] = model_name
            df['condition'] = condition
            all_results.append(df)

# Combine all data
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
else:
    print("No valid results found.")
    exit(1)

# Set up plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid", font_scale=1.2)
# Use colors matching the diagram
model_color_map = {
    "Base setup": "#4a91e9",     # Blue
    "Scratch setup": "#f2993b",  # Orange
    "Hybrid setup": "#5fa55b",   # Green
    "Majority Class Baseline": "#8e7cc3",       # Red (not in diagram, using a complementary color)
    "Base setup Time Shifts": "#4a91e9",     # Blue
    "Hybrid setup Time Shifts": "#5fa55b",   # Green
}

# Create a list of colors in the same order as model_names
colors = [model_color_map[model] for model in model_names]

# Create output directories
output_dir = "result_plots_time_shifts"
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Train Loss by Model for each Condition
plt.figure(figsize=(15, 10))
for i, condition in enumerate(conditions):
    plt.subplot(2, 2, i+1)
    for j, model_name in enumerate(model_names):
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'epoch' in model_data.columns and 'train_loss' in model_data.columns:
            plt.plot(model_data['epoch'], model_data['train_loss'], 
                    marker='o', label=model_name, color=colors[j], linewidth=2)
    
    plt.title(f'Training Loss - {condition}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'train_loss_by_condition.png'), dpi=300, bbox_inches='tight')

# Plot 2: Test Loss by Model for each Condition
plt.figure(figsize=(15, 10))
for i, condition in enumerate(conditions):
    plt.subplot(2, 2, i+1)
    for j, model_name in enumerate(model_names):
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'epoch' in model_data.columns and 'test_loss' in model_data.columns:
            plt.plot(model_data['epoch'], model_data['test_loss'], 
                    marker='o', label=model_name, color=colors[j], linewidth=2)
    
    plt.title(f'Test Loss - {condition}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'test_loss_by_condition.png'), dpi=300, bbox_inches='tight')

# Define baseline accuracies based on class distributions
train_baseline_accuracies = {
    "feedback": 50.00,      # Feedback (train): 50.0% (balanced)
    "sologroup": 60.00,     # Solo/Group (train): 66.89% (group class is dominant)
    "gender": 61.97,        # Gender (train): 61.97% (male class is dominant)
    "friendship": 67.52,    # Friendship (train): 67.52% (friends class is dominant)
}

test_baseline_accuracies = {
    "feedback": 50.03,      # Feedback (test): 50.03% (balanced)
    "sologroup": 60.00,     # Solo/Group (test): 66.74% (group class is dominant)
    "gender": 66.65,        # Gender (test): 66.65% (male class is dominant)
    "friendship": 66.46,    # Friendship (test): 66.46% (friends class is dominant)
}

# Plot 3: Train Accuracy by Model for each Condition
plt.figure(figsize=(15, 10))
for i, condition in enumerate(conditions):
    plt.subplot(2, 2, i+1)
    for j, model_name in enumerate(model_names):
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'epoch' in model_data.columns and 'train_class_acc' in model_data.columns:
            plt.plot(model_data['epoch'], model_data['train_class_acc']*100, 
                    marker='o', label=model_name, color=colors[j], linewidth=2)
    
    # Add baseline accuracy as a red horizontal line
    if condition in train_baseline_accuracies:
        plt.axhline(y=train_baseline_accuracies[condition], color='red', linestyle='--', 
                   linewidth=1.5, label='Majority Class Baseline')
    
    plt.title(f'Training Accuracy - {condition}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'train_accuracy_by_condition.png'), dpi=300, bbox_inches='tight')

# Plot 4: Test Accuracy by Model for each Condition
plt.figure(figsize=(15, 10))
for i, condition in enumerate(conditions):
    plt.subplot(2, 2, i+1)
    for j, model_name in enumerate(model_names):
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'epoch' in model_data.columns and 'test_accuracy' in model_data.columns:
            plt.plot(model_data['epoch'], model_data['test_accuracy']*100, 
                    marker='o', label=model_name, color=colors[j], linewidth=2)
    
    # Add baseline accuracy as a red horizontal line
    if condition in test_baseline_accuracies:
        plt.axhline(y=test_baseline_accuracies[condition], color='red', linestyle='--', 
                   linewidth=1.5, label='Majority Class Baseline')
    
    plt.title(f'Test Accuracy - {condition}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'test_accuracy_by_condition.png'), dpi=300, bbox_inches='tight')

# Plot 5: Final Test Accuracy by Model and Condition (bar chart)
plt.figure(figsize=(14, 8))

# Get final epoch results for each model and condition
final_results = []
for model_name in model_names:
    for condition in conditions:
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'test_accuracy' in model_data.columns:
            final_epoch = model_data['epoch'].max()
            final_acc = model_data[model_data['epoch'] == final_epoch]['test_accuracy'].values[0]
            final_results.append({
                'model': model_name,
                'condition': condition,
                'final_accuracy': final_acc * 100  # Convert to percentage
            })

if final_results:
    final_df = pd.DataFrame(final_results)
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=final_df, 
        x='condition', 
        y='final_accuracy', 
        hue='model',
        palette=colors
    )
    
    # Add baseline accuracy lines for each condition
    for i, condition in enumerate(conditions):
        if condition in test_baseline_accuracies:
            # Calculate the x-coordinate for each condition
            x_coord = i
            # Calculate the width of each condition group
            bar_width = 0.8  # This is typically the default width in seaborn
            # Draw the line just for that condition's position
            plt.plot([x_coord - bar_width/2, x_coord + bar_width/2], 
                     [test_baseline_accuracies[condition], test_baseline_accuracies[condition]], 
                     color='red', linestyle='--', linewidth=2)
    
    # Add a "Majority Class Baseline" entry to the legend
    from matplotlib.lines import Line2D
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', lw=2))
    
    labels = [model for model in model_names]
    labels.append('Majority Class Baseline')
    
    plt.legend(handles=legend_elements, labels=labels, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title('Final Test Accuracy by Model and Condition')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'final_test_accuracy.png'), dpi=300, bbox_inches='tight')

# Plot 6: Training vs Test Loss Comparison
plt.figure(figsize=(15, 10))
for i, model_name in enumerate(model_names):
    plt.subplot(2, 2, i+1)
    for j, condition in enumerate(conditions):
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'epoch' in model_data.columns and 'train_loss' in model_data.columns and 'test_loss' in model_data.columns:
            plt.plot(model_data['epoch'], model_data['train_loss'], 
                    'o-', label=f'{condition} (Train)', alpha=0.7)
            plt.plot(model_data['epoch'], model_data['test_loss'], 
                    's--', label=f'{condition} (Test)', alpha=0.7)
    
    plt.title(f'Training vs Test Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'train_vs_test_loss.png'), dpi=300, bbox_inches='tight')

# Plot 7: Highest Test Accuracy by Model and Condition (bar chart)
plt.figure(figsize=(14, 8))

# Get highest test accuracy for each model and condition
highest_results = []
for model_name in model_names:
    for condition in conditions:
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'test_accuracy' in model_data.columns:
            highest_acc = model_data['test_accuracy'].max()
            best_epoch = model_data[model_data['test_accuracy'] == highest_acc]['epoch'].values[0]
            highest_results.append({
                'model': model_name,
                'condition': condition,
                'highest_accuracy': highest_acc * 100,  # Convert to percentage
                'best_epoch': best_epoch
            })

if highest_results:
    highest_df = pd.DataFrame(highest_results)
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=highest_df, 
        x='condition', 
        y='highest_accuracy', 
        hue='model',
        palette=colors
    )
    
    # Add baseline accuracy lines for each condition
    for i, condition in enumerate(conditions):
        if condition in test_baseline_accuracies:
            # Calculate the x-coordinate for each condition
            x_coord = i
            # Calculate the width of each condition group
            bar_width = 0.8  # This is typically the default width in seaborn
            # Draw the line just for that condition's position
            plt.plot([x_coord - bar_width/2, x_coord + bar_width/2], 
                     [test_baseline_accuracies[condition], test_baseline_accuracies[condition]], 
                     color='red', linestyle='--', linewidth=2)
    
    # Add a "Majority Class Baseline" entry to the legend
    from matplotlib.lines import Line2D
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', lw=2))
    
    labels = [model for model in model_names]
    labels.append('Majority Class Baseline')
    
    plt.legend(handles=legend_elements, labels=labels, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add text annotations for best epoch
    for i, row in enumerate(ax.patches):
        if i < len(highest_df):
            best_epoch = highest_df.iloc[i]['best_epoch']
            x = row.get_x() + row.get_width() / 2
            y = row.get_height() + 0.5  # Slightly above the bar
            ax.text(x, y, f"Epoch {int(best_epoch)}", ha='center', va='bottom', fontsize=8)
    
    plt.title('Highest Test Accuracy by Model and Condition')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'highest_test_accuracy.png'), dpi=300, bbox_inches='tight')
# Create a summary table with best performance
summary_data = []
for model_name in model_names:
    for condition in conditions:
        model_data = combined_df[(combined_df['model'] == model_name) & 
                               (combined_df['condition'] == condition)]
        if not model_data.empty and 'test_accuracy' in model_data.columns:
            best_acc_idx = model_data['test_accuracy'].idxmax()
            best_acc = model_data.loc[best_acc_idx, 'test_accuracy'] * 100
            best_epoch = model_data.loc[best_acc_idx, 'epoch']
            
            summary_data.append({
                'Model': model_name,
                'Condition': condition,
                'Best Test Accuracy (%)': round(best_acc, 2),
                'Best Epoch': int(best_epoch),
                'Final Train Accuracy (%)': round(model_data.iloc[-1]['train_class_acc'] * 100, 2)
            })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    
    # Print summary
    print("\nPerformance Summary:\n")
    print(summary_df.to_string(index=False))

print(f"\nPlots and summary saved to {output_dir}/")