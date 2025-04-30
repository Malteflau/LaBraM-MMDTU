#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Analyzer for LaBraM EEG Classification Tasks

This script analyzes the DTU Force Game dataset to get class distributions 
for different classification targets: feedback, solo/group, gender, and friendship status.
"""

import os
import pickle
import argparse
import pandas as pd
import numpy as np
import traceback
from collections import Counter
from pathlib import Path

def analyze_pickle_file(file_path, debug=False):
    """
    Extracts classification targets from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file
        debug (bool): Whether to print debug information
        
    Returns:
        dict: Dictionary containing classification targets
    """
    try:
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
        
        # The condition may be stored directly or in a different field
        # First try to get directly from known fields
        condition = None
        participant_num = None
        has_feedback = None
        gender = None
        friend_status = None
        
        # Try getting 'condition' field
        if 'condition' in sample:
            condition = sample['condition']
        elif 'trial_condition' in sample:
            condition = sample['trial_condition']
        
        # Try getting 'participant_num' field
        if 'participant_num' in sample:
            participant_num = sample['participant_num']
        elif 'participant' in sample:
            participant_num = sample['participant']
            
        # Try getting feedback info
        if 'has_feedback' in sample:
            has_feedback = sample['has_feedback']
        elif 'feedback' in sample:
            has_feedback = sample['feedback']
        else:
            # Try to infer from condition name if it ends with 'Pn' (no feedback)
            if condition and isinstance(condition, str):
                has_feedback = not condition.endswith('Pn')
        
        # Try getting gender info
        if 'gender' in sample:
            gender = sample['gender']
            
        # Try getting friendship info
        if 'friend_status' in sample:
            friend_status = sample['friend_status']
        elif 'friendship' in sample:
            friend_status = sample['friendship']
            
        # If we're still missing critical info, try a more comprehensive search
        if condition is None or participant_num is None:
            # Check if condition might be in a nested field
            for key, value in sample.items():
                if isinstance(value, dict):
                    if 'condition' in value and condition is None:
                        condition = value['condition']
                    if 'participant_num' in value and participant_num is None:
                        participant_num = value['participant_num']
                    if 'has_feedback' in value and has_feedback is None:
                        has_feedback = value['has_feedback']
                    if 'gender' in value and gender is None:
                        gender = value['gender']
                    if 'friend_status' in value and friend_status is None:
                        friend_status = value['friend_status']
        
        # If still no condition, let's look at the filename for clues
        if condition is None:
            filename = os.path.basename(file_path)
            if "_T1P_" in filename:
                condition = "T1P"
            elif "_T1Pn_" in filename:
                condition = "T1Pn"
            elif "_T3P_" in filename:
                condition = "T3P"
            elif "_T3Pn_" in filename:
                condition = "T3Pn"
            elif "_T12P_" in filename:
                condition = "T12P"
            elif "_T12Pn_" in filename:
                condition = "T12Pn"
            elif "_T13P_" in filename:
                condition = "T13P"
            elif "_T13Pn_" in filename:
                condition = "T13Pn"
            elif "_T23P_" in filename:
                condition = "T23P"
            elif "_T23Pn_" in filename:
                condition = "T23Pn"
        
        # We need condition and participant_num to determine solo/group
        if condition is None or participant_num is None:
            if debug:
                print(f"Warning: Missing critical data in {file_path}")
                print(f"  condition: {condition}")
                print(f"  participant_num: {participant_num}")
            return None
        
        # Determine solo/group classification using correct logic
        is_solo = is_solo_condition(condition, participant_num)
            
        return {
            'feedback': 1 if has_feedback else 0,
            'solo': 1 if is_solo else 0,
            'gender': 1 if gender == 'M' else 0 if gender == 'F' else None,
            'friendship': 1 if friend_status == 'Yes' else 0 if friend_status == 'No' else None,
            'condition': condition,
            'participant': participant_num
        }
        
    except Exception as e:
        if debug:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
        return None

def is_solo_condition(condition_str, participant_num):
    """
    Helper method to determine if a trial is solo for this participant.
    
    Args:
        condition_str (str): The condition string (e.g., 'T1P', 'T23Pn')
        participant_num (str): The participant number ('P1', 'P2', 'P3')
        
    Returns:
        bool: True if this is a solo condition for this participant, False otherwise
    """
    # Handle possible None or non-string inputs
    if not condition_str or not isinstance(condition_str, str):
        return False
    if not participant_num or not isinstance(participant_num, str):
        return False
        
    # Make sure condition_str and participant_num are properly formatted
    condition_str = condition_str.strip()
    participant_num = participant_num.strip()
    
    # Make sure participant_num is in the right format (P1, P2, P3)
    if not participant_num.startswith('P'):
        if participant_num == '1':
            participant_num = 'P1'
        elif participant_num == '2':
            participant_num = 'P2'
        elif participant_num == '3':
            participant_num = 'P3'
        else:
            return False  # Invalid participant number
    
    # Strip any potential prefixes from condition (e.g., "condition_" prefix)
    if "_" in condition_str:
        parts = condition_str.split("_")
        for part in parts:
            if part.startswith("T") and (part.startswith("T1") or part.startswith("T2") or part.startswith("T3")):
                condition_str = part
                break
    
    # Based on the DTULoader logic in the provided code
    # Check specific conditions where participant is not involved (these are solo conditions)
    if condition_str.startswith("T23") and participant_num == "P1":
        return True  # P1 is not involved in T23 condition
    elif condition_str.startswith("T13") and participant_num == "P2":
        return True  # P2 is not involved in T13 condition
    elif condition_str.startswith("T12") and participant_num == "P3":
        return True  # P3 is not involved in T12 condition
    
    # Direct solo conditions
    if (condition_str.startswith("T1") and not condition_str.startswith("T12") and 
        not condition_str.startswith("T13") and participant_num == "P1"):
        return True
        
    if (condition_str.startswith("T3") and not condition_str.startswith("T32") and 
        not condition_str.startswith("T31") and participant_num == "P3"):
        return True
    
    # If none of the above, it's a group condition
    return False

def analyze_dataset_directory(directory, debug_limit=5):
    """
    Analyzes all pickle files in a directory.
    
    Args:
        directory (str): Directory containing pickle files
        debug_limit (int): Number of files to print debug info for
        
    Returns:
        dict: Dictionary with counts for each classification target
    """
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return None
    
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    if not files:
        print(f"No pickle files found in: {directory}")
        return None
    
    # Only analyze a few files for debugging purposes to avoid overwhelming output
    print(f"\nDEBUG - Examining {min(debug_limit, len(files))} sample files from {directory}:")
    
    # Look at a few files in detail
    for i, file in enumerate(files[:debug_limit]):
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            print(f"\nDEBUG - Sample file {i+1}: {file}")
            print(f"DEBUG - Keys in sample: {list(sample.keys())}")
            print(f"DEBUG - Example values:")
            
            # Extract and print important fields to debug solo/group logic
            for key in ['condition', 'participant_num', 'condition_type', 'has_feedback', 'gender', 'friend_status']:
                if key in sample:
                    print(f"  - {key}: {sample[key]}")
        except Exception as e:
            print(f"Error examining {file_path}: {e}")
    
    # Now process all files normally
    results = []
    
    for file in files:
        file_path = os.path.join(directory, file)
        result = analyze_pickle_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        return None
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Get counts for each target
    feedback_counts = df['feedback'].value_counts()
    solo_counts = df['solo'].value_counts()
    gender_counts = df['gender'].value_counts()
    friendship_counts = df['friendship'].value_counts()
    
    # Calculate percentages
    total = len(df)
    feedback_pct = (feedback_counts / total * 100).round(2)
    solo_pct = (solo_counts / total * 100).round(2)
    gender_pct = (gender_counts / total * 100).round(2)
    friendship_pct = (friendship_counts / total * 100).round(2)
    
    return {
        'total_samples': total,
        'feedback': {
            'counts': feedback_counts.to_dict(),
            'percentages': feedback_pct.to_dict()
        },
        'solo': {
            'counts': solo_counts.to_dict(),
            'percentages': solo_pct.to_dict()
        },
        'gender': {
            'counts': gender_counts.to_dict(),
            'percentages': gender_pct.to_dict()
        },
        'friendship': {
            'counts': friendship_counts.to_dict(),
            'percentages': friendship_pct.to_dict()
        }
    }

def print_distribution(dataset_name, distribution):
    """
    Prints the distribution in a readable format.
    
    Args:
        dataset_name (str): Name of the dataset
        distribution (dict): Distribution dictionary
    """
    print(f"\n=== {dataset_name} Dataset Distribution ===")
    print(f"Total samples: {distribution['total_samples']}")
    
    print("\nFeedback Distribution (1=Feedback, 0=No Feedback):")
    for k, v in distribution['feedback']['counts'].items():
        print(f"  Class {k}: {v} samples ({distribution['feedback']['percentages'][k]}%)")
    
    print("\nSolo/Group Distribution (1=Solo, 0=Group):")
    for k, v in distribution['solo']['counts'].items():
        print(f"  Class {k}: {v} samples ({distribution['solo']['percentages'][k]}%)")
    
    print("\nGender Distribution (1=Male, 0=Female):")
    for k, v in distribution['gender']['counts'].items():
        print(f"  Class {k}: {v} samples ({distribution['gender']['percentages'][k]}%)")
    
    print("\nFriendship Distribution (1=Friends, 0=Not Friends):")
    for k, v in distribution['friendship']['counts'].items():
        print(f"  Class {k}: {v} samples ({distribution['friendship']['percentages'][k]}%)")

def analyze_all_datasets(base_dir):
    """
    Analyzes all datasets (train, val, test).
    
    Args:
        base_dir (str): Base directory containing train, val, test subdirectories
    """
    results = {}
    
    # Analyze each dataset separately
    print("\n" + "="*80)
    print("INDIVIDUAL DATASET DISTRIBUTIONS")
    print("="*80)
    
    for dataset in ['train', 'val', 'test']:
        dataset_dir = os.path.join(base_dir, dataset)
        distribution = analyze_dataset_directory(dataset_dir)
        if distribution:
            results[dataset] = distribution
            print_distribution(dataset, distribution)
            print("\n" + "-"*80)
    
    # Calculate and print combined distribution
    print("\n" + "="*80)
    print("COMBINED DISTRIBUTION (ALL DATASETS)")
    print("="*80)
    
    all_samples = sum(results[dataset]['total_samples'] for dataset in results)
    print(f"Total samples across all datasets: {all_samples}")
    
    # Initialize dictionaries for combined counts
    total_feedback = {0: 0, 1: 0}
    total_solo = {0: 0, 1: 0}
    total_gender = {0: 0, 1: 0}
    total_friendship = {0: 0, 1: 0}
    
    # Combine counts across datasets
    for dataset in results:
        for k, v in results[dataset]['feedback']['counts'].items():
            total_feedback[k] = total_feedback.get(k, 0) + v
        for k, v in results[dataset]['solo']['counts'].items():
            total_solo[k] = total_solo.get(k, 0) + v
        for k, v in results[dataset]['gender']['counts'].items():
            total_gender[k] = total_gender.get(k, 0) + v
        for k, v in results[dataset]['friendship']['counts'].items():
            total_friendship[k] = total_friendship.get(k, 0) + v
    
    # Print combined distributions
    print("\nFeedback Distribution (1=Feedback, 0=No Feedback):")
    for k, v in total_feedback.items():
        print(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)")
    
    print("\nSolo/Group Distribution (1=Solo, 0=Group):")
    for k, v in total_solo.items():
        print(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)")
    
    print("\nGender Distribution (1=Male, 0=Female):")
    for k, v in total_gender.items():
        print(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)")
    
    print("\nFriendship Distribution (1=Friends, 0=Not Friends):")
    for k, v in total_friendship.items():
        print(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)")
    
    # Table view for summary at the end
    print("\n" + "="*80)
    print("DISTRIBUTION SUMMARY (ALL DATASETS)")
    print("="*80)
    
    # Create a summary table
    print(f"{'Dataset':<10} {'Total':<8} {'Feedback':<20} {'Solo/Group':<20} {'Gender':<20} {'Friendship':<20}")
    print(f"{'':^10} {'':^8} {'(1/0)':<20} {'(1/0)':<20} {'(M/F)':<20} {'(Yes/No)':<20}")
    print("-"*100)
    
    for dataset in results:
        fb_1 = results[dataset]['feedback']['counts'].get(1, 0)
        fb_0 = results[dataset]['feedback']['counts'].get(0, 0)
        solo_1 = results[dataset]['solo']['counts'].get(1, 0)
        solo_0 = results[dataset]['solo']['counts'].get(0, 0)
        gender_1 = results[dataset]['gender']['counts'].get(1, 0)
        gender_0 = results[dataset]['gender']['counts'].get(0, 0)
        friend_1 = results[dataset]['friendship']['counts'].get(1, 0)
        friend_0 = results[dataset]['friendship']['counts'].get(0, 0)
        
        total = results[dataset]['total_samples']
        
        print(f"{dataset:<10} {total:<8} {fb_1}/{fb_0:<18} {solo_1}/{solo_0:<18} {gender_1}/{gender_0:<18} {friend_1}/{friend_0:<18}")
    
    # All datasets row
    print("-"*100)
    print(f"{'All':<10} {all_samples:<8} {total_feedback.get(1,0)}/{total_feedback.get(0,0):<18} " +
          f"{total_solo.get(1,0)}/{total_solo.get(0,0):<18} {total_gender.get(1,0)}/{total_gender.get(0,0):<18} " +
          f"{total_friendship.get(1,0)}/{total_friendship.get(0,0):<18}")
    
    # Save results to CSV files
    save_results_to_csv(results, base_dir)

def save_results_to_csv(results, base_dir):
    """
    Saves analysis results to CSV files.
    
    Args:
        results (dict): Results dictionary
        base_dir (str): Base directory to save CSV files
    """
    output_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames for each classification task
    feedback_df = pd.DataFrame(columns=['dataset', 'class', 'count', 'percentage'])
    solo_df = pd.DataFrame(columns=['dataset', 'class', 'count', 'percentage'])
    gender_df = pd.DataFrame(columns=['dataset', 'class', 'count', 'percentage'])
    friendship_df = pd.DataFrame(columns=['dataset', 'class', 'count', 'percentage'])
    
    # Fill DataFrames
    for dataset, distribution in results.items():
        for k, v in distribution['feedback']['counts'].items():
            new_row = {
                'dataset': dataset,
                'class': k,
                'count': v,
                'percentage': distribution['feedback']['percentages'][k]
            }
            feedback_df = pd.concat([feedback_df, pd.DataFrame([new_row])], ignore_index=True)
        
        for k, v in distribution['solo']['counts'].items():
            new_row = {
                'dataset': dataset,
                'class': k,
                'count': v,
                'percentage': distribution['solo']['percentages'][k]
            }
            solo_df = pd.concat([solo_df, pd.DataFrame([new_row])], ignore_index=True)
        
        for k, v in distribution['gender']['counts'].items():
            new_row = {
                'dataset': dataset,
                'class': k,
                'count': v,
                'percentage': distribution['gender']['percentages'][k]
            }
            gender_df = pd.concat([gender_df, pd.DataFrame([new_row])], ignore_index=True)
        
        for k, v in distribution['friendship']['counts'].items():
            new_row = {
                'dataset': dataset,
                'class': k,
                'count': v,
                'percentage': distribution['friendship']['percentages'][k]
            }
            friendship_df = pd.concat([friendship_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Create a single summary CSV with all distributions
    summary_data = []
    for dataset, distribution in results.items():
        for task in ['feedback', 'solo', 'gender', 'friendship']:
            for cls, count in distribution[task]['counts'].items():
                summary_data.append({
                    'dataset': dataset,
                    'task': task,
                    'class': cls,
                    'count': count,
                    'percentage': distribution[task]['percentages'][cls]
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    feedback_df.to_csv(os.path.join(output_dir, 'feedback_distribution.csv'), index=False)
    solo_df.to_csv(os.path.join(output_dir, 'solo_group_distribution.csv'), index=False)
    gender_df.to_csv(os.path.join(output_dir, 'gender_distribution.csv'), index=False)
    friendship_df.to_csv(os.path.join(output_dir, 'friendship_distribution.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'all_distributions_summary.csv'), index=False)
    
    # Create a highly readable text file report
    with open(os.path.join(output_dir, 'dataset_distribution_report.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET DISTRIBUTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Write distribution for each dataset
        for dataset in results:
            f.write(f"=== {dataset.upper()} DATASET ===\n")
            f.write(f"Total samples: {results[dataset]['total_samples']}\n\n")
            
            f.write("Feedback Distribution (1=Feedback, 0=No Feedback):\n")
            for k, v in results[dataset]['feedback']['counts'].items():
                f.write(f"  Class {k}: {v} samples ({results[dataset]['feedback']['percentages'][k]}%)\n")
            
            f.write("\nSolo/Group Distribution (1=Solo, 0=Group):\n")
            for k, v in results[dataset]['solo']['counts'].items():
                f.write(f"  Class {k}: {v} samples ({results[dataset]['solo']['percentages'][k]}%)\n")
            
            f.write("\nGender Distribution (1=Male, 0=Female):\n")
            for k, v in results[dataset]['gender']['counts'].items():
                f.write(f"  Class {k}: {v} samples ({results[dataset]['gender']['percentages'][k]}%)\n")
            
            f.write("\nFriendship Distribution (1=Friends, 0=Not Friends):\n")
            for k, v in results[dataset]['friendship']['counts'].items():
                f.write(f"  Class {k}: {v} samples ({results[dataset]['friendship']['percentages'][k]}%)\n")
            
            f.write("\n" + "-"*80 + "\n\n")
        
        # Calculate combined distributions
        all_samples = sum(results[dataset]['total_samples'] for dataset in results)
        
        total_feedback = {0: 0, 1: 0}
        total_solo = {0: 0, 1: 0}
        total_gender = {0: 0, 1: 0}
        total_friendship = {0: 0, 1: 0}
        
        for dataset in results:
            for k, v in results[dataset]['feedback']['counts'].items():
                total_feedback[k] = total_feedback.get(k, 0) + v
            for k, v in results[dataset]['solo']['counts'].items():
                total_solo[k] = total_solo.get(k, 0) + v
            for k, v in results[dataset]['gender']['counts'].items():
                total_gender[k] = total_gender.get(k, 0) + v
            for k, v in results[dataset]['friendship']['counts'].items():
                total_friendship[k] = total_friendship.get(k, 0) + v
        
        # Write combined distribution
        f.write("=== COMBINED (ALL DATASETS) ===\n")
        f.write(f"Total samples: {all_samples}\n\n")
        
        f.write("Feedback Distribution (1=Feedback, 0=No Feedback):\n")
        for k, v in total_feedback.items():
            f.write(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)\n")
        
        f.write("\nSolo/Group Distribution (1=Solo, 0=Group):\n")
        for k, v in total_solo.items():
            f.write(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)\n")
        
        f.write("\nGender Distribution (1=Male, 0=Female):\n")
        for k, v in total_gender.items():
            f.write(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)\n")
        
        f.write("\nFriendship Distribution (1=Friends, 0=Not Friends):\n")
        for k, v in total_friendship.items():
            f.write(f"  Class {k}: {v} samples ({v/all_samples*100:.2f}%)\n")
    
    print(f"\nDistribution CSV files and text report saved to {output_dir}")

def create_sample_distribution_plots(base_dir):
    """
    Creates bar plots for distributions.
    
    Args:
        base_dir (str): Base directory containing analysis CSV files
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        analysis_dir = os.path.join(base_dir, 'analysis')
        output_dir = os.path.join(analysis_dir, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the CSV files to plot
        csv_files = {
            'feedback': os.path.join(analysis_dir, 'feedback_distribution.csv'),
            'solo_group': os.path.join(analysis_dir, 'solo_group_distribution.csv'),
            'gender': os.path.join(analysis_dir, 'gender_distribution.csv'),
            'friendship': os.path.join(analysis_dir, 'friendship_distribution.csv')
        }
        
        # Labels for the plots
        class_labels = {
            'feedback': {0: 'No Feedback', 1: 'Feedback'},
            'solo_group': {0: 'Group', 1: 'Solo'},
            'gender': {0: 'Female', 1: 'Male'},
            'friendship': {0: 'Not Friends', 1: 'Friends'}
        }
        
        # Create plots for each classification task
        for task, csv_file in csv_files.items():
            df = pd.read_csv(csv_file)
            
            # Create figure with 2 subplots (counts and percentages)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Set a custom color palette
            colors = sns.color_palette("Set2", n_colors=2)
            
            # Plot counts
            sns.barplot(x='dataset', y='count', hue='class', data=df, ax=ax1, palette=colors)
            ax1.set_title(f'{task.capitalize()} Distribution (Counts)')
            ax1.set_xlabel('Dataset')
            ax1.set_ylabel('Count')
            
            # Replace numeric class labels with text
            legend_labels = [class_labels[task][int(label.get_text())] for label in ax1.get_legend().get_texts()]
            for i, text in enumerate(ax1.get_legend().get_texts()):
                text.set_text(legend_labels[i])
            
            # Plot percentages
            sns.barplot(x='dataset', y='percentage', hue='class', data=df, ax=ax2, palette=colors)
            ax2.set_title(f'{task.capitalize()} Distribution (Percentages)')
            ax2.set_xlabel('Dataset')
            ax2.set_ylabel('Percentage (%)')
            
            # Replace numeric class labels with text
            legend_labels = [class_labels[task][int(label.get_text())] for label in ax2.get_legend().get_texts()]
            for i, text in enumerate(ax2.get_legend().get_texts()):
                text.set_text(legend_labels[i])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task}_distribution.png'))
            plt.close()
        
        print(f"Distribution plots saved to {output_dir}")
        
    except ImportError:
        print("Matplotlib and/or seaborn not installed. Skipping plot creation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze DTU dataset class distributions')
    parser.add_argument('--data_dir', type=str, default='/work3/s224183/LaBraM_data',
                        help='Base directory containing train, val, test subdirectories')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Create distribution plots (requires matplotlib and seaborn)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Print detailed debug information')
    parser.add_argument('--debug_samples', type=int, default=5,
                        help='Number of sample files to examine in detail for debugging')
    
    args = parser.parse_args()
    
    print(f"Analyzing datasets in {args.data_dir}")
    
    if args.debug:
        # Examine one sample file in extreme detail to debug
        print("\n" + "="*80)
        print("DETAILED FILE EXAMINATION")
        print("="*80)
        
        # Find a pickle file to examine
        sample_file = None
        for subset in ['train', 'val', 'test']:
            subset_dir = os.path.join(args.data_dir, subset)
            if os.path.exists(subset_dir):
                files = [f for f in os.listdir(subset_dir) if f.endswith('.pkl')]
                if files:
                    sample_file = os.path.join(subset_dir, files[0])
                    break
        
        if sample_file:
            print(f"Examining sample file: {sample_file}")
            try:
                with open(sample_file, 'rb') as f:
                    sample = pickle.load(f)
                
                print(f"File type: {type(sample)}")
                if isinstance(sample, dict):
                    print(f"Top-level keys: {list(sample.keys())}")
                    for key, value in sample.items():
                        print(f"\nKey: {key}")
                        print(f"Value type: {type(value)}")
                        if isinstance(value, (str, int, float, bool)):
                            print(f"Value: {value}")
                        elif isinstance(value, (list, tuple)) and len(value) < 20:
                            print(f"Value: {value}")
                        elif isinstance(value, dict):
                            print(f"Nested dict keys: {list(value.keys())}")
                            for k, v in value.items():
                                if isinstance(v, (str, int, float, bool)):
                                    print(f"  {k}: {v}")
                        else:
                            print(f"Value: (complex type, showing summary)")
                            print(f"  {str(value)[:100]}...")
                else:
                    print(f"Sample is not a dictionary but a {type(sample)}")
                    print(f"Sample content: {str(sample)[:500]}...")
                
                # Try to manually determine the solo condition
                condition = None
                participant = None
                
                # Look for condition string
                if 'condition' in sample:
                    condition = sample['condition']
                elif any(k.endswith('condition') for k in sample.keys()):
                    for k in sample.keys():
                        if k.endswith('condition'):
                            condition = sample[k]
                            break
                
                # Look for participant number
                if 'participant_num' in sample:
                    participant = sample['participant_num']
                elif 'participant' in sample:
                    participant = sample['participant']
                
                if condition and participant:
                    print(f"\nManual solo condition check:")
                    print(f"Condition: {condition}")
                    print(f"Participant: {participant}")
                    
                    # Test all the solo condition logic paths
                    test1 = condition.startswith("T1") and not condition.startswith("T12") and not condition.startswith("T13")
                    test2 = condition.startswith("T23") and participant == "P1"
                    test3 = condition.startswith("T13") and participant == "P2"
                    test4 = condition.startswith("T12") and participant == "P3"
                    
                    print(f"T1 and not T12/T13: {test1}")
                    print(f"T23 and P1: {test2}")
                    print(f"T13 and P2: {test3}")
                    print(f"T12 and P3: {test4}")
                    
                    is_solo = test1 or test2 or test3 or test4
                    print(f"Final solo determination: {is_solo}")
                
            except Exception as e:
                print(f"Error examining file: {e}")
                traceback.print_exc()
    
    analyze_all_datasets(args.data_dir)
    
    if args.plot:
        create_sample_distribution_plots(args.data_dir)