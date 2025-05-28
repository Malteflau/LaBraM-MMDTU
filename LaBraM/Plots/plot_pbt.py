import pandas as pd
import matplotlib.pyplot as plt
import os

def debug_csv_content(file_path):
    """Print CSV content for debugging"""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"\nFile: {file_path}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        return df
    else:
        print(f"File not found: {file_path}")
        return None

def create_plots(csv_paths):
    """Create 2x2 plot from CSV files"""
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Hardcoded colors for consistency
    colors = {
        'feedback': '#1f77b4',          # blue
        'feedback no time shift': '#ff7f0e',  # orange
        'friendship': '#2ca02c',        # green
        'gender': '#d62728',            # red
        'solo vs group': '#9467bd'          # purple
    }
    
    # Debug all CSV files first
    train_acc_df = debug_csv_content(csv_paths['acc_train'])
    test_acc_df = debug_csv_content(csv_paths['acc_test'])
    train_loss_df = debug_csv_content(csv_paths['train_loss'])
    test_loss_df = debug_csv_content(csv_paths['test_loss'])
    
    # Plot training accuracy (top left)
    if train_acc_df is not None:
        for condition, color in colors.items():
            # Try different column name formats
            col_options = [
                f'{condition} - train',  # Format in your example
                f'{condition} - acc_train',  # Alternative format
                f'{condition} - accuracy',  # Another possibility
                f'{condition}',  # Simple format
            ]
            
            for col in col_options:
                if col in train_acc_df.columns:
                    print(f"Found column '{col}' in training accuracy CSV")
                    axs[0, 0].plot(train_acc_df['Step'], train_acc_df[col], color=color, label=condition, linewidth=2)
                    break
    
    # Plot test accuracy (top right)
    if test_acc_df is not None:
        for condition, color in colors.items():
            # Try different column name formats
            col_options = [
                f'{condition} - test',  # Format in your example
                f'{condition} - acc_test',  # Alternative format
                f'{condition} - accuracy',  # Another possibility
                f'{condition}',  # Simple format
            ]
            
            for col in col_options:
                if col in test_acc_df.columns:
                    print(f"Found column '{col}' in test accuracy CSV")
                    axs[0, 1].plot(test_acc_df['Step'], test_acc_df[col], color=color, label=condition, linewidth=2)
                    break
    
    # Plot training loss (bottom left)
    if train_loss_df is not None:
        for condition, color in colors.items():
            col = f'{condition} - train_loss'
            if col in train_loss_df.columns:
                axs[1, 0].plot(train_loss_df['Step'], train_loss_df[col], color=color, label=condition, linewidth=2)
    
    # Plot test loss (bottom right)
    if test_loss_df is not None:
        for condition, color in colors.items():
            col = f'{condition} - test_loss'
            if col in test_loss_df.columns:
                axs[1, 1].plot(test_loss_df['Step'], test_loss_df[col], color=color, label=condition, linewidth=2)
    
    # Set titles and labels
    axs[0, 0].set_title('Training Accuracy', fontsize=16)
    axs[0, 1].set_title('Test Accuracy', fontsize=16)
    axs[1, 0].set_title('Training Loss', fontsize=16)
    axs[1, 1].set_title('Test Loss', fontsize=16)
    
    # Set appropriate y-limits for each plot based on data
    axs[0, 0].set_ylim(0.5, 1.02)  # Training accuracy
    axs[0, 1].set_ylim(0.4, 0.8)  # Test accuracy
    axs[1, 0].set_ylim(0.0, 0.75)  # Training loss
    axs[1, 1].set_ylim(0.0, 7)  # Test loss - adjusted to keep everything in frame
    
    # Add grid and labels to all subplots
    for i in range(2):
        for j in range(2):
            axs[i, j].grid(True, alpha=0.3)
            axs[i, j].set_xlabel('Steps', fontsize=14)
            
            if j == 0:  # First column
                ylabel = 'Accuracy' if i == 0 else 'Loss'
                axs[i, j].set_ylabel(ylabel, fontsize=14)
            
            # Add legend if there are lines
            if axs[i, j].get_legend_handles_labels()[0]:
                axs[i, j].legend(fontsize=10)
    
    # Add a main title
    fig.suptitle('Training and Testing Metrics Comparison', fontsize=20)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Create output directory if needed
    output_dir = os.path.dirname(csv_paths['acc_train'])
    output_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'all_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    return "Plot created successfully"

if __name__ == "__main__":
    # CSV file paths
    csv_paths = {
        'acc_train': '/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/PBT logs/acc_train.csv',
        'acc_test': '/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/PBT logs/acc_test.csv',
        'train_loss': '/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/PBT logs/train_loss.csv',
        'test_loss': '/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/PBT logs/test_loss.csv'
    }
    
    # Create the plots
    result = create_plots(csv_paths)
    print(result)