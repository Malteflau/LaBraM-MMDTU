import json
import matplotlib.pyplot as plt

# Path to your log file
#log_path = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/metadata_training/friendship_cls_csv/log.txt"
log_path = "/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/feedback_nopretrain/log.txt"
# Read the log file and parse JSON lines
data = []
with open(log_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                data_point = json.loads(line)
                data.append(data_point)
            except json.JSONDecodeError:
                print("Skipping malformed line:", line)

# Extract epochs and test accuracy
epochs = [entry['epoch'] for entry in data]
test_accuracy = [entry['test_accuracy'] for entry in data]

# Find the top 5 epochs with highest test accuracy
top_5_indices = sorted(range(len(test_accuracy)), key=lambda i: test_accuracy[i], reverse=True)[:5]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(epochs, test_accuracy, label="Test Accuracy", color='blue', marker='o')

# Plot formatting
plt.title("Test Accuracy on feedback condition without pretraining")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the plot in /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/result_plots
plt.savefig("/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/result_plots/feedback_test_accuracy_nopretrain.png", dpi=300)
print(f"Individual plot saved")

