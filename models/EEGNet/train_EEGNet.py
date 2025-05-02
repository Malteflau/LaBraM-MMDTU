# 
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dataloader_EEGNet import load_eeg_data_from_pkl
from EEGModels import EEGNet
import matplotlib.pyplot as plt
print("GPUs detected:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Built with GPU support:", tf.test.is_built_with_gpu_support())
# Paths
TRAIN_DIR = "/work3/s224188/LaBraM_data/train"
TEST_DIR = "/work3/s224188/LaBraM_data/test"
VAL_DIR = "/work3/s224188/LaBraM_data/val"
CKPT_DIR = "ckpts"

# Create checkpoint directory if it doesn't exist
os.makedirs(CKPT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
INPUT_SHAPE = (64, 800, 1)  # Adjust to match EEGNet config


# Load data
labels = 'solo_vs_group'
x_train, y_train = load_eeg_data_from_pkl(TRAIN_DIR, label_mode=labels)
x_test, y_test   = load_eeg_data_from_pkl(TEST_DIR, label_mode=labels)
x_val, y_val     = load_eeg_data_from_pkl(VAL_DIR, label_mode=labels)
print(np.bincount(y_train))

# Build model
model = EEGNet(nb_classes=2, Chans=64, Samples=800, dropoutRate=0.5, kernLength=100, F1=8, D=2, F2=16, norm_rate=0.25)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
numParams = model.count_params()

# Callbacks
checkpoint_cb = ModelCheckpoint(os.path.join(CKPT_DIR, f"eegnet_{labels}_ckpt.h5"), save_best_only=True, monitor="val_accuracy", mode="max")
earlystop_cb = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

# Training
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f"training_history_{labels}_2.png")