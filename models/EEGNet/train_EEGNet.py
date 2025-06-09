# 
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dataloader_EEGNet import load_eeg_data_from_pkl
from EEGModels import EEGNet
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.dpi'] = 400
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2(np.linspace(0, 1, 8)))

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
LEARNING_RATE = 0.001
INPUT_SHAPE = (64, 800, 1)  # Adjust to match EEGNet config


fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax = ax.flatten()
test_acc = {}
test_loss = {}
# Load data
for i, label in enumerate(['feedback', 'friendship', 'gender', 'sologroup']):
    print(f'Starting training for {label} prediction')
    x_train, y_train = load_eeg_data_from_pkl(TRAIN_DIR, label_mode=label)
    x_test, y_test   = load_eeg_data_from_pkl(TEST_DIR, label_mode=label)
    x_val, y_val     = load_eeg_data_from_pkl(VAL_DIR, label_mode=label)
    print(f'Class distribution: {np.bincount(y_train)}')

    # Build model
    model = EEGNet(nb_classes=2, Chans=64, Samples=800, dropoutRate=0.5, kernLength=100, F1=8, D=2, F2=16, norm_rate=0.25)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])
    numParams = model.count_params()

    # Callbacks
    checkpoint_cb = ModelCheckpoint(os.path.join(CKPT_DIR, f"eegnet_{label}_ckpt.h5"), save_best_only=True, monitor="val_accuracy", mode="max")
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
    test_loss[label], test_acc[label] = model.evaluate(x_test, y_test)

    # Plot training history
    ax[0].plot(history.history['loss'], label=label)
    ax[1].plot(history.history['val_loss'], label=label)

    ax[2].plot(history.history['accuracy'], label=label)
    ax[3].plot(history.history['val_accuracy'], label=label)

ax[0].set_title('Training Loss'), ax[1].set_title('Validation Loss')
ax[2].set_title('Training Accuracy'), ax[3].set_title('Validation Accuracy')
ax[0].set_xlabel('Epochs'), ax[1].set_xlabel('Epochs'), ax[2].set_xlabel('Epochs'), ax[3].set_xlabel('Epochs'),
ax[0].set_ylabel('Loss'), ax[1].set_ylabel('Loss'), ax[2].set_ylabel('Accuracy'), ax[3].set_ylabel('Accuracy'),
try:
    ax[0].set_legend(), ax[1].set_legend(), ax[2].set_legend(), ax[3].set_legend()
except:
    plt.legend()
plt.suptitle('EEGNet Training history')
plt.tight_layout()
plt.savefig(f"EEGNet_training_history.png")

print('Test loss')
for (label, loss) in test_loss.items():
    print(f'{label}: {loss}')

print('\n Test Accuracy')
for (label, acc) in test_acc.items():
    print(f'{label}: {acc}')