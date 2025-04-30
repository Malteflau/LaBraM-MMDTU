# 
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.EEGNet.dataloader_EEGNet import load_eeg_data_from_pkl
from models.EEGNet.EEGModels import EEGNet
import matplotlib.pyplot as plt

# Paths
TRAIN_DIR = "/work3/s224188/LaBraM_data/train"
TEST_DIR = "/work3/s224188/LaBraM_data/test"
CKPT_DIR = "ckpts"

# Create checkpoint directory if it doesn't exist
os.makedirs(CKPT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
INPUT_SHAPE = (64, 800, 1)  # Adjust to match EEGNet config

# Conditions
conditions = ('T1P', 'T1Pn')

# Load data
x_train, y_train = load_eeg_data_from_pkl(TRAIN_DIR, conditions)
x_test, y_test = load_eeg_data_from_pkl(TEST_DIR, conditions)

# Build model
model = EEGNet(nb_classes=2, Chans=64, Samples=800, dropoutRate=0.5, kernLength=100, F1=8, D=2, F2=16, norm_rate=0.25)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
numParams = model.count_params()

# Callbacks
checkpoint_cb = ModelCheckpoint("eegnet_ckpt.h5", save_best_only=True, monitor="val_accuracy", mode="max")
earlystop_cb = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

# Training
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
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
plt.savefig("training_history.png")
plt.show()