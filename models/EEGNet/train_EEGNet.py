# 
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dataset import load_eeg_data
from models.EEGNet.EEGModels import EEGNet

# Paths
TRAIN_DIR = "/work3/s224188/LaBraM_data/train"
TEST_DIR = "/work3/s224188/LaBraM_data/test"

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
INPUT_SHAPE = (64, 800, 1)  # Adjust to match EEGNet config

# Load data
x_train, y_train = load_eeg_data(TRAIN_DIR)
x_test, y_test = load_eeg_data(TEST_DIR)

# Build model
model = EEGNet(nb_classes=2, Chans=64, Samples=800, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
checkpoint_cb = ModelCheckpoint("best_eegnet_tf.h5", save_best_only=True, monitor="val_accuracy", mode="max")
earlystop_cb = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

# Training
model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")