#!/bin/bash
# Run LaBraM pre-training fine-tuning on your social interaction EEG data

# Path to your processed EEG data in HDF5 format
H5_PATH="./DataProcessed/eeg_data.h5"

# Paths to pre-trained model and tokenizer
MODEL_PATH="/Users/maltelau/Desktop/LaBraM-MMDTU/LaBraM-MMDTU/checkpoints/labram-base.pth"
TOKENIZER_PATH="/Users/maltelau/Desktop/LaBraM-MMDTU/LaBraM-MMDTU/checkpoints/vqnsp.pth"

# Output directory
OUTPUT_DIR="./pretrain_output"
LOG_DIR="./pretrain_logs"

# Create directories
mkdir -p $OUTPUT_DIR $LOG_DIR

# Run the pre-training script
python labram_pretrain_finetuning.py \
  --h5-path $H5_PATH \
  --output-dir $OUTPUT_DIR \
  --log-dir $LOG_DIR \
  --model labram_base_patch200_1600_8k_vocab \
  --model-path $MODEL_PATH \
  --tokenizer-model vqnsp_encoder_base_decoder_3x200x12 \
  --tokenizer-weight $TOKENIZER_PATH \
  --batch-size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --warmup-epochs 5 \
  --clip-grad 3.0 \
  --drop-path 0.1 \
  --layer-scale-init-value 0.1 \
  --resample-hz 200 \
  --num-workers 4 \
  --seed 42
