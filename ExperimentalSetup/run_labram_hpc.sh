#!/bin/bash

# LaBraM Social Interaction EEG Processing Pipeline for HPC
# Authors: Magnus Evensen, Malte Færgemann Lau
# Project: Bachelor's Project - EEG Social Interaction with LaBraM

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU"
WORK_DIR="${BASE_DIR}/ExperimentalSetup"
LOG_DIR="${WORK_DIR}/logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/pipeline.log"

# Configuration
DATA_DIR="${BASE_DIR}/DTUDATA/FG_Data"
EEG_DIR="${DATA_DIR}/PreprocessedEEGData"
OVERVIEW_PATH="${DATA_DIR}/FG_overview_df_v2.pkl"
BEHAVIOR_PATH="${DATA_DIR}/Beh_feat_df_v2.pkl"
PROCESSED_DIR="${WORK_DIR}/DataProcessed"
PRETRAIN_DIR="${WORK_DIR}/pretrain_output"
OUTPUT_DIR="${WORK_DIR}/output_${TIMESTAMP}"

# LaBraM model paths
LABRAM_PATH="${BASE_DIR}/checkpoints/labram-base.pth"
TOKENIZER_PATH="${BASE_DIR}/checkpoints/vqnsp.pth"

# Training config
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=50
LEARNING_RATE=1e-4
PATIENCE=10
SEED=42
RESAMPLE_HZ=200
PATCH_SIZE=200

# Function for logging
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "$message"
    echo "$message" >> "$MAIN_LOG"
}

# Function to handle errors
handle_error() {
    log "ERROR: $1"
    log "Pipeline failed at stage: $2"
    exit 1
}

# Change to working directory
cd "$WORK_DIR" || handle_error "Cannot change to working directory: $WORK_DIR" "setup"

# Create output directories
mkdir -p "$PROCESSED_DIR" "$OUTPUT_DIR" "$PRETRAIN_DIR"

log "Starting LaBraM Social Interaction EEG Processing Pipeline on HPC"
log "Configuration:"
log "  Base directory: $BASE_DIR"
log "  Working directory: $WORK_DIR"
log "  Data directory: $DATA_DIR"
log "  EEG directory: $EEG_DIR"
log "  Processed data directory: $PROCESSED_DIR"
log "  Output directory: $OUTPUT_DIR"
log "  LaBraM model path: $LABRAM_PATH"
log "  Tokenizer path: $TOKENIZER_PATH"

# Check if paths exist
if [ ! -d "$EEG_DIR" ]; then
    handle_error "EEG directory does not exist: $EEG_DIR" "path_check"
fi

if [ ! -f "$OVERVIEW_PATH" ]; then
    handle_error "Overview file does not exist: $OVERVIEW_PATH" "path_check"
fi

if [ ! -f "$BEHAVIOR_PATH" ]; then
    handle_error "Behavior file does not exist: $BEHAVIOR_PATH" "path_check"
fi

if [ ! -f "$LABRAM_PATH" ]; then
    handle_error "LaBraM model file does not exist: $LABRAM_PATH" "path_check"
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    handle_error "Tokenizer file does not exist: $TOKENIZER_PATH" "path_check"
fi

# Step 1: Data Preparation
log "Step 1: Efficient Data Preparation"
python efficient_data_preparation.py \
    --eeg-folder-path "$EEG_DIR" \
    --overview-path "$OVERVIEW_PATH" \
    --behavior-path "$BEHAVIOR_PATH" \
    --output-dir "$PROCESSED_DIR" \
    2>&1 | tee "${LOG_DIR}/data_preparation.log" || handle_error "Data preparation failed" "data_preparation"

log "Data preparation complete. Files created:"
log "  $(ls -la ${PROCESSED_DIR})"

# Step 2: Test DataLoader
log "Step 2: Testing DataLoader"
python dataloader.py \
    --metadata "${PROCESSED_DIR}/metadata.pkl" \
    --h5 "${PROCESSED_DIR}/eeg_data.h5" \
    --mapping "feedback_prediction" \
    --batch-size 8 \
    --all-patches \
    2>&1 | tee "${LOG_DIR}/dataloader_test.log" || handle_error "DataLoader test failed" "dataloader_test"

# Step 3: Domain adaptation pre-training (optional)
# On HPC, we'll use environment variable to control this instead of interactive prompt
if [[ "${RUN_PRETRAINING:-no}" == "yes" ]]; then
    log "Step 3: Domain Adaptation Pre-training"
    python labrampre.py \
        --h5-path "${PROCESSED_DIR}/eeg_data.h5" \
        --output-dir "$PRETRAIN_DIR" \
        --log-dir "${LOG_DIR}/pretrain_tensorboard" \
        --model "labram_base_patch200_1600_8k_vocab" \
        --model-path "$LABRAM_PATH" \
        --tokenizer-model "vqnsp_encoder_base_decoder_3x200x12" \
        --tokenizer-weight "$TOKENIZER_PATH" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --resample-hz "$RESAMPLE_HZ" \
        --epochs 20 \
        --lr 5e-5 \
        --seed "$SEED" \
        2>&1 | tee "${LOG_DIR}/pretraining.log" || handle_error "Pre-training failed" "pretraining"
    
    # Update model path to use pre-trained model
    PRETRAINED_MODEL="${PRETRAIN_DIR}/final_model.pth"
    log "Domain adaptation complete. Using pre-trained model: $PRETRAINED_MODEL"
else
    log "Skipping domain adaptation pre-training"
    PRETRAINED_MODEL="$LABRAM_PATH"
fi

# Step 4: Supervised fine-tuning
log "Step 4: Supervised Fine-tuning"

# Use environment variable to set mapping type or default to feedback_prediction
MAPPING="${MAPPING_TYPE:-feedback_prediction}"
log "Using mapping: $MAPPING"

# Get number of classes based on mapping
NUM_CLASSES=2
if [[ "$MAPPING" == "participant_position" ]]; then
    NUM_CLASSES=3
fi

log "Fine-tuning with $NUM_CLASSES classes"

# Run fine-tuning
python finetuning.py \
    --metadata "${PROCESSED_DIR}/metadata.pkl" \
    --h5 "${PROCESSED_DIR}/eeg_data.h5" \
    --mapping "$MAPPING" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$PRETRAINED_MODEL" \
    --model-name "labram_base_patch200_200" \
    --num-classes "$NUM_CLASSES" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --resample-hz "$RESAMPLE_HZ" \
    --patch-size "$PATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --patience "$PATIENCE" \
    --seed "$SEED" \
    --optimizer "adamw" \
    --scheduler "cosine" \
    2>&1 | tee "${LOG_DIR}/finetuning.log" || handle_error "Fine-tuning failed" "finetuning"

# Final summary
log "Pipeline completed successfully!"
log "Results are available in: $OUTPUT_DIR"
log "Training history: ${OUTPUT_DIR}/training_history.pkl"
log "Best model: ${OUTPUT_DIR}/checkpoints/best_model.pt"
log "Logs: $LOG_DIR"

# Save configuration summary to file
{
    echo "LaBraM Social Interaction EEG Pipeline Configuration"
    echo "Timestamp: $(date)"
    echo "Mapping type: $MAPPING"
    echo "Batch size: $BATCH_SIZE"
    echo "Learning rate: $LEARNING_RATE"
    echo "Number of epochs: $NUM_EPOCHS"
    echo "Number of classes: $NUM_CLASSES"
    echo "Patience: $PATIENCE"
    echo "Seed: $SEED"
    echo "Domain adaptation: ${RUN_PRETRAINING:-no}"
    echo "Model path: $PRETRAINED_MODEL"
} > "${OUTPUT_DIR}/configuration.txt"

log "Configuration summary saved to: ${OUTPUT_DIR}/configuration.txt"