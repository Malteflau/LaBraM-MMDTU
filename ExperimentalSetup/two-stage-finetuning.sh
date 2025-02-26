#!/bin/bash
# Two-stage fine-tuning pipeline for LaBraM

# Parameters
H5_PATH="./DataProcessed/eeg_data.h5"
METADATA_PATH="./DataProcessed/metadata.pkl"
MODEL_PATH="/Users/maltelau/Desktop/LaBraM-MMDTU/LaBraM-MMDTU/checkpoints/labram-base.pth"
TOKENIZER_PATH="/Users/maltelau/Desktop/LaBraM-MMDTU/LaBraM-MMDTU/checkpoints/vqnsp.pth"
MAPPING="feedback_prediction"  # or any other mapping you created

# Stage 1: Unsupervised domain adaptation with masked EEG modeling
echo "Starting Stage 1: Domain adaptation with masked EEG modeling"
python labram_pretrain_finetuning.py \
  --h5-path $H5_PATH \
  --output-dir "./stage1_output" \
  --log-dir "./stage1_logs" \
  --model labram_base_patch200_1600_8k_vocab \
  --model-path $MODEL_PATH \
  --tokenizer-model vqnsp_encoder_base_decoder_3x200x12 \
  --tokenizer-weight $TOKENIZER_PATH \
  --batch-size 32 \
  --epochs 25 \
  --lr 5e-4 \
  --warmup-epochs 3 \
  --clip-grad 3.0 \
  --drop-path 0.1 \
  --layer-scale-init-value 0.1 \
  --resample-hz 200 \
  --num-workers 4 \
  --seed 42

# Get the path to the best model from Stage 1
STAGE1_MODEL="./stage1_output/final_model.pth"

# Stage 2: Supervised fine-tuning for the specific task
echo "Starting Stage 2: Supervised fine-tuning for classification"
python finetuning.py \
  --metadata $METADATA_PATH \
  --h5 $H5_PATH \
  --mapping $MAPPING \
  --output-dir "./stage2_output" \
  --model-path $STAGE1_MODEL \
  --model-name "labram_base_patch200_200" \
  --num-classes 2 \
  --batch-size 64 \
  --num-epochs 50 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --optimizer adamw \
  --scheduler cosine \
  --patience 10 \
  --seed 42 \
  --device cuda

echo "Two-stage fine-tuning complete!"
