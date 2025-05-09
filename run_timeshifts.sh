#!/bin/bash
#BSUB -J LaBraM_DTU_betaband_models
#BSUB -q gpuv100
#BSUB -R "rusage[mem=3GB]"
#BSUB -B
#BSUB -N
#BSUB -u s224183@dtu.dk
#BSUB -o ./log/finetune_dtu_base/betaband_models/LaBraM_DTU_betaband_models_%J.out
#BSUB -e ./log/finetune_dtu_base/betaband_models/LaBraM_DTU_betaband_models_%J.err
#BSUB -W 24:00
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

# Create log directory
mkdir -p ./log/finetune_dtu_base/betaband_models

# Define model paths
BASE_MODEL="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/labram-base.pth"
HYBRID_MODEL="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/finetune_original_vqnsp/checkpoint.pth"

# Define conditions

#CONDITIONS=("sologroup" "friendship" "feedback" "gender")
CONDITIONS=("friendship")

# Set resource limits
ulimit -c unlimited
ulimit -s unlimited

# Turn off buffering
export PYTHONUNBUFFERED=1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

# Training function
run_training() {
    local model_path=$1
    local model_name=$2
    local condition=$3

    echo "Running training for ${condition} with model ${model_name}..."

    local output_dir="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/${model_name}/${condition}ts=30_normal_seed2"
    mkdir -p "${output_dir}"
    mkdir -p "./log/finetune_dtu_base/betaband_models/${model_name}/${condition}"

    python run_class_finetuning.py \
        --output_dir "${output_dir}" \
        --log_dir "./log/finetune_dtu_base/betaband_models/${model_name}/${condition}" \
        --model labram_base_patch200_200 \
        --finetune "${model_path}" \
        --weight_decay 0.05 \
        --batch_size 64 \
        --lr 5e-5 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 100 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset DTU \
        --disable_qkv_bias \
        --seed 2 \
        --condition "${condition}"
}

# Run training for base model
echo "Starting training with base model..."
for condition in "${CONDITIONS[@]}"; do
    run_training "${BASE_MODEL}" "base_model_time_shifts" "${condition}"
done

# # Run training for hybrid model
# echo "Starting training with hybrid model..."
# for condition in "${CONDITIONS[@]}"; do
#     run_training "${HYBRID_MODEL}" "hybrid_model_time_shifts" "${condition}"
# done

#echo "All training runs completed."
