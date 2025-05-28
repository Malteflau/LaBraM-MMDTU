#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J LaBraM_DTU_all_models_conditions
# -- choose queue --
#BSUB -q gpua100
# -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=3GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends --
#BSUB -N
# -- email address -- 
#BSUB -u s224183@dtu.dk
# -- Output File --
#BSUB -o ./log/finetune_dtu_base/all_models/LaBraM_DTU_all_models_%J.out
# -- Error File --
#BSUB -e ./log/finetune_dtu_base/all_models/LaBraM_DTU_all_models_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00
# -- Number of cores requested -- 
#BSUB -n 4
# -- GPU requirements --
#BSUB -gpu "num=1:mode=exclusive_process"
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- end of LSF options -- t

# Create log directory if it doesn't exist
mkdir -p ./log/finetune_dtu_base/all_models

# Define model paths
MODEL1="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/finetune_original_vqnsp/checkpoint.pth"
MODEL2="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/labram-base.pth"
MODEL3="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/pretrain_dtu_labram/checkpoint.pth"
MODEL4="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/pretrain_PSpec_betaband/checkpoint.pth"

# Define conditions
CONDITIONS=("sologroup" "friendship" "feedback" "gender")

# Export unlimited file size for core dumps and stack traces
ulimit -c unlimited
ulimit -s unlimited

# Turn off output buffering
export PYTHONUNBUFFERED=1

# Use conda in batch mode - this is critical
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

# Function to run training for a given model and condition
run_training() {
    local model_path=$1
    local model_name=$2
    local condition=$3
    
    echo "Running training for ${condition} condition with model ${model_name}..."
    
    # Create output directory
    local output_dir="/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/Final_models/${model_name}/${condition}"
    mkdir -p "${output_dir}"
    mkdir -p "./log/finetune_dtu_base/all_models/${model_name}/${condition}"
    
    python run_class_finetuning.py \
        --output_dir "${output_dir}" \
        --log_dir "./log/finetune_dtu_base/all_models/${model_name}/${condition}" \
        --model labram_base_patch200_200 \
        --finetune "${model_path}" \
        --weight_decay 0.05 \
        --batch_size 64 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 50 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset DTU \
        --disable_qkv_bias \
        --seed 0 \
        --condition "${condition}"
}

# Run training for all combinations of models and conditions
# Model 1
echo "Starting training with Model 1..."
for condition in "${CONDITIONS[@]}"; do
    run_training "${MODEL1}" "Hybrid setup" "${condition}"
done

# Model 2
echo "Starting training with Model 2..."
for condition in "${CONDITIONS[@]}"; do
    run_training "${MODEL2}" "Base setup" "${condition}"
done

# Model 3
echo "Starting training with Model 3..."
for condition in "${CONDITIONS[@]}"; do
    run_training "${MODEL3}" "Scratch setup" "${condition}"
done

# Model 4
echo "Starting training with Model 4..."
for condition in "${CONDITIONS[@]}"; do
    run_training "${MODEL4}" "Beta band" "${condition}"
done

echo "All training runs completed."