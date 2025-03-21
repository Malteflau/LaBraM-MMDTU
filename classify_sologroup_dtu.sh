#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J LaBraM_DTU_solovsgroup_50epoch
# -- choose queue --
#BSUB -q gpuv100
# -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=2GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends --
#BSUB -N
# -- email address -- 
#BSUB -u s224183@dtu.dk
# -- Output File --
#BSUB -o ./log/finetune_dtu_base/solovsgroup/LaBraM_DTU_solovsgroup_%J.out
# -- Error File --
#BSUB -e ./log/finetune_dtu_base/solovsgroup/LaBraM_DTU_solovsgroup_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00
# -- Number of cores requested -- 
#BSUB -n 4
# -- GPU requirements --
#BSUB -gpu "num=1:mode=exclusive_process"
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- end of LSF options -- 

# Create log directories if they don't exist
mkdir -p ./log/finetune_dtu_base/solovsgroup
mkdir -p ./checkpoints/finetune_dtu_base/solovsgroup

# Export unlimited file size for core dumps and stack traces
ulimit -c unlimited
ulimit -s unlimited

# Turn off output buffering
export PYTHONUNBUFFERED=1

# Use conda in batch mode - this is critical
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

# Run the training with full output logging
python run_class_finetuning.py \
    --output_dir ./checkpoints/finetune_dtu_base/solovsgroup \
    --log_dir ./log/finetune_dtu_base/solovsgroup \
    --model labram_base_patch200_200 \
    --finetune /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/finetune_dtu_labram1/checkpoint-49.pth \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs 50 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset DTU \
    --disable_qkv_bias \
    --seed 0