#!/bin/bash
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J "LaBraM_DTU_Friendship 10 epoch"
### -- ask for number of cores (default: 4) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host and GPU usage -- 
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify memory requirements -- 
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- send notification at completion -- 
#BSUB -N 
#BSUB -u s224183@dtu.dk
### -- Specify the output and error file -- 
#BSUB -oo ./log/finetune_dtu_base/friendship/LaBraM_DTU_Friendship_%J.out 
#BSUB -eo ./log/finetune_dtu_base/friendship/LaBraM_DTU_Friendship_%J.err 

# Create output directories if they don't exist
mkdir -p ./checkpoints/finetune_dtu_base/friendship
mkdir -p ./log/finetune_dtu_base/friendship

# Export unlimited file size for core dumps and stack traces
ulimit -c unlimited
ulimit -s unlimited

# Turn off output buffering
export PYTHONUNBUFFERED=1

# Run the training with full output logging
python run_class_finetuning.py \
    --output_dir ./checkpoints/finetune_dtu_base/friendship \
    --log_dir ./log/finetune_dtu_base/friendship \
    --model labram_base_patch200_200 \
    --finetune /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/labram-base.pth \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs 10 \
    --layer_decay 0.4 \
    --drop_path 0.1 \
    --save_ckpt_freq 1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset DTU \
    --disable_qkv_bias \
    --seed 0