#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J VQNSP_DTU_100epoch
# -- choose queue --
#BSUB -q gpuv100
# -- specify that we need 1GB of memory per core/slot --
#BSUB -R "rusage[mem=1GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends --
#BSUB -N
# -- email address -- 
#BSUB -u s224183@dtu.dk
# -- Output File --
#BSUB -o ./log/vqnsp_finetuned_dtu/VQNSP_DTU_100epoch_%J.out
# -- Error File --
#BSUB -e ./log/vqnsp_finetuned_dtu/VQNSP_DTU_100epoch_%J.err
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
mkdir -p /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/log/finetune_dtu_vqnsp
mkdir -p /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/finetune_dtu_vqnsp

# Export unlimited file size for core dumps and stack traces
ulimit -c unlimited
ulimit -s unlimited

# Turn off output buffering
export PYTHONUNBUFFERED=1

# Use conda in batch mode - this is critical
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

# Run the training with full output logging - using absolute paths
python /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/run_vqnsp_training.py \
    --output_dir /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/finetune_dtu_vqnsp \
    --log_dir /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/log/finetune_dtu_vqnsp \
    --model vqnsp_encoder_base_decoder_3x200x12 \
    --resume /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/vqnsp.pth \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 64 \
    --batch_size 64 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4 \
    --lr 5e-5 \
    --min_lr 1e-6 \
    --warmup_epochs 5 \
    --epochs 50 \
    --save_ckpt_freq 10 \
    --input_size 800 \
    --num_workers 4 \
    --seed 42 \
    --use_dtu_loader