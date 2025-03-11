#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J LaBraM_DTU_100epoch
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
#BSUB -o /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/log/finetune_dtu_labram/LaBraM_DTU_100epoch_%J.out
# -- Error File --
#BSUB -e /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/log/finetune_dtu_labram/LaBraM_DTU_100epoch_%J.err
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
mkdir -p /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/log/finetune_dtu_labram
mkdir -p /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/finetune_dtu_labram

# Export unlimited file size for core dumps and stack traces
ulimit -c unlimited
ulimit -s unlimited

# Turn off output buffering
export PYTHONUNBUFFERED=1

# Use conda in batch mode - this is critical
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

# Run the training with full output logging - using absolute paths
python /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/run_labram_finetuning.py \
    --output_dir /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/finetune_dtu_labram \
    --log_dir /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/log/finetune_dtu_labram \
    --model labram_base_patch200_1600_8k_vocab \
    --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
    --tokenizer_weight /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/vqnsp.pth \
    --pretrained_model /zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/labram-base.pth \
    --batch_size 64 \
    --lr 5e-5 \
    --min_lr 1e-6 \
    --warmup_epochs 10 \
    --clip_grad 3.0 \
    --drop_path 0.1 \
    --layer_scale_init_value 0.1 \
    --opt_betas 0.9 0.98 \
    --opt_eps 1e-8 \
    --weight_decay 0.05 \
    --epochs 100 \
    --save_ckpt_freq 10 \
    --codebook_dim 64 \
    --input_size 800 \
    --gradient_accumulation_steps 1 \
    --use_dtu_loader \
    --num_workers 4

