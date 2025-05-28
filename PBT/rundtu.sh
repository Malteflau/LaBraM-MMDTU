#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J LaBraM_DTU_solovsgroup_50epoch
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

# Export unlimited file size for core dumps and stack traces
ulimit -c unlimited
ulimit -s unlimited

# Turn off output buffering
export PYTHONUNBUFFERED=1

# Use conda in batch mode - this is critical
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

export WANDB_API_KEY=yf29269453df26ab5325e4cf04d9fb5d4ef56545d

# Optional: Set project name and other configurations
export WANDB_PROJECT="Patched Brain Transformer"
export WANDB_NAME="PBT_DTU_sologroup"

# Run your script
python dtu_main.py