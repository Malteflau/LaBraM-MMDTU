#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J LABRAM_Training
### -- ask for number of cores --
#BSUB -n 8
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds the memory limit --
#BSUB -M 20GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
#BSUB -u s224183@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

# Load necessary modules (adjust as needed for your environment)
module load python3/3.9.11
module load cuda/11.7

# Activate your virtual environment (if using)
source /path/to/your/venv/bin/activate

# Define the base command
BASE_CMD="python ExperimentalSetup/finetuning.py \
--metadata DataProcessed/metadata.pkl \
--h5 DataProcessed/eeg_data.h5 \
--mapping feedback_prediction \
--num-epochs 50 \
--lr 1e-4 \
--optimizer adamw \
--output-dir ./output"

# Run LABRAM Base
python ExperimentalSetup/finetuning.py \
--metadata DataProcessed/metadata.pkl \
--h5 DataProcessed/eeg_data.h5 \
--mapping feedback_prediction \
--model-path checkpoints/labram-base.pth \
--model-name labram_base_patch200_200 \
--output-dir ./output/base \
--num-epochs 50 \
--lr 1e-4 \
--optimizer adamw

# Run LABRAM Larger
python ExperimentalSetup/finetuning.py \
--metadata DataProcessed/metadata.pkl \
--h5 DataProcessed/eeg_data.h5 \
--mapping feedback_prediction \
--model-path checkpoints/labram-larger.pth \
--model-name labram_larger_patch200_200 \
--output-dir ./output/larger \
--num-epochs 50 \
--lr 1e-4 \
--optimizer adamw

# Run LABRAM Largest
python ExperimentalSetup/finetuning.py \
--metadata DataProcessed/metadata.pkl \
--h5 DataProcessed/eeg_data.h5 \
--mapping feedback_prediction \
--model-path checkpoints/labram-largest.pth \
--model-name labram_largest_patch200_200 \
--output-dir ./output/largest \
--num-epochs 50 \
--lr 1e-4 \
--optimizer adamw

# Deactivate virtual environment
deactivate