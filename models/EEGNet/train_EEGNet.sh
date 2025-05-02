#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name -- 
#BSUB -J train_eegnet_svsg
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 9GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s224188@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_svsg_%J.out 
#BSUB -e Output_svsg_%J.err 

module load cuda/11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# Make conda available (add it to PATH)
export PATH=/zhome/62/3/187432/anaconda3/bin:$PATH
# Source the Conda shell hook manually
source /zhome/62/3/187432/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate EEGNet

# Run your script
python models/EEGNet/train_EEGNet.py