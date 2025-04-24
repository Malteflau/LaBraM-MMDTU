This code is based on the LaBraM implementation. For the most part their training code is used, but in some places modifications were needed.

To train the full implementation from scratch you would need to run "run_vqnsp_training.py", then run "run_labram_pretraining.py" and finally run "run_labram_finetuning.py". 

Finetuning doesnt mask data and doesnt use the codebook produced by the vqnsp, it just takes the weights produced by pretraining.


To run the entire code here, you need to first create the data set, which takes all the files from your folder and processes them and splits them into the training and test set.
Then you can run the subsequent files. Using the pretrain is probably going to be the best option in most cases.

We run the files using bash scripts to connect to HPC and log it.