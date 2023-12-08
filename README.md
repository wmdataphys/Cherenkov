# Cherenkov


Base repository for training Cherenkov models.

The current model is an abstract representation of a Convolutional deepRICH. We will improve this.

First you will need to replicate my conda environment on sciclone. Follow these steps:

1. Take out an interactive job on HIMA -> qsub -I -l nodes=1:hima:gpu:ppn=20,walltime=01:00:00
2. Load the modules: (1) module load anaconda3/2021.11 (2) module load cuda/11.7
3. Replicate my environment -> conda env create -f pytorch_env.yaml
4. Test its working with: conda activate ptorch

Everything should be configuration file based -> config/default_config.json
You will need to adjust your paths in here. The data files are located on the HPC and I can point you to them once you are there. We will need to get them copied over to your data directory.
I also reccomend that you link your models to fields within the config file if possible so that it is easy to replicate. Obviously when things move fast this isn't the number one priority but it is good practice.

You can test the code with an interactive job with a command like this:

1. Take out an interactive job on HIMA:  qsub -I -l nodes=1:hima:gpu:ppn=20,walltime=01:00:00
2. module load anaconda3/2021.11
3. module load cuda/11.7
4. conda activate ptorch
5. python train_mnf.py --config config/default_config.json

This will create a new directory (name will be experiment field in config file) in your data directory once you configure the file paths correctly. The config file will be copied into the folder, and the model weights will
be saved after each epoch of training. It will throw an error if the folder exists to prevent overwriting.

You can also train the model via the torque submission script I have provided. You will also need to modify the paths here to point to your directories (i.e. jgiroux -> your username )
This will submit a batch job to torque on HIMA, and you will get one of the 16GB GPUs on the system. We can modify this once the Kubernetes cluster is operational to deploy in a pod.

To submit the batch job: qsub submit_torque

You can monitor your job status with: qstat -u your_username

Let me know if there are further questions and when you want to start getting data copied over.
