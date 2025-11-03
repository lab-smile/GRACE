#!/bin/bash

#SBATCH options go here

module load singularity # Optional, depending on your HPC setup

# Check if cuda enabled
singularity exec --nv /path/to/monai/container/monaicore150 python3 -c "import torch; print(torch.cuda.is_available())"

# Example command to run training
singularity exec --nv --bind /path/to/working/directory:/mnt /path/to/monai/container/monaicore150 python3 /mnt/train.py --num_gpu 1 --data_dir '/mnt/data_folder/' --model_save_name "grace" --N_classes 12 --max_iteration 1000