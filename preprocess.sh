#!/bin/bash

#SBATCH options go here

module load singularity # Optional, depending on your HPC setup

# Example command to run preprocessing
singularity exec --bind /path/to/working/directory:/mnt /path/to/monai/container/monaicore150 python3 /mnt/preprocess.py --data_dir '/mnt/data_folder/' --source-folders folder1 folder2 folder3 --verbose