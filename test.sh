#!/bin/bash

#SBATCH options go here

module load singularity # Optional, depending on your HPC setup

# Check if cuda enabled
singularity exec --nv /path/to/monai/container/monaicore150 python3 -c "import torch; print(torch.cuda.is_available())"

# Example command to run testing
singularity exec --nv --bind /path/to/working/directory:/mnt /path/to/monai/container/monaicore081 python3 /mnt/test.py --num_gpu 2 --data_dir '/mnt/data_folder/' --model_load_name "grace.pth" --N_classes 12 --dataparallel "True" --a_max_value 255 --spatial_size 64