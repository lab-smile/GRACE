#!/bin/bash

#module load pytorch
module load singularity
#module load monai

# Check if cuda enabled
#python -c "import torch; print(torch.cuda.is_available())"
singularity exec --nv /path/to/monai/container/monaicore081 python3 -c "import torch; print(torch.cuda.is_available())"

#run code
#python monai_test.py --num_gpu 2 --data_dir "/path/to/your/data/" --N_classes 12 --model_load_name "grace.pth" --dataparallel "True"

singularity exec --nv --bind /path/to/working/directory:/mnt /path/to/monai/container/monaicore081 python3 /mnt/test.py --num_gpu 2 --data_dir '/mnt/data_folder/' --model_load_name "grace.pth" --N_classes 12 --dataparallel "True" --a_max_value 255 --spatial_size 64

#singularity version assumes that data_folder is in the working directory.