#!/bin/bash

#module load pytorch
module load singularity
#module load monai

# Check if cuda enabled
singularity exec --nv /path/to/monai/container/monaicore081 python3 -c "import torch; print(torch.cuda.is_available())"

#run code
#singularity exec --nv /path/to/monai/container/monaicore08 python3 /path/to/working/directory/train.py --num_gpu 3
#python3 /path/to/working/directory/train.py --num_gpu 3 --a_min_value 0 --a_max_value 255 --N_classes 12 --max_iteration 1000

singularity exec --nv --bind /path/to/working/directory:/mnt /path/to/monai/container/monaicore081 python3 /mnt/train.py --num_gpu 1 --data_dir '/mnt/data_folder/' --model_save_name "grace" --N_classes 12 --max_iteration 1000



