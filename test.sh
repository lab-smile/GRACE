#!/bin/bash

#module load pytorch
module load singularity
#module load monai

# Check if cuda enabled
#python -c "import torch; print(torch.cuda.is_available())"
singularity exec --nv /blue/ruogu.fang/skylastolte4444/monaicore081 python3 -c "import torch; print(torch.cuda.is_available())"

#run code
#python monai_test.py --num_gpu 2 --data_dir "/red/nvidia-ai/SkylarStolte/training_pairs_v5_bfc/" --N_classes 12 --model_load_name "unetr_v5_bfc.pth" --dataparallel "True"

singularity exec --nv --bind /blue/ruogu.fang/skylastolte4444/hackathon:/mnt /blue/ruogu.fang/skylastolte4444/monaicore081 python3 /mnt/monai_test.py --num_gpu 2 --data_dir '/mnt/training_pairs_v5/' --model_load_name "unetr_v5_06-19-22.pth" --N_classes 12 --dataparallel "True" --a_max_value 255 --spatial_size 64