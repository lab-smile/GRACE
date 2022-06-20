#!/bin/bash
#SBATCH --job-name=monai
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=skylastolte4444@ufl.edu     	
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpg-ai
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60gb                     
#SBATCH --time=72:00:00          
#SBATCH --output=%x_%j.log 

#module load pytorch
module load singularity
#module load monai

# Check if cuda enabled
singularity exec --nv /blue/ruogu.fang/skylastolte4444/monaicore081 python3 -c "import torch; print(torch.cuda.is_available())"

#run code
#singularity exec --nv /red/nvidia-ai/SkylarStolte/monaicore08 python3 /red/nvidia-ai/SkylarStolte/monai_train.py --num_gpu 3
#python3 /red/nvidia-ai/SkylarStolte/monai_train.py --num_gpu 3 --a_min_value 0 --a_max_value 255

singularity exec --nv --bind /blue/ruogu.fang/skylastolte4444/hackathon:/mnt /blue/ruogu.fang/skylastolte4444/monaicore081 python3 /mnt/monai_train.py --num_gpu 1 --data_dir '/mnt/training_pairs_v5/' --model_save_name "unetr_v5_06-19-22" --N_classes 12 --max_iteration 100



#module load pytorch
#module load monai

#python3 monai_train.py --num_gpu 1 --data_dir '/blue/ruogu.fang/skylastolte4444/hackathon/training_pairs_v5/' --model_save_name "unetr_v5_06-19-22" --N_classes 12 --max_iteration 100