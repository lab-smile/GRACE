#!/bin/bash

date;hostname;pwd

module load singularity

# build a Singularity sandbox container (container in a writable directory) from MONAI Core docker image
singularity build --sandbox /red/nvidia-ai/SkylarStolte/monaicore08/ docker://projectmonai/monai:0.8.1
# check nsys environment
singularity exec --nv /red/nvidia-ai/SkylarStolte/monaicore08 nsys status -e