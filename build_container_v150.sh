#!/bin/bash

date;hostname;pwd

module load singularity

# build a Singularity sandbox container (container in a writable directory) from MONAI Core docker image
singularity build --sandbox /path/to/monai/container/monaicore150/ docker://projectmonai/monai:1.5.0

# check nsys environment
singularity exec --nv /path/to/monai/container/monaicore150 nsys status -e