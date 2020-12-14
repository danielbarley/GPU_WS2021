#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -o out/%A_%a_2n
#SBATCH --array 1-32

./bin/matMul -s 1024 -t ${SLURM_ARRAY_TASK_ID}
