#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -o out/%A_%a
#SBATCH --array 0-5

./bin/matMul -s 8192 -t $(( 2 ** ${SLURM_ARRAY_TASK_ID} )) -c --shared
