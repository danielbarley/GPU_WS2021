#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -o out/%A_%a
#SBATCH --array 4-14

./bin/matMul -s $(( 2 ** ${SLURM_ARRAY_TASK_ID} )) -t 32 -c --shared
