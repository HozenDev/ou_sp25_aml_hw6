#!/bin/bash
#
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=results/exp_mha/%j_stdout.txt
#SBATCH --error=results/exp_mha/%j_stderr.txt
#SBATCH --time=06:00:00
#SBATCH --job-name=exp_mha
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw6
#SBATCH --array=0-4

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

CODE_DIR=code
CONFIG_DIR=configs

## SHALLOW
python ${CODE_DIR}/main.py \
       @${CONFIG_DIR}/exp.txt \
       @${CONFIG_DIR}/net_mha.txt \
       --exp_index $SLURM_ARRAY_TASK_ID \
       --cpus_per_task $SLURM_CPUS_PER_TASK \
       --save_model --render \
       --results_path "./results/exp_mha/" \
       -vvv
