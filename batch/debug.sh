#!/bin/bash
#
#SBATCH --partition=debug_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10G
#SBATCH --output=results/debug/stdout.txt
#SBATCH --error=results/debug/stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=exp
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw6
##SBATCH --array=0

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

CODE_DIR=code
CONFIG_DIR=configs

## SHALLOW
python ${CODE_DIR}/main.py \
       @${CONFIG_DIR}/exp.txt \
       @${CONFIG_DIR}/net_rnn.txt \
       --exp_index 0 \
       --cpus_per_task 32 \
       --save_model --render --cache "" \
       --results_path "./results/debug/" \
       --epoch 3 \
       --steps_per_epoch 3 \
       -vvv
