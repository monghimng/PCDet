#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:0


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_parta2_centered_5

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
python \
test.py \
--cfg_file cfgs/argo/PartA2_centered.yaml \
--batch_size 14 \
--extra_tag $NAME \
--eval_all \
--set \
DATA_CONFIG.VOXEL_GENERATOR.MAX_POINTS_PER_VOXEL 7


<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/parta2_eval.sh
cksbatch --nodelist=manchester ~/BEVSEG/PCDet2/scripts/argo/parta2_eval.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/parta2_eval.sh
sample_cmds

