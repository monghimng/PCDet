#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:4


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_parta2_2

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
#python \
python -m torch.distributed.launch --nproc_per_node=4 \
train.py \
--cfg_file cfgs/argo/PartA2.yaml \
--batch_size 32 \
--extra_tag $NAME \
--launcher pytorch \
--sync_bn \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
sample_cmds

