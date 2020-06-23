#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:7


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_4

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
#python \
python -m torch.distributed.launch --nproc_per_node=7 \
train.py \
--cfg_file cfgs/argo/pointpillar.yaml \
--batch_size 119 \
--extra_tag $NAME \
--launcher pytorch \
--sync_bn \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
sample_cmds

