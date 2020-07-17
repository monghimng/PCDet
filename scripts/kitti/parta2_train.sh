#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:2


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=parta2_lidar_0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
python -m torch.distributed.launch --nproc_per_node=2 \
train.py \
--cfg_file cfgs/PartA2_car.yaml \
--sync_bn \
--launcher pytorch \
--batch_size 20 \
--extra_tag $NAME \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
bash ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
sample_cmds

