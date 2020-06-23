#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:6

echo Begin!

source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_parta2_4

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
#python \
python -m torch.distributed.launch --nproc_per_node=6 \
train.py \
--cfg_file cfgs/argo/PartA2.yaml \
--extra_tag $NAME \
--batch_size 42 \
--launcher pytorch \
--sync_bn \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/PartA2_car.pth
#--batch_size 6 \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
cksbatch --nodelist=bombe ~/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
cksbatch --nodelist=manchester ~/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
sample_cmds

