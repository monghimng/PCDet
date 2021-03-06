#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:6

echo Begin!

source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NAME=argo_parta2_centered_5

### debug cmd, no distributed training, small batch size, no dataloader thread
python \
train.py \
--cfg_file cfgs/argo/PartA2_centered.yaml \
--extra_tag debug_$RANDOM \
--batch_size 2 \
--workers 0 \

exit

#python \
python -m torch.distributed.launch --nproc_per_node=6 \
train.py \
--cfg_file cfgs/argo/PartA2_centered.yaml \
--extra_tag $NAME \
--batch_size 36 \
--launcher pytorch \
--sync_bn \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/PartA2_car.pth \
--set \
MODEL.TRAIN.OPTIMIZATION.LR 0.0003 \
DATA_CONFIG.VOXEL_GENERATOR.MAX_POINTS_PER_VOXEL 7 \
#--batch_size 6 \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
cksbatch --nodelist=bombe ~/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
cksbatch --nodelist=manchester ~/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/parta2_train.sh
sample_cmds

