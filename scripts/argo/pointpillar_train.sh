#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:4


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_ptpillar_centered_adam_8

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# for debugging, with only 1 gpu, no distributed training, no dataloading thread
python \
train.py \
--cfg_file cfgs/argo/pointpillar_centered.yaml \
--batch_size 2 \
--extra_tag debug_$RANDOM \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
--workers 0 \

exit

#python \
python -m torch.distributed.launch --nproc_per_node=4 \
train.py \
--cfg_file cfgs/argo/pointpillar_centered.yaml \
--batch_size 64 \
--extra_tag $NAME \
--launcher pytorch \
--sync_bn \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
--epochs 200 \
--set \
MODEL.TRAIN.OPTIMIZATION.LR 0.0003 \
MODEL.TRAIN.OPTIMIZATION.OPTIMIZER adam \
MODEL.TRAIN.OPTIMIZATION.LR 0.0001 \
MODEL.TRAIN.OPTIMIZATION.DECAY_STEP_LIST '[30, 60, 100]' \

<< notes

notes

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
sample_cmds

