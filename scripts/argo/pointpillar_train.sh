#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:2


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

## for debugging, with only 1 gpu, no distributed training, no dataloading thread
#python \
#train.py \
#--cfg_file cfgs/argo/pointpillar_centered50x50.yaml \
#--batch_size 20 \
#--extra_tag debug_$RANDOM \
#--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
#--workers 5 \
#
#exit


NAME=argo_ptpillar_centered_adam50x50_9
NAME=bev_lmbda0.001_5
NAME=bev_lrsteps_6
NAME=bev_lrsteps_halvedposweights_7
NAME=bev_lrsteps_halvedposweights_lr0.03_7

#python \
python -m torch.distributed.launch --nproc_per_node=2 \
train.py \
--cfg_file cfgs/argo/pointpillar_centered50x50.yaml \
--extra_tag $NAME \
--launcher pytorch \
--sync_bn \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
--batch_size 32 \
--tcp_port 10006 \
--set \
MODEL.TRAIN.OPTIMIZATION.OPTIMIZER adam \
MODEL.TRAIN.OPTIMIZATION.DECAY_STEP_LIST '[20, 40]' \
MODEL.TRAIN.OPTIMIZATION.LR_WARMUP True \
MODEL.TRAIN.OPTIMIZATION.LR 0.03 \
#--epochs 200 \
#--batch_size 64 \
#MODEL.TRAIN.OPTIMIZATION.LR 0.0001 \

<< notes
batch sizes:
4 32gb 64
2 32gb 32
notes

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
sample_cmds
