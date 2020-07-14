#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:2


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

#DEBUG=true
DEBUG=false

if [ "$DEBUG" = true ] ; then
# for debugging, with only 1 gpu, no distributed training, no dataloading thread
python \
train.py \
--cfg_file cfgs/argo/pointpillar_centered50x50.yaml \
--batch_size 2 \
--extra_tag debug_$RANDOM \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
--workers 0 \
--pretrained_model /data/ck/BEVSEG/PCDet2/output/pointpillar_centered50x50/noposweight/ckpt/checkpoint_epoch_44.pth \

exit
fi

NAME=argo_ptpillar_centered_adam50x50_9
NAME=bev_lmbda0.001_5
NAME=bev_lrsteps_6
NAME=bev_lrsteps_halvedposweights_7
NAME=bev_lrsteps_halvedposweights_lr0.03_7
NAME=bev_vehicle_only_9
NAME=bev_vehicle_only_halfposweight_9
NAME=focal_9
NAME=bev_l1loss_10
NAME=bev_moreblocks_11
NAME=bev_moreblocks_oldloss_12
NAME=bev_tagbboxpoints_13
NAME=bev_projected_taggedpoints_17
NAME=bev_tagpts_usingresenet18_18
NAME=bev_ptpillar_usingresenet18_20
NAME=bev_ptpillar_usingresenet18_21
NAME=1pt5weight
NAME=frozen_pcdet_layers_usedpretrained_2
NAME=bilinear_interp

#python \
python -m torch.distributed.launch --nproc_per_node=2 \
train.py \
--cfg_file cfgs/argo/pointpillar_centered50x50.yaml \
--extra_tag $NAME \
--launcher pytorch \
--sync_bn \
--batch_size 20 \
--tcp_port 10005 \
--pretrained_model /data/ck/BEVSEG/PCDet2/output/pointpillar_centered50x50/noposweight/ckpt/checkpoint_epoch_44.pth \
--set \
MODEL.TRAIN.OPTIMIZATION.OPTIMIZER adam \
MODEL.TRAIN.OPTIMIZATION.DECAY_STEP_LIST '[10, 20]' \
MODEL.TRAIN.OPTIMIZATION.LR 0.0008 \
MODEL.TRAIN.OPTIMIZATION.LR_WARMUP False \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
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
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
sample_cmds
