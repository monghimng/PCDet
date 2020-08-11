#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:3


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools
#source $ROOT/virtualenv/risep36_testing/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

#DEBUG=true
DEBUG=false

if [ "$DEBUG" = true ] ; then
# for debugging, with only 1 gpu, no distributed training, no dataloading thread
python \
train.py \
--cfg_file cfgs/argo/pointpillar_forward50x50.yaml \
--batch_size 2 \
--workers 0 \
--extra_tag debug_$RANDOM \
--pretrained_model /data/ck/BEVSEG/PCDet2/output/pointpillar_forward50x50/debugging_lidar_2/ckpt/checkpoint_epoch_6.pth \
--set \
DATA_CONFIG.TRAIN.SHUFFLE_POINTS False \
MODE bev \
INJECT_SEMANTICS True \
INJECT_SEMANTICS_WIDTH 1240 \
INJECT_SEMANTICS_MODE 'logit_car_mask' \
#TRAIN_SEMANTIC_NETWORK True \
#USE_PSEUDOLIDAR True \
#DATA_CONFIG.TRAIN.SHUFFLE_POINTS False \
#DATA_CONFIG.FOV_POINTS_ONLY True \
#--batch_size 2 \
#--workers 0 \


exit
fi

NAME=bev_ptswithrgb_normalized_2
NAME=bev_newvoxelization
NAME=bev_5pct_0
NAME=bev_50pct_1
NAME=bev_10pct_1
NAME=bev_1pct_1
NAME=zeroed_out
NAME=debug_$RANDOM
NAME=debugging_lidar_2
NAME=bev_semantics_lidar_0

#python \
python -m torch.distributed.launch --nproc_per_node=3 \
train.py \
--cfg_file cfgs/argo/pointpillar_forward50x50.yaml \
--extra_tag $NAME \
--launcher pytorch \
--sync_bn \
--batch_size 18 \
--tcp_port 10033 \
--set \
MODE bev \
INJECT_SEMANTICS True \
INJECT_SEMANTICS_WIDTH 1240 \
INJECT_SEMANTICS_MODE 'logit_car_mask' \
#TRAIN_SEMANTIC_NETWORK True \
#USE_PSEUDOLIDAR True \
#--pretrained_model /data/ck/BEVSEG/PCDet2/output/pointpillar_centered50x50/noposweight/ckpt/checkpoint_epoch_44.pth \
#--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
#--epochs 200 \
#--batch_size 64 \
#MODEL.TRAIN.OPTIMIZATION.LR 0.0001 \


<< notes
batch sizes:
4 32gb 64
2 32gb 32
+resnet bs:
1 32gb = 16

notes

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_train.sh
sample_cmds
