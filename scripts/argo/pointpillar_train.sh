#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:6


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

DEBUG=true
#DEBUG=false

if [ "$DEBUG" = true ] ; then
# for debugging, with only 1 gpu, no distributed training, no dataloading thread
python \
train.py \
--cfg_file cfgs/argo/pointpillar_forward50x50.yaml \
--batch_size 2 \
--extra_tag debug_$RANDOM \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
--workers 0 \
--pretrained_model /data/ck/BEVSEG/PCDet2/output/pointpillar_centered50x50/noposweight/ckpt/checkpoint_epoch_44.pth \
--set \
MODE bev \
VOXELIZE_IN_MODEL_FORWARD True \
INJECT_SEMANTICS True \
DATA_CONFIG.FOV_POINTS_ONLY True \
INJECT_SEMANTICS_WIDTH 2048 \

exit
fi

NAME=debug_$RANDOM
NAME=bev_ptswithrgb_normalized_2
NAME=bev_newvoxelization
NAME=bev_5pct_0
NAME=bev_50pct_1
NAME=bev_10pct_1
NAME=bev_1pct_1
NAME=bev_semantics_1pct_0

#python \
python -m torch.distributed.launch --nproc_per_node=6 \
train.py \
--cfg_file cfgs/argo/pointpillar_forward50x50.yaml \
--extra_tag $NAME \
--launcher pytorch \
--sync_bn \
--batch_size 48 \
--tcp_port 10030 \
--set \
MODEL.TRAIN.OPTIMIZATION.OPTIMIZER adam \
MODEL.TRAIN.OPTIMIZATION.DECAY_STEP_LIST '[10, 20]' \
MODEL.TRAIN.OPTIMIZATION.LR 0.0008 \
MODEL.TRAIN.OPTIMIZATION.LR_WARMUP False \
MODE bev \
VOXELIZE_IN_MODEL_FORWARD True \
INJECT_SEMANTICS True \
DATA_CONFIG.FOV_POINTS_ONLY True \
PERCENT_OF_PTS 1 \
#VOXELIZE_IN_MODEL_FORWARD True \
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
