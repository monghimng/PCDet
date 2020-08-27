#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:1


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=sord_nosemantics_0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
python \
test.py \
--cfg_file cfgs/argo/pointpillar_forward50x50.yaml \
--batch_size 18 \
--extra_tag $NAME \
--workers 4 \
--eval_all \
--set \
MODE bev \
INJECT_SEMANTICS False \
INJECT_SEMANTICS_WIDTH 1250 \
INJECT_SEMANTICS_MODE 'binary_car_mask' \
USE_PSEUDOLIDAR True \
TORCH_VOXEL_GENERATOR True \
SPARSIFY_PL_PTS False \
#--ckpt /data/ck/BEVSEG/PCDet2/output/pointpillar_forward50x50/sord_0/ckpt/checkpoint_epoch_18.pth \
#INJECT_SEMANTICS True \
#INJECT_SEMANTICS_WIDTH 1240 \
#INJECT_SEMANTICS_MODE 'logit_car_mask' \
#TRAIN_SEMANTIC_NETWORK True \
#USE_PSEUDOLIDAR True \
#--pretrained_model /data/ck/BEVSEG/PCDet2/output/pointpillar_centered50x50/noposweight/ckpt/checkpoint_epoch_44.pth \
#--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
#--epochs 200 \
#--batch_size 64 \
#MODEL.TRAIN.OPTIMIZATION.LR 0.0001 \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
sample_cmds

