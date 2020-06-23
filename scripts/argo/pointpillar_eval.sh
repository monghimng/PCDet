#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:1


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_3

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
python \
test.py \
--cfg_file cfgs/argo/pointpillar.yaml \
--batch_size 20 \
--save_to_file \
--ckpt /data/ck/BEVSEG/PCDet2/output/pointpillar/argo_2/ckpt/checkpoint_epoch_80.pth \
#--eval_all \
#--extra_tag $NAME \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
sample_cmds

