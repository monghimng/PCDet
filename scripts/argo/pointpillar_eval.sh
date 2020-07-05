#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:1


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_ptpillar_centered_adam50x50_9

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
python \
test.py \
--cfg_file cfgs/argo/pointpillar_centered50x50.yaml \
--batch_size 10 \
--extra_tag $NAME \
--eval_all \
#--ckpt /data/ck/BEVSEG/PCDet2/output/pointpillar_centered/$NAME/ckpt/checkpoint_epoch_28.pth \
#--set \
#DATA_CONFIG.TEST.INFO_PATH '["data/argo/kitti_infos_train.pkl"]' \
#--save_to_file \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
sample_cmds

