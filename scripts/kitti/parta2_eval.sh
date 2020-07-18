#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:1


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

# name for the experiments
NAME=parta2_lidar_1
NAME=parta2_lidar_noaug_0
#NAME=parta2_lidar_pl2_0

CKPT=6

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
python \
test.py \
--cfg_file cfgs/PartA2_car.yaml \
--batch_size 16 \
--extra_tag ${NAME} \
--eval_all \
--set \
DATA_CONFIG.AUGMENTATION.NOISE_PER_OBJECT.ENABLED False \
DATA_CONFIG.AUGMENTATION.NOISE_GLOBAL_SCENE.ENABLED False \
DATA_CONFIG.AUGMENTATION.DB_SAMPLER.ENABLED False \
#ALTERNATE_PT_CLOUD_ABS_DIR '/data/ck/data/kitti_pl2/sdn_kitti_train_set_sparse' \
#--ckpt /data/ck/BEVSEG/PCDet2/output/PartA2_car/$NAME/ckpt/checkpoint_epoch_$CKPT.pth \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/kitti/parta2_eval.sh
bash ~/BEVSEG/PCDet2/scripts/kitti/parta2_eval.sh
sample_cmds

