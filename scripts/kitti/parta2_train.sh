#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:2


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

# name for the experiments
#NAME=parta2_lidar_1
#NAME=parta2_lidar_noaug_0
#NAME=parta2_lidar_pl2_0
NAME=parta2_mono_0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
python -m torch.distributed.launch --nproc_per_node=2 \
train.py \
--cfg_file cfgs/PartA2_car.yaml \
--sync_bn \
--launcher pytorch \
--batch_size 16 \
--extra_tag $NAME \
--tcp_port 11112 \
--set \
DATA_CONFIG.AUGMENTATION.NOISE_PER_OBJECT.ENABLED False \
DATA_CONFIG.AUGMENTATION.NOISE_GLOBAL_SCENE.ENABLED False \
DATA_CONFIG.AUGMENTATION.DB_SAMPLER.ENABLED False \
ALTERNATE_PT_CLOUD_ABS_DIR '/data/ck/data/kitti_obj_bts_pred/result_bts_eigen_v2_pytorch_densenet161/training/plidar_sparsified' \
#ALTERNATE_PT_CLOUD_ABS_DIR '/data/ck/data/kitti_pl2/sdn_kitti_train_set_sparse' \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
cksbatch --nodelist=flaminio ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
bash ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
sample_cmds

