#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:3


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

DEBUG=true
#DEBUG=false

if [ "$DEBUG" = true ] ; then
# for debugging, with only 1 gpu, no distributed training, no dataloading thread
#python -m pdb \
python \
train.py \
--cfg_file cfgs/PartA2_car.yaml \
--batch_size 2 \
--extra_tag debug_$RANDOM \
--pretrained_model /home/eecs/monghim.ng/BEVSEG/PCDet2/pointpillar.pth \
--workers 0 \
--set \
INJECT_SEMANTICS True \
INJECT_SEMANTICS_HEIGHT 375 \
INJECT_SEMANTICS_WIDTH 1240 \
DATA_CONFIG.FOV_POINTS_ONLY True \
INJECT_SEMANTICS_MODE 'logit_car_mask' \
DATA_CONFIG.AUGMENTATION.NOISE_PER_OBJECT.ENABLED False \
DATA_CONFIG.AUGMENTATION.NOISE_GLOBAL_SCENE.ENABLED False \
DATA_CONFIG.AUGMENTATION.DB_SAMPLER.ENABLED False \
ALTERNATE_PT_CLOUD_ABS_DIR '/data/ck/data/kitti_obj_bts_pred/result_bts_eigen_v2_pytorch_densenet161/training/plidar_sparsified' \
#TRAIN_SEMANTIC_NETWORK True \
#ALTERNATE_PT_CLOUD_ABS_DIR '/data/ck/data/kitti_pl2/sdn_kitti_train_set_sparse' \
#TAG_PTS_IF_IN_GT_BBOXES True \

exit
fi

# name for the experiments
#NAME=parta2_lidar_1
#NAME=parta2_lidar_noaug_0
#NAME=parta2_mono_0
#NAME=parta2_lidar_pl2_gttaged_0
#NAME=parta2_mono_gttagged_0
#NAME=parta2_lidar_semantic_injection_0
#NAME=parta2_mono_semantic_injection_0
NAME=carprob
NAME=parta2_pl2_7
NAME=parta2_pl2_semantic_injection_3

python -m torch.distributed.launch --nproc_per_node=3 \
train.py \
--cfg_file cfgs/PartA2_car.yaml \
--sync_bn \
--launcher pytorch \
--batch_size 21 \
--extra_tag $NAME \
--tcp_port 11106 \
--epochs 200 \
--set \
DATA_CONFIG.FOV_POINTS_ONLY True \
DATA_CONFIG.AUGMENTATION.NOISE_PER_OBJECT.ENABLED False \
DATA_CONFIG.AUGMENTATION.NOISE_GLOBAL_SCENE.ENABLED False \
DATA_CONFIG.AUGMENTATION.DB_SAMPLER.ENABLED False \
ALTERNATE_PT_CLOUD_ABS_DIR '/data/ck/data/kitti_pl2/sdn_kitti_train_set_sparse' \
INJECT_SEMANTICS True \
INJECT_SEMANTICS_HEIGHT 375 \
INJECT_SEMANTICS_WIDTH 1240 \
#ALTERNATE_PT_CLOUD_ABS_DIR '/data/ck/data/kitti_obj_bts_pred/result_bts_eigen_v2_pytorch_densenet161/training/plidar_sparsified' \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
cksbatch --nodelist=flaminio ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
bash ~/BEVSEG/PCDet2/scripts/kitti/parta2_train.sh
sample_cmds

