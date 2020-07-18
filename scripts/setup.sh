#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:3


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/
python setup.py develop
rm -r kitti/
ln -s /data/ck/data/kitti_pcdet/kitti $CODE/BEVSEG/PCDet2/data/kitti
ln -s /data/ck/data/argoverse/argoverse-tracking-kitti-format $CODE/BEVSEG/PCDet2/data/argo
mkdir -p /data/ck/BEVSEG/PCDet2/output/ # write model checkpoints to bigger disk
ln -s /data/ck/BEVSEG/PCDet2/output/ $CODE/BEVSEG/PCDet2/output
rm -r /data/ck/data/argoverse/argoverse-tracking-kitti-format/
rsync_local_data_to_remote_data /data/ck/data/argoverse/argoverse-tracking-kitti-format/ pavia como
rsync_local_data_to_remote_data /data/ck/data/argoverse/argoverse-tracking-kitti-format/ r11 pavia

rsync_local_data_to_remote_data /data/ck/data/kitti_obj_bts_pred/result_bts_eigen_v2_pytorch_densenet161/training/ r11 flaminio
rsync_local_data_to_remote_data /data/ck/data/kitti_obj_det r11 flaminio
rsync_local_data_to_remote_data /data/ck/data/kitti_pcdet r11 flaminio
# locally
#git remote add upstream https://github.com/sshaoshuai/PCDet.git

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}



<< sample_cmds
cksbatch --nodelist=pavia ~/pcdet2/scripts/train_parta2.sh
bash $CODE/BEVSEG/PCDet2/scripts/setup.sh
sample_cmds

