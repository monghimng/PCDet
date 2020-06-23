#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:1


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
python test.py --cfg_file cfgs/pointpillar.yaml --batch_size 2 \
--ckpt /data/ck/BEVSEG/PCDet2/output/pointpillar/verifying_repo_works_with_training2/ckpt/checkpoint_epoch_80.pth \
#--ckpt $CODE/BEVSEG/PCDet2/pointpillar.pth

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/verifying_repo_works_with_testing.sh
bash $CODE/BEVSEG/PCDet2/scripts/verifying_repo_works_with_testing.sh
sample_cmds

