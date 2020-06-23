#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:4


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=verifying_repo_works_with_training3

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
python -m torch.distributed.launch --nproc_per_node=4 \
train.py \
--cfg_file cfgs/pointpillar.yaml \
--sync_bn \
--launcher pytorch \
--batch_size 60 \
--extra_tag $NAME \

<< sample_cmds
cksbatch --nodelist=pavia ~/BEVSEG/PCDet2/scripts/verifying_repo_works_with_training.sh
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/verifying_repo_works_with_training.sh
bash $CODE/BEVSEG/PCDet2/scripts/verifying_repo_works_with_training.sh
sample_cmds

