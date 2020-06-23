#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 48
#SBATCH  --gres gpu:1


source ~/.bashrc &> /dev/null
cd $CODE/BEVSEG/PCDet2/tools

NAME=argo_2

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
python \
test.py \
--cfg_file cfgs/argo/pointpillar.yaml \
--batch_size 20 \
--extra_tag $NAME \
--eval_all \

<< sample_cmds
cksbatch --nodelist=como ~/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
bash $CODE/BEVSEG/PCDet2/scripts/argo/pointpillar_eval.sh
sample_cmds

