#!/bin/sh
username=$1
modelname=$2
gpu_ids=$3
echo "username=$1"
echo "modelname=$2"

python Evaluation/pytorch-fid/fid_score.py --gpu $gpu_ids /home/$username/codelab/datasets/db_crossview_pano_sep/test/test_pano  \
/home/$username/codelab/I2I/panogan_test_results/$modelname/test_latest/images | tee eva_fid_$modelname.txt
