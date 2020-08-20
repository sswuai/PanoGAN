#!/bin/sh
username=$1
modelname=$2
gpu_ids=$3
echo "username=$1"
echo "modelname=$2"

CUDA_VISIBLE_DEVICES=$gpu_ids python Evaluation/KL_model_data.py /home/$username/codelab/datasets/opcrossview/pano_test  \
/home/$username/codelab/I2I/panogan_test_results/$modelname/test_latest/images #| tee eva_KL_$modelname.txt
