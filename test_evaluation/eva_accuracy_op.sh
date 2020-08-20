#!/bin/sh
echo "-----------------------------------evale accuracy baseline1a1 for panorama partial ----------------------------------------------------"
username=$1
modelname=$2
gpu_ids=$3
echo "username=$1"
echo "modelname=$2"

CUDA_VISIBLE_DEVICES=$gpu_ids python Evaluation/compute_accuracies.py /home/$username/codelab/datasets/opcrossview/pano_test  \
/home/$username/codelab/I2I/panogan_test_results/$modelname/test_latest/images #| tee eva_acc_$modelname.txt