#!/bin/sh
echo "-----------------------------------evale topK KL for panorama partial ----------------------------------------------------"
username=$1
modelname=$2
gpuid=$3
echo "username=$1"
echo "modelname=$2"

CUDA_VISIBLE_DEVICES=$gpuid  python Evaluation/compute_topK_KL.py /home/$username/codelab/datasets/db_crossview_pano_sep/test/test_pano  \
/home/$username/codelab/I2I/panogan_test_results/$modelname/test_latest/images #| tee eva_topK_KL_$modelname.txt
