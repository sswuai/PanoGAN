#!/bin/sh
#python 3
echo "-----------------------------------evale ssmi and psnr ----------------------------------------------------"
username=$1
modelname=$2
gpuid=$3
echo "username=$1"
echo "modelname=$2"

CUDA_VISIBLE_DEVICES=$gpuid python Evaluation/computer_ssmi_psnr_sharpness/compute_ssmi_psnr.py /home/$username/codelab/datasets/db_crossview_pano_sep/test/test_pano  \
/home/$username/codelab/I2I/panogan_test_results/$modelname/test_latest/images | tee eva_ssmi_psnr_$modelname.txt

#python /home/dh/Evaluation-project/computer_ssmi_psnsr_sharpness/compute_ssmi_psnr.py /home/dh/Evaluation-project/pano_test_results/img/img_real /home/dh/Evaluation-project/pano_test_results/img/img_synthesized
