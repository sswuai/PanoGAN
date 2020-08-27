set -ex
username=$1
loop_count=$2
gpu_ids=$3

python test.py \
--dataroot \
/home/$username/codelab/datasets/db_crossview_pano_APS_test \
--results_dir \
/home/$username/codelab/I2I/panogan_test_results \
--name \
panogan_cvusa_partial_feedback_$loop_count \
--model \
panogan \
--netG \
unet_afl_v5 \
--netD \
fpd \
--ngf \
64 \
--ndf \
64 \
--direction \
AtoB \
--epoch \
latest \
--dataset_mode \
panoaligned \
--norm \
instance \
--preprocess \
none \
--num_test \
10000000 \
--eval \
--loop_count \
$loop_count \
--alpha \
0.5 0.5 0.5 0.5 0.5 \
--gpu_ids \
$gpu_ids
