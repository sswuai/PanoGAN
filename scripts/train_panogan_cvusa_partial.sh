set -ex
username=$1
batch_size=$2
loop_count=$3
gpu_ids=$4

echo "username=$username"
echo "batch_size=$batch_size"
echo "feedback iteration count=$loop_count"
echo "gpu_ids=$gpu_ids"

python train.py \
--dataroot \
/home/$username/codelab/datasets/db_crossview_pano_APS_train_partial \
--name \
panogan_cvusa_partial_feedback_$loop_count \
--model  \
panogan  \
--netG \
unet_afl_v5 \
--netD \
fpd \
--ngf \
64 \
--ndf \
64 \
--norm \
instance \
--direction \
AtoB \
--lambda_L1 \
100 \
--lambda_L1_seg \
100 \
--dataset_mode \
panoaligned \
--preprocess \
none \
--gan_mode \
vanilla \
--gpu_ids \
$gpu_ids \
--batch_size \
$batch_size \
--niter \
15 \
--niter_decay \
15 \
--save_epoch_freq \
15 \
--display_id \
1 \
--display_port \
8097 \
--display_ncols \
0 \
--display_freq \
20 \
--display_winsize \
1024 \
--loop_count \
$loop_count \
--alpha \
0.5 0.5 0.5 0.5 0.5 \
--verbose
