set -ex
username=$1
gpu_ids=$2

echo "username=$1"

python test.py \
--dataroot \
/home/$username/codelab/datasets/opcrossview \
--results_dir \
/home/$username/codelab/I2I/panogan_test_results \
--name \
panogan_op \
--model \
panoganBaseline5a \
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
panoaligned4 \
--norm \
instance \
--preprocess \
none \
--crop_size \
256 \
--num_test \
10000000 \
--eval \
--loop_count \
3 \
--alpha \
0.5 0.5 0.5 0.5 0.5 \
--gpu_ids \
$gpu_ids