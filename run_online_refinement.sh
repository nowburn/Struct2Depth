#!/usr/bin/env bash
prediction_dir="/home/nowburn/disk/data/Struct2Depth/kitti/refinement"
model_ckpt="pretrained_model/kitti/model-199160"
handle_motion="false"
size_constraint_weight="0" # This must be zero when not handling motion.

# If running on KITTI, set as follows:
data_dir="/home/nowburn/disk/data/Struct2Depth/kitti/output/"
triplet_list_file="$data_dir/test_files_eigen_triplets.txt"
triplet_list_file_remains="$data_dir/test_files_eigen_triplets_remains.txt"
ft_name="kitti"

## If running on Cityscapes, set as follows:
#data_dir="CITYSCAPES_SEQ2_LR_TEST/" # Set for Cityscapes
#triplet_list_file="/CITYSCAPES_SEQ2_LR_TEST/test_files_cityscapes_triplets.txt"
#triplet_list_file_remains="CITYSCAPES_SEQ2_LR_TEST/test_files_cityscapes_triplets_remains.txt"
#ft_name="cityscapes"

python optimize.py \
  --logtostderr \
  --output_dir $prediction_dir \
  --data_dir $data_dir \
  --triplet_list_file $triplet_list_file \
  --triplet_list_file_remains $triplet_list_file_remains \
  --ft_name $ft_name \
  --model_ckpt $model_ckpt \
  --file_extension png \
  --handle_motion $handle_motion \
  --size_constraint_weight $size_constraint_weight