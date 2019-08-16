#!/usr/bin/env bash
prediction_dir="/home/nowburn/disk/data/Struct2Depth/kitti/output"

# Use these settings for KITTI:
eval_list_file="/home/nowburn/python_projects/cv/Struct2Depth/dataset/kitti/test_scenes_eigen.txt"
eval_crop="garg"
eval_mode="kitti"

## Use these settings for Cityscapes:
#eval_list_file="CITYSCAPES_FULL/test_files_cityscapes.txt"
#eval_crop="none"
#eval_mode="cityscapes"

python evaluate.py \
  --logtostderr \
  --prediction_dir $prediction_dir \
  --eval_list_file $eval_list_file \
  --eval_crop $eval_crop \
  --eval_mode $eval_mode