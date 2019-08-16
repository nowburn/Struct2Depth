#!/usr/bin/env bash
input_dir='/home/nowburn/disk/data/Struct2Depth/kitti/kitti_raw/all/2011_09_26/2011_09_26_drive_0013_sync/image_02/data'
output_dir='/home/nowburn/disk/data/Struct2Depth/kitti/output2'
#model_checkpoint='/home/nowburn/disk/data/Struct2Depth/model/cityscapes/model-154688'
model_checkpoint='/home/nowburn/disk/data/Struct2Depth/kitti/experiment2/model-4850'

python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion True \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint
