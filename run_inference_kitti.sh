#!/usr/bin/env bash
#kitti
input_dir='/home/nowburn/disk/data/Struct2Depth/kitti/kitti_raw/all/2011_09_26/2011_09_26_drive_0013_sync/image_02/data'
output_dir='/home/nowburn/disk/data/Struct2Depth/kitti/output'

#me
#input_dir='/home/nowburn/disk/data/Struct2Depth/me/split/Recording002_freq3'
#output_dir='/home/nowburn/disk/data/Struct2Depth/me/output/Recording002_freq3'


model_checkpoint='/home/nowburn/disk/data/Struct2Depth/model/kitti/model-199160'
#model_checkpoint='/home/nowburn/disk/data/Struct2Depth/kitti/experiment2/model-4850'

python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion True \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint
