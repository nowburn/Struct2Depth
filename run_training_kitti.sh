#!/usr/bin/env bash
ckpt_dir="/home/nowburn/disk/data/Struct2Depth/kitti/experiment2"
data_dir="/home/nowburn/disk/data/Struct2Depth/kitti/kitti_processed2/" # Set for KITTI
imagenet_ckpt="/home/nowburn/disk/data/Struct2Depth/model/resnet18/model.ckpt"

python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --architecture resnet \
  --imagenet_ckpt $imagenet_ckpt \
  --imagenet_norm True


