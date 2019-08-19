# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Common utilities for data pre-processing, e.g. matching moving object across frames."""
import cv2
import numpy as np
import os
import time
import shutil
from PIL import Image

SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128
STEPSIZE = 1
path = '/home/nowburn/disk/data/Struct2Depth/kitti/kitti_processed2/'


def compute_overlap(mask1, mask2):
    # Use IoU here.
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)


def align(seg_img1, seg_img2, seg_img3, threshold_same=0.3):
    res_img1 = np.zeros_like(seg_img1)
    res_img2 = np.zeros_like(seg_img2)
    res_img3 = np.zeros_like(seg_img3)
    remaining_objects2 = list(np.unique(seg_img2.flatten()))
    remaining_objects3 = list(np.unique(seg_img3.flatten()))
    for seg_id in np.unique(seg_img1):
        # See if we can find correspondences to seg_id in seg_img2.
        max_overlap2 = float('-inf')
        max_segid2 = -1
        for seg_id2 in remaining_objects2:
            overlap = compute_overlap(seg_img1 == seg_id, seg_img2 == seg_id2)
            if overlap > max_overlap2:
                max_overlap2 = overlap
                max_segid2 = seg_id2
        if max_overlap2 > threshold_same:
            max_overlap3 = float('-inf')
            max_segid3 = -1
            for seg_id3 in remaining_objects3:
                overlap = compute_overlap(seg_img2 == max_segid2, seg_img3 == seg_id3)
                if overlap > max_overlap3:
                    max_overlap3 = overlap
                    max_segid3 = seg_id3
            if max_overlap3 > threshold_same:
                res_img1[seg_img1 == seg_id] = seg_id
                res_img2[seg_img2 == max_segid2] = seg_id
                res_img3[seg_img3 == max_segid3] = seg_id
                remaining_objects2.remove(max_segid2)
                remaining_objects3.remove(max_segid3)
    return res_img1, res_img2, res_img3


def triplet_align():
    img_dir_list = [path + dir for dir in os.listdir(path) if dir.split('_')[-1] == 'mask']
    for img_dir in img_dir_list:
        print('Processing %s ' % img_dir.split('/')[-1])
        output_dir = img_dir[:-4] + 'tmp/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        file_names = os.listdir(img_dir)
        file_names.sort()
        file_list = [os.path.join(img_dir, file) for file in file_names]

        img_list = []
        res_img = []
        step = 0
        step_size = 1
        cnt = 0
        n = len(file_list)
        while step <= n - 3:
            for i in range(step, step + 3):
                img_list.append(np.array(Image.open(file_list[i])))
            res_img1, res_img2, res_img3 = align(img_list[0], img_list[1], img_list[2])
            res_img.append(res_img1)
            res_img.append(res_img2)
            res_img.append(res_img3)
            for i in range(3):
                img_name = '0' * (10 - len(str(cnt))) + str(cnt)
                Image.fromarray(res_img[i]).save(output_dir + img_name + '-fseg.png')
                cnt += 1
            res_img.clear()
            img_list.clear()
            step += step_size


def stick_imgs():
    fseg_dir_list = [path + dir for dir in os.listdir(path) if dir.split('_')[-1] == 'tmp']
    for img_dir in fseg_dir_list:
        ct = 0
        file_names = os.listdir(img_dir)
        file_names.sort()
        file_list = [os.path.join(img_dir, file) for file in file_names]
        output_dir = img_dir[:-3] + 'align/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for i in range(SEQ_LENGTH, len(file_list) + 1, 3):
            imgnum = str(ct).zfill(10)
            if os.path.exists(output_dir + imgnum + '-fseg.png'):
                ct += 1
                continue
            big_img = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))
            wct = 0

            for j in range(i - SEQ_LENGTH, i):  # Collect frames for this sample.
                img = cv2.imread(file_list[j])
                img = cv2.resize(img, (WIDTH, HEIGHT))
                big_img[:, wct * WIDTH:(wct + 1) * WIDTH] = img
                wct += 1
            cv2.imwrite(output_dir + imgnum + '-fseg.png', big_img)
            ct += 1


def copy_del():
    origin = [path + dir for dir in os.listdir(path) if
              dir.split('_')[-1] == 'align']
    to = [path + dir for dir in os.listdir(path) if
          len(dir.split('_')) == 7]
    origin.sort()
    to.sort()

    for i, dir in enumerate(origin):
        for file in os.listdir(dir):
            shutil.move(origin[i] + '/' + file, to[i] + '/' + file)

    del_dir_list = [path + dir for dir in os.listdir(path) if
                    dir.split('_')[-1] == 'mask' or dir.split('_')[-1] == 'tmp' or dir.split('_')[-1] == 'align']
    for dir in del_dir_list:
        shutil.rmtree(dir)


if __name__ == '__main__':
    start_time = time.time()
    triplet_align()
    print('Processing gen_align_data ...')
    stick_imgs()
    print('Processing copy_del ...')
    copy_del()
    end_time = time.time()
    print(end_time - start_time, "s")
