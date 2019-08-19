# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:30:09 2019

@author: x

将视频分解为图片
decompose each video into images by hand !!!!!!


calib.mp4 -> ./calib_jpg/0001.jpg
             ./calib_jpg/0002.jpg
             ...


The original image size is too large, and there is no need 
to write them in folder.
The size of the input image required by the network is 416 x 128.

1.mp4 -> ./dataset/video1/000005.jpg
         ./dataset/video1/000010.jpg
         ...

2.mp4 -> ./dataset/video2/000005.jpg
         ./dataset/video2/000010.jpg
         ...

...

"""

import cv2
import os

split_freq = 3
split_num = 300

video_path = '/home/nowburn/disk/data/Struct2Depth/me/001/Recording002.mp4'
output_dir = '/home/nowburn/disk/data/Struct2Depth/me/split/Recording002_freq' + str(split_freq) + '/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# cap = cv2.VideoCapture("./video/calib.mp4")
cap = cv2.VideoCapture(video_path)

i = 0
cnt = 0
while True:
    i += 1
    ret, frame = cap.read()
    if cnt == split_num or ret is False:
        break

    cv2.imshow("image", frame)
    cv2.waitKey(10)

    # if i%20 == 0:  # calib.mp4
    #     print("i:", i)
    #     cv2.imwrite("./calib_jpg/"  + str('%04d'%i) + ".jpg", frame)

    if i % split_freq == 0:
        print("cnt:", cnt)
        resized_frame = cv2.resize(frame, (416, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_dir + str('%010d' % cnt) + ".png", resized_frame)
        cnt += 1

cap.release()
cv2.destroyAllWindows()
