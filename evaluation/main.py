#! /usr/bin/env python
import glob
import os
from detect import Detector
from model import *

# define model_path, image_path
image_path = '/home/ambastha/abhishek/matlab-eval/caltech_tool_kit/data-INRIA/images/set01/V000/'
model_path = '/home/ambastha/abhishek/ssd/caffe/models/VGGNet/VOC0712/SSD_500x500/INRIA-success/'

curr_dir = os.getcwd()
os.chdir(image_path)

# get image list from image_path in sorted manner
img_list = sorted(glob.glob('*.png'))


d = Detector(model_path, image_path)

frames = list()
for i, img in enumerate(img_list):
    img_obj = Image(i+1) # 1 based index for matlab
    img_obj.bboxes.extend(d.detect(img))
    print '[INFO]: Detecting {}'.format(img)
    frames.append(img_obj)

os.chdir(curr_dir)
with open('V000.txt', 'w') as f:
    for fr in frames:
        boxes = fr.bboxes
        for box in boxes:
            data = str(fr.index) + ' ' + box.repr() + '\n'
            f.write(data)



# detect all images
# write to files
