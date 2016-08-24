#! /usr/bin/env python

from ParseAnnotation import *
from model import *
import os
import glob


def get_files():
    annotations = glob.glob('*.txt')
    return annotations


def main():
    annotation_path = '/home/ambastha/data/inria-person/INRIAPerson/Train/annotations'
    curr_dir = os.getcwd()
    os.chdir(annotation_path)
    images = get_files()
    parser = InriaParser(annotation_path)
    main_dict = {}
    for image in images:
        im = parser.get_annotation(image)
        kitti(im)
    ## Modify loop 1- Kitti , LMDB options
    # for image in images:
    #     im = parser.get_annotation(image)
    #     temp = im.get_dict()
    #     main_dict[temp['name']] = temp
    #
    # os.chdir(curr_dir)
    # with open('results.json', 'w') as f:
    #     json.dump(main_dict, f, indent=4)

def kitti(img_obj):
    """
        Steps:
            1. Create a .txt file from image name
            2. Get the bbox coordinates from image obj
            3. Write to file
    """
    image_name = os.path.splitext(img_obj.name)[0]
    path = '/home/ambastha/data/inria-person/kitti'
    with open(os.path.join(path, image_name + '.txt'), 'w') as f:
        for box in img_obj.bboxes:
            f.write('Person' + ' 0.00' +  ' 1' +  ' 0' + ' ' + str(box.x1) + ' ' + str(box.y1) + ' ' + str(box.x2) + ' ' + str(box.y2) + ' '\
                '0' +  ' 0' +  ' 0'  + ' 0' + ' 0' + ' 0' + ' 1' + '\n')



if __name__ == '__main__':
    main()
