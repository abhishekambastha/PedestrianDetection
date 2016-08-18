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
    os.chdir(annotation_path)
    images = get_files()
    parser = InriaParser(annotation_path)
    main_dict = {}
    for image in images:
        im = parser.get_annotation(image)
        temp = im.get_dict()
        main_dict[temp['name']] = temp

    print main_dict


if __name__ == '__main__':
    main()
