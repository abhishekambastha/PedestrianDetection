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
        temp = im.get_dict()
        main_dict[temp['name']] = temp

    os.chdir(curr_dir)
    with open('results.json', 'w') as f:
        json.dump(main_dict, f, indent=4)


if __name__ == '__main__':
    main()
