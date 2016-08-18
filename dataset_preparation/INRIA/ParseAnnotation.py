import os
import re
from model import *

class InriaParser():
    def __init__(self, path):
        self.path = path


    def get_annotation(self, img_name):
        filepath = os.path.join(self.path, img_name)
        with open(filepath, 'r') as f:
            data = f.readlines()

        bboxes = list()

        for d in data:
            if re.search(img_name.strip('.txt'), d):
                filename = re.findall(img_name.strip('.txt') + '.[A-z]*', d)
            if re.search('age size', d):
                img_size = re.findall('\d+', d)
            if re.search('.*erson.*\(\d+,\s\d+\)\s-\s\(\d+,\s\d+\)', d):
                coords = re.findall('\d+', d)
                box = Bbox()
                box.x1 = coords[1]
                box.y1 = coords[2]
                box.x2 = coords[3]
                box.y2 = coords[4]
                bboxes.append(box)

        im = Image(filename[0])
        im.width = img_size[0]
        im.height = img_size[1]
        im.add_rectangles(bboxes)

        return im

if __name__ == '__main__':
    parser = InriaParser('/home/ambastha/data/inria-person/INRIAPerson/Train/annotations')
    im = parser.get_annotation('person_177.txt')
    print im.get_dict()
    # print box
