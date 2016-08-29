import lmdb
import os
import cv2
import glob
import random
import json
import numpy as np
from model import *
import py.datum_pb2 as datum_pb2
from ParseAnnotation import InriaParser

class CaltechLMDB():

    def __init__(self, dataset_path, skip=30):
        self.dataset_path = dataset_path
        self.annotation_file = os.path.join(dataset_path, 'Annotations', 'annotations.json')
        self.image_path = os.path.join(dataset_path, 'Images')
        self._annotations = None
        self.skip = skip
        self.img_height = 640
        self.img_width = 480

    @property
    def annotation(self):
        if not self._annotations:
            with open(self.annotation_file, 'rb') as f:
                print 'reading annotations file'
                self._annotations = json.load(f)
            print 'Loaded annotations from {}'.format(self.annotation_file)
        return self._annotations


    def get_image_list(self):
        cur_dir = os.getcwd()
        os.chdir(os.path.join(self.dataset_path, 'Images'))
        full_image_list = sorted(glob.glob('*.jpg'))
        os.chdir(cur_dir)

        #Create Train - Test split
        offset = 3475 ## To include train upto set05
        train_list = list(full_image_list[0:(len(full_image_list)/2 + offset):self.skip])
        test_list = list(full_image_list[(len(full_image_list)/2 + offset): len(full_image_list):self.skip])

        #Shuffle the data
        random.shuffle(train_list)
        random.shuffle(test_list)


        return train_list, test_list


    def valid_bbox(self, bbox):
        is_valid = True

        if bbox.x1 < 0 or bbox.x1 > self.img_width:
            is_valid = False

        if bbox.x2 < 0 or bbox.x2 > self.img_width:
            is_valid = False

        if bbox.y1 < 0 or bbox.y1 > self.img_height:
            is_valid = False

        if bbox.y2 < 0 or bbox.y2 > self.img_height:
            is_valid = False

        return is_valid


    def get_annotation(self, img):
        boxes = list()
        anno = self.annotation[img]
        for box in anno['coords_list']:
            bbox = Bbox()
            bbox.x1 = box['x1']
            bbox.x2 = box['x2']
            bbox.y1 = box['y1']
            bbox.y2 = box['y2']
            if self.valid_bbox(bbox):
                boxes.append(bbox)
        return boxes


    def get_image_obj(self, img):
        boxes = self.get_annotation(img)
        img_obj = Image(img)
        img_obj.bboxes.extend(boxes)
        img_obj.image_path = self.image_path
        img_obj.height = self.img_height
        img_obj.width = self.img_width
        return img_obj


    def create_img_objects(self):
        img_objs = list()
        train_list, test_list = self.get_image_list()
        print 'Number of samples in training set {}'.format(len(train_list))
        print 'Number of samples in test set {}'.format(len(test_list))
        train_img_objs = list()
        test_img_objs = list()

        for sample in train_list:
            img_obj = self.get_image_obj(sample)
            train_img_objs.append(img_obj)

        for sample in test_list:
            img_obj = self.get_image_obj(sample)
            test_img_objs.append(img_obj)

        return train_img_objs, test_img_objs


class LMDB():

    def __init__(self, lmdb_name):
        self.lmdb_name = lmdb_name


    def normalize_bbox(self, im):
        width = im.width
        print '[DEBUG]: ',im.width, im.height
        height = im.height
        normal_boxes = list()
        boxes = im.bboxes
        for box in boxes:
            normal_box = Bbox()
            normal_box.x1 = (1.0 * int(box.x1) ) / int(im.width)
            normal_box.x2 = (1.0 * int(box.x2) ) / int(im.width)
            normal_box.y1 = (1.0 * int(box.y1) ) / int(im.height)
            normal_box.y2 = (1.0 * int(box.y2) ) / int(im.height)
            normal_boxes.append(normal_box)

        print '[DEBUG]: no. of bbox', len(normal_boxes)
        return normal_boxes


    def create_lmdb(self, img_objs):

        map_size = 1099511627776
        print '[INFO] Initialized lmdb with size {}'.format(map_size)
        print '[INFO] LMDB In', os.getcwd()
        env = lmdb.open(self.lmdb_name, map_size=map_size)
        num_images = len(img_objs)
        # mean_image = np.zeros_like(cv2.imread(img_objs[0].image_path), dtype=np.float32)

        with env.begin(write=True) as txn:
            for i, im in enumerate(img_objs):
                annotated_datum = datum_pb2.AnnotatedDatum()
                img_file = cv2.imread(os.path.join(im.image_path, im.name))
                print 'Processing {}'.format(os.path.join(im.image_path, im.name))
                # mean_image += (1000.0 / num_images) * img_file
                annotated_datum.datum.data = cv2.imencode('.jpg', img_file)[1].tostring()

                annotated_datum.datum.label = -1
                annotated_datum.datum.encoded = True
                annotated_datum.type = datum_pb2.AnnotatedDatum.BBOX

                normal_boxes = self.normalize_bbox(im)

                for box in normal_boxes:
                    annotation_group = annotated_datum.annotation_group.add()

                    #using pascal dataset's label
                    annotation_group.group_label = 1
                    annotation = annotation_group.annotation.add()
                    annotation.instance_id = 0
                    datum_bbox = annotation.bbox

                    datum_bbox.xmin = box.x1
                    datum_bbox.ymin= box.y1
                    datum_bbox.xmax = box.x2
                    datum_bbox.ymax = box.y2
                    datum_bbox.difficult = False

                str_id = '{:08}'.format(i)
                txn.put(str_id.encode('ascii'), annotated_datum.SerializeToString())
                print '[INFO]: Added image {} with {} bboxes'.format(im.image_path, len(normal_boxes))


if __name__ == '__main__':
    caltech = CaltechLMDB('/home/ambastha/repo/caltech-dataset-convertor/target')
    train_objs, test_objs = caltech.create_img_objects()

    train_lmdb = LMDB('Train.lmdb')
    test_lmdb = LMDB('Test.lmdb')

    train_lmdb.create_lmdb(train_objs)
    test_lmdb.create_lmdb(test_objs)

    print train_objs[0].get_dict()
