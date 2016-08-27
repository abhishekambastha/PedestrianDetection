import lmdb
import os
import cv2
import glob
import random
import numpy as np
from model import *
import py.datum_pb2 as datum_pb2
from ParseAnnotation import InriaParser

class InriaLMDB():

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.parser = InriaParser(self.dataset_path)


    def get_image_list(self):
        self.pos_images_path = os.path.join(self.dataset_path, 'pos')
        self.neg_images_path = os.path.join(self.dataset_path, 'neg')
        cdir = os.getcwd()
        os.chdir(self.pos_images_path)
        pos_samples = glob.glob('*.png')
        os.chdir(self.neg_images_path)
        neg_samples = glob.glob('*.png')
        os.chdir(cdir)
        return (pos_samples, neg_samples)


    def create_img_objects(self):
        pos, neg = self.get_image_list()
        print 'Loaded {} positive and {} negative samples'.format(len(pos), len(neg))
        img_objs = list()

        for image in pos:
            img_objs.append(self.parser.get_annotation(image))

        for image in neg:
            img_objs.append(self._negative_samples(image))

        random.shuffle(img_objs)

        return img_objs


    def _negative_samples(self, img):
        img_obj = Image(img)
        img_obj.image_path = os.path.join(self.dataset_path, 'neg')
        return img_obj


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
    inria = InriaLMDB('/home/ambastha/data/inria-person/INRIAPerson/Test')
    img_objs = inria.create_img_objects()
    l = LMDB('test.lmdb')
    l.create_lmdb(img_objs)
