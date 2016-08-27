import numpy as np
import caffe
import os
from model import *

class Detector():
    def __init__(self, model_path, image_path):
        self.model_path = model_path
        self.image_path = image_path
        self._initialize_device()
        self._initialize_caffe_model()
        self._configure_preprocessor()

    def _initialize_device(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

    def _initialize_caffe_model(self):
        self.model_def = os.path.join(self.model_path, 'deploy.prototxt')
        self.model_weights = os.path.join(self.model_path, 'VGG_VOC0712_SSD_500x500_iter_60000.caffemodel')
        self.net = caffe.Net(self.model_def, self.model_weights, caffe.TEST)

    def _configure_preprocessor(self):
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))

    def _load_image(self, img_path):
        self.image_resize = 500
        self.net.blobs['data'].reshape(1,3, self.image_resize, self.image_resize)
        self.image = caffe.io.load_image(img_path)
        transformed_image = self.transformer.preprocess('data', self.image)
        self.net.blobs['data'].data[...] = transformed_image

    def detect(self, img):
        self._load_image(os.path.join(self.image_path, img))
        return self._parse_detections(0.02)

    def _parse_detections(self, threshold):
        detections = self.net.forward()['detection_out']
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= threshold]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        bboxes = list()
        for i in xrange(top_conf.shape[0]):
            box = Bbox()
            xmin = int(round(top_xmin[i] * self.image.shape[1]))
            ymin = int(round(top_ymin[i] * self.image.shape[0]))
            xmax = int(round(top_xmax[i] * self.image.shape[1]))
            ymax = int(round(top_ymax[i] * self.image.shape[0]))
            width = xmax - xmin
            height = ymax - ymin
            score = top_conf[i]

            box.update_cords(xmin, ymin, width, height, score)
            bboxes.append(box)


        return bboxes
