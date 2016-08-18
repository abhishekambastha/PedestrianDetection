import json

class Image():
    def __init__(self, name):
        self.name = name
        self.bboxes = []
        self.num_objects = 0
        self.width = 0
        self.height = 0

    def add_rectangles(self, rects):
        self.bboxes.extend(rects)

    def to_json(self):
        im = {}
        rects = {}
        rects['num_objects'] = len(self.bboxes)
        rects['coords_list'] = [d.__dict__ for d in self.bboxes]
        im[self.name] = rects
        return im

    def get_dict(self):
        self.__dict__['bboxes'] = [rect.__dict__ for rect in self.bboxes]
        self.__dict__['num_objects'] = len(self.bboxes)
        return self.__dict__


class Bbox():
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0


if __name__ == '__main__':
    a = Bbox()
    a.x1 = 12
    a.x2 = 14
    a.y1 = 90
    a.y2 = 109

    im = Image('myimage.jpg')
    im.add_rectangles([a, a])

    d = im.to_json()

    print im.get_dict()
