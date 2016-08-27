
class Image():
    def __init__(self, index):
        self.bboxes = list()
        self.index = index

class Bbox():
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.height = 0
        self.width = 0
        self.score = 0

    def update_cords(self, x1, y1, widht, height, score):
        self.x1 = x1
        self.y1 = y1
        self.width = widht
        self.height = height
        self.score = score

    def repr(self):
        string = str(self.x1) + ' ' + str(self.y1) + ' ' + str(self.width) +\
        ' ' + str(self.height) + ' ' + str(self.score)
        return string
