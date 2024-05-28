import numpy as np
import glob
import cv2

class Data(object):
    def __init__(self):
        self.img_path = '/home/a/PycharmProjects/dataset/CCD/'
        self.img_list = sorted(glob.glob(self.img_path + '*.png'))
        self.pointer = 0
        self.num_train_example = len(self.img_list)
        self.labels = np.load('labels_train.npy')

    @staticmethod
    def read_img(img_name, img_size):
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(img_size, img_size))
        img = img / img.max()
        return img

    def load_image_label_batch(self, batch_size, img_size):
        if self.pointer + batch_size < self.num_train_example:
            first = self.pointer
            last = first + batch_size
            self.pointer = last
        else:
            first = self.num_train_example - batch_size
            last = first + batch_size
            self.pointer = 0

        img_batch_path = self.img_list[first : last]
        label_batch_path = self.labels[first : last]
        images = []
        labels = []

        for i in range(batch_size):
            image = self.read_img(img_batch_path[i], img_size)
            image = image[np.newaxis, :, :, :]
            label = label_batch_path[i, :]
            label = label[np.newaxis, :]

            if i == 0:
                images = image
                labels = label
            else:
                images = np.concatenate((images, image), axis=0)
                labels = np.concatenate((labels, label), axis=0)

        return images, labels