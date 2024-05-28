import tensorflow as tf
import glob
import data
import numpy as np
import main
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

Data = data.Data()

img_path = '../dataset/CCD/test/'

file_list = sorted(glob.glob(img_path + '*.png'))
label_list = np.load('labels_test.npy')

images = []
labels = []

for i in range(10):
    image = Data.read_img(img_name=file_list[i], img_size=128)
    image = image[np.newaxis, :, :, :]

    label = label_list[i]
    label = label[np.newaxis, :]

    if i == 0:
        images = image
        labels = label
    else:
        images = np.concatenate((images, image), axis=0)
        labels = np.concatenate((labels, label), axis=0)


def angle_losses(prediction, labels):
    safe_v = tf.constant(0.999999)

    predicting_illuminant = prediction
    standard_illuminant = labels

    dot = tf.reduce_sum(predicting_illuminant * standard_illuminant, axis=1)
    dot = tf.clip_by_value(dot, -safe_v, safe_v)

    angle = tf.acos(dot) * (180 / np.pi)

    return angle


input_data = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
input_target = tf.placeholder(tf.float32, shape=[None, 3])

logits = main.net(net_in=input_data)
loss = angle_losses(prediction=logits, labels=input_target)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path='./Model/Model.ckpt')
    feed_dict = {input_data: images, input_target: labels}
    loss_mean = sess.run(loss, feed_dict=feed_dict)
    print(loss_mean)
