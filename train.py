import os
import tensorflow as tf
import numpy as np
import main
import glob
import datetime

# 获取当前时间，并格式化为需要的字符串
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

batch_size = 16
learning_rate = 0.0001
img_size = 128
iteration = 1000
epochs = 5
image_path = '../dataset/CCD/train/'
label_path = './labels_train.npy'
ifrestore = 1
checkpoint_restore_path = "./Model/Model.ckpt"
checkpoint_save_path = f"./Model-{current_time}"

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU
gpus = [0]

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

img_path = image_path
img_list = sorted(glob.glob(img_path + '*.png'))  # 储存所有图片的路径
labels = np.load(label_path)  # 加载标签


def _parse_function(filename, label):
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize_images(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    return image, label

# 角度损失函数
def angle_losses(prediction, _labels):
    safe_v = tf.constant(0.999999)

    predicting_illuminant = prediction
    standard_illuminant = _labels

    dot = tf.reduce_sum(predicting_illuminant * standard_illuminant, axis=1)
    dot = tf.clip_by_value(dot, -safe_v, safe_v)

    angle = tf.acos(dot) * (180 / np.pi)
    angle = tf.reduce_mean(angle)

    return angle


def train():
    with tf.Graph().as_default():
        # 图片和标签创建两个张量
        filenames = tf.constant(img_list)
        label = tf.constant(labels, dtype=float)
        # 从定义好的张量创建数据集，每一个元素都是（图片，标签）元组
        datasets = tf.data.Dataset.from_tensor_slices((filenames, label))
        # 使用_parse_function（）函数对数据集中的每一个元素进行处理
        datasets = datasets.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # datasets = datasets.shuffle(buffer_size=1000)
        # 将数据集分为大小为‘batch_size’的批次
        datasets = datasets.batch(batch_size=batch_size)
        # 将数据集重复‘epochs’次
        datasets = datasets.repeat(epochs)
        # 为数据集创建迭代器
        iterator = datasets.make_initializable_iterator()
        # 从迭代器中返回下一个数据批次和标签
        input_data, input_target = iterator.get_next()

        # tower_grads = []
        tower_loss = []

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # 拆分数据，进行多块gpu并行训练
        images_splits = tf.split(value=input_data, num_or_size_splits=len(gpus), axis=0)
        labels_splits = tf.split(value=input_target, num_or_size_splits=len(gpus), axis=0)

        for i in range(len(gpus)):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):
                with tf.variable_scope(name_or_scope='my_net', reuse=tf.AUTO_REUSE):
                    prediction = main.net(net_in=images_splits[i])

                    loss_op = angle_losses(prediction=prediction, _labels=labels_splits[i])

                    tower_loss.append(loss_op)

        avg_tower_loss = tf.reduce_mean(input_tensor=tower_loss, axis=0)

        train_op = optimizer.minimize(loss=avg_tower_loss)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

            saver = tf.train.Saver()
            if ifrestore==1:
                saver.restore(sess=sess, save_path=checkpoint_restore_path)

            i = 0
            while True:
                i = i + 1
                try:
                    _, predict, angle = sess.run([train_op, prediction, loss_op])
                    print(i, angle)
                except tf.errors.OutOfRangeError:
                    break

            saver.save(sess, checkpoint_save_path)


train()