import os
import tensorflow as tf
import numpy as np
import main
import glob
import statistics

batch_size = 1
learning_rate = 0.0001
img_size = 128
epochs = 1
image_path = '../dataset/CCD/test/'
label_path = './labels_test.npy'

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


def analyze(data):
    # 确保列表不为空
    if not data:
        raise ValueError("The data list is empty")

    # 排序列表
    sorted_data = sorted(data)

    # print(sorted_data)
    for i in range(56):
        print(i+1, sorted_data[i])

    # 计算平均值
    mean_value = statistics.mean(sorted_data)

    # 计算中位数
    median_value = statistics.median(sorted_data)

    # 计算最小25%的平均值
    n = len(sorted_data)
    lower_25 = sorted_data[:n // 4]
    mean_lower_25 = statistics.mean(lower_25)

    # 计算最大25%的平均值
    upper_25 = sorted_data[-n // 4:]
    mean_upper_25 = statistics.mean(upper_25)

    return mean_value, median_value, mean_lower_25, mean_upper_25


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

        # avg_tower_loss = tf.reduce_mean(input_tensor=tower_loss, axis=0)

        # train_op = optimizer.minimize(loss=avg_tower_loss)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

            saver = tf.train.Saver()
            saver.restore(sess=sess, save_path='./Model_3/Model.ckpt')

            losses = []
            predict_list = []

            i = 0
            while True:
                i = i + 1
                try:
                    predict, angle = sess.run([prediction, loss_op])
                    # print(type(predict))
                    predict_list.append(predict.tolist())
                    # print(i, angle)
                    losses.append(angle)
                except tf.errors.OutOfRangeError:
                    break

            for index, value in enumerate(losses):
                if value < 0.6:
                    print("序号:", index, "损失值:", value)
            # print(losses)
            # print(predict_list)

            # 打开一个文本文件来写入
            # with open('prediction.txt', 'w') as file:
            #     # 将列表中的每个元素写入文件，每个元素占一行
            #     for item in predict_list:
            #         file.write(str(item) + '\n')

            # 调用函数并打印结果
            mean_value, median_value, mean_lower_25, mean_upper_25 = analyze(losses)
            print(f"平均值: {mean_value}")
            print(f"中位数: {median_value}")
            print(f"最小25%的平均值: {mean_lower_25}")
            print(f"最大25%的平均值: {mean_upper_25}")

            return predict_list


train()