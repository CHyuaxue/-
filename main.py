import tensorflow as tf
# import numpy as np
import conf_net
import rgb_net
# import slit

cn = conf_net.ConfNet()
rn = rgb_net.RgbNet()


def net(net_in):  # tensor
    tensor_1, tensor_2 = tf.split(value=net_in, num_or_size_splits=2, axis=1)  # 16*64*128*3
    tensor_1_1, tensor_1_2 = tf.split(value=tensor_1, num_or_size_splits=2, axis=2)  # 16*64*64*3
    tensor_2_1, tensor_2_2 = tf.split(value=tensor_2, num_or_size_splits=2, axis=2)
    # print(tensor_1_1.shape)

    cn_output_1_1 = cn.conf_model(tensor_1_1, 'cn_tensor_1_1')  # 16*9*9*1
    cn_output_1_2 = cn.conf_model(tensor_1_2, 'cn_tensor_1_2')
    cn_output_2_1 = cn.conf_model(tensor_2_1, 'cn_tensor_2_1')
    cn_output_2_2 = cn.conf_model(tensor_2_2, 'cn_tensor_2_2')
    # print(cn_output_1_1.shape)

    conf_1_1 = tf.reduce_sum(input_tensor=cn_output_1_1, axis=(1, 2))
    conf_1_2 = tf.reduce_sum(input_tensor=cn_output_1_2, axis=(1, 2))
    conf_2_1 = tf.reduce_sum(input_tensor=cn_output_2_1, axis=(1, 2))
    conf_2_2 = tf.reduce_sum(input_tensor=cn_output_2_2, axis=(1, 2))

    rn_output_1_1 = rn.rgb_model(tensor_1_1, 'rn_tensor_1_1')  # 16*18*18*3
    rn_output_1_2 = rn.rgb_model(tensor_1_2, 'rn_tensor_1_2')
    rn_output_2_1 = rn.rgb_model(tensor_2_1, 'rn_tensor_2_1')
    rn_output_2_2 = rn.rgb_model(tensor_2_2, 'rn_tensor_2_2')
    # print(rn_output_1_1.shape)

    rgb_1_1 = tf.reduce_sum(input_tensor=rn_output_1_1, axis=(1, 2))
    rgb_1_2 = tf.reduce_sum(input_tensor=rn_output_1_2, axis=(1, 2))
    rgb_2_1 = tf.reduce_sum(input_tensor=rn_output_2_1, axis=(1, 2))
    rgb_2_2 = tf.reduce_sum(input_tensor=rn_output_2_2, axis=(1, 2))

    prediction_1_1 = conf_1_1 * rgb_1_1
    prediction_1_2 = conf_1_2 * rgb_1_2
    prediction_2_1 = conf_2_1 * rgb_2_1
    prediction_2_2 = conf_2_2 * rgb_2_2

    prediction_1 = tf.add(prediction_1_1, prediction_1_2)
    prediction_2 = tf.add(prediction_2_1, prediction_2_2)
    prediction = tf.add(prediction_1, prediction_2)
    prediction = tf.nn.l2_normalize(prediction, 1)

    return prediction
