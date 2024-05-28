# 权重网络模型
import tensorflow as tf
# import tensorflow.contrib.layers
from tensorflow.contrib import layers


class ConfNet(object):
    # def __init__(self):

    def conf_model(self, net_in, name):
        net_conv_1 = self.conv_layer(name=name + 'conf_conv_1', inputs=net_in, filter_shape=[3, 3, 3, 64],
                                     strides=[1, 2, 2, 1], padding='VALID')
        net_relu_1 = self.relu_layer(name=name + 'conf_relu_1', inputs=net_conv_1)
        net_pool_1 = self.pool_layer(name=name + 'conf_pool_1', inputs=net_relu_1, padding='VALID')

        net_fire_2 = self.fire_module(name=name + 'conf_fire_2', inputs=net_pool_1, s1=16, e1=64, e3=64)

        net_fire_3 = self.fire_module(name=name + 'conf_fire_3', inputs=net_fire_2, s1=16, e1=64, e3=64)
        net_pool_3 = self.pool_layer(name=name + 'conf_pool_3', inputs=net_fire_3, padding='SAME')

        net_fire_4 = self.fire_module(name=name + 'conf_fire_4', inputs=net_pool_3, s1=32, e1=128, e3=128)

        net_fire_5 = self.fire_module(name=name + 'conf_fire_5', inputs=net_fire_4, s1=32, e1=128, e3=128)
        net_pool_5 = self.pool_layer(name=name + 'conf_pool_5', inputs=net_fire_5, padding='SAME')

        net_fire_6 = self.fire_module(name=name + 'conf_fire_6', inputs=net_pool_5, s1=48, e1=192, e3=192)

        net_fire_7 = self.fire_module(name=name + 'conf_fire_7', inputs=net_fire_6, s1=48, e1=192, e3=192)

        net_fire_8 = self.fire_module(name=name + 'conf_fire_8', inputs=net_fire_7, s1=64, e1=256, e3=256)

        net_pool_extra = self.extra_pool_layer(name=name + 'conf_pool_8', inputs=net_fire_8, padding='SAME')

        # net_out = net_pool_extra

        net_fire_9 = self.fire_module(name=name + 'conf_fire_9', inputs=net_pool_extra, s1=64, e1=256, e3=256)
        num_channel_fire_9 = net_fire_9.get_shape().as_list()[3]

        net_conv_10 = self.conv_layer(name=name + 'conf_conv_10', inputs=net_fire_9,
                                      filter_shape=[1, 1, num_channel_fire_9, 1000], strides=[1, 1, 1, 1],
                                      padding='VALID')
        net_relu_10 = self.relu_layer(name=name + 'conf_relu_10', inputs=net_conv_10)
        net_pool_10 = self.pool_layer(name=name + 'conf_pool_10', inputs=net_relu_10, padding='SAME')

        # ======================================================================================================

        net_out = self.conv_layer(name=name + 'conf_net_out', inputs=net_pool_10,
                                  filter_shape=[1, 1, 1000, 1], strides=[1, 1, 1, 1],
                                  padding='SAME')

        return net_out

    def conv_layer(self, name, inputs, filter_shape, strides, padding, b_value=None):
        with tf.variable_scope(name):
            tf.cast(inputs, tf.float32)
            filters = self.get_weight(name='weight_' + name, shape=filter_shape)
            conv_out = tf.nn.conv2d(input=inputs, filter=filters, strides=strides, padding=padding)

            if b_value:
                bias = self.get_bias(name='bias_' + name, shape=[filter_shape[-1]], value=b_value)
                conv_out = tf.nn.bias_add(value=conv_out, bias=bias)
            else:
                conv_out = conv_out

            return conv_out

    @staticmethod
    def relu_layer(name, inputs):
        relu_out = tf.nn.relu(name=name, features=inputs)
        return relu_out

    @staticmethod
    def pool_layer(name, inputs, padding, pooling_type='max'):
        if pooling_type == max:
            pool = tf.nn.max_pool(name=name, value=inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
        else:
            pool = tf.nn.avg_pool(name=name, value=inputs, ksize=[1, 14, 14, 1], strides=[1, 1, 1, 1], padding=padding)

        return pool

    def fire_module(self, name, inputs, s1, e1, e3):
        with tf.variable_scope(name):
            num_channel = inputs.get_shape().as_list()[3]

            s_conv_1 = self.conv_layer(name=name + '_s_conv_1', inputs=inputs, filter_shape=[1, 1, num_channel, s1],
                                       strides=[1, 1, 1, 1], padding='SAME', b_value=0.1)
            s_relu_1 = self.relu_layer(name=name + '_s_relu_1', inputs=s_conv_1)

            e_conv_1 = self.conv_layer(name=name + '_e_conv_1', inputs=s_relu_1, filter_shape=[1, 1, s1, e1],
                                       strides=[1, 1, 1, 1], padding='SAME', b_value=0.1)

            e_conv_3 = self.conv_layer(name=name + '_e_conv_3', inputs=s_conv_1, filter_shape=[3, 3, s1, e3],
                                       strides=[1, 1, 1, 1], padding='SAME', b_value=0.1)

            concat = tf.concat(values=[e_conv_1, e_conv_3], axis=3)
            concat = self.relu_layer(name=name + 'concat', inputs=concat)

            return concat

    @staticmethod
    def extra_pool_layer(name, inputs, padding):
        out = tf.nn.max_pool(name=name, value=inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
        return out

    @staticmethod
    def get_weight(name, shape, init='xavier'):
        if init == 'variance':
            w_initial = tf.get_variable(name=name, shape=shape, initializer=tf.variance_scaling_initializer())

        elif init == 'xavier':
            w_initial = tf.get_variable(name=name, shape=shape, initializer=layers.xavier_initializer())

        else:
            w_initial = tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=init))

        # w_initial = tf.Variable(name=name, initial_value=tf.random_normal(shape=shape, stddev=0.1))
        return w_initial

    @staticmethod
    def get_bias(name, shape, value=0.1):
        b_initial = tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=value))
        return b_initial

