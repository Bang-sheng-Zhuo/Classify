# basic cell for CNN
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def conv_bn_relu_dropout(self, input_tensor, depth, size, stride, padding='same', 
                         activation=tf.nn.relu, is_training=True, bn=True, bn_scale=True, prob=0.2):
    net = input_tensor
    net = tf.layers.conv2d(inputs=net, filters=depth, kernel_size=size, strides=stride, padding=padding)
    if bn:
        net = tf.layers.batch_normalization(inputs=net,
                                            center=True,
                                            scale=bn_scale,
                                            momentum=0.99,
                                            training=is_training
                                            )
    if activation is not None:
        net = activation(net)
    net = tf.layers.dropout(inputs=net, rate=prob, training=is_training)
    return net

def bn_relu_conv_dropout(self, input_tensor, depth, size, stride, padding='same', 
                         activation=tf.nn.relu, is_training=True, bn=True, bn_scale=True, prob=0.2):
    net = input_tensor
    if bn:
        net = tf.layers.batch_normalization(inputs=net,
                                            center=True,
                                            scale=bn_scale,
                                            momentum=0.99,
                                            training=is_training
                                            )
    if activation is not None:
        net = activation(net)
    net = tf.layers.dropout(inputs=net, rate=prob, training=is_training)
    net = tf.layers.conv2d(inputs=net, filters=depth, kernel_size=size, strides=stride, padding=padding)
    return net