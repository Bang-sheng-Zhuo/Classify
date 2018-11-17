import numpy as np
import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim

tf.logging.set_verbosity(tf.logging.INFO)


class VGG16(object):
    def __init__(self, height=28, width=28, depth=1, nums_classes=10, lr=0.001, prob=0.2, bn=True):
        self.height = height
        self.width = width
        self.depth = depth
        self.nums_classes = nums_classes
        self.is_training = trainable
        self.lr = lr
        self.bn = bn
        self.prob = prob
        self.X = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.depth], name='inputs')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.nums_classes], name="labels")
        self.logits = self.build_network()
        self.loss = self.loss_fn()
        self.accuracy = self.accuracy_fn()
        self.global_step = tf.get_variable("global", initializer=tf.constant(0), trainable=False)
        self.train_op = self.build_optimizer()

    def conv_bn_relu(self, input_tensor, depth, size, stride, padding='same', 
                     activation=tf.nn.relu, is_training=True, bn=True, bn_scale=True):
        net = tf.layers.conv2d(inputs=input_tensor, filters=depth, kernel_size=size, strides=stride, padding=padding)
        if bn:
            net = tf.layers.batch_normalization(inputs=net,
                                                center=True,
                                                scale=bn_scale,
                                                momentum=0.96,
                                                training=is_training
                                                )
        if activation is not None:
            net = activation(net)
        return net

    def build_network(self, activation=None, 
                      kernel_init=None,    # tf.variance_scaling_initializer()
                      bias_init=None,    # tf.constant_initializer()
                      kernel_reg=None,    # tf.contrib.layers.l2_regularizer(scale=0.001)
                      bias_reg=None
                      ):
        net = self.X
        # Block 1
        for i in range(2):
            net = self.conv_bn_relu(input_tensor=net, depth=64, size=3, stride=1, 
                                    is_training=self.is_training, bn=self.bn)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2, padding='same')
        # Block2
        for i in range(3):
            net = self.conv_bn_relu(input_tensor=net, depth=128, size=3, stride=1, 
                                    is_training=self.is_training, bn=self.bn)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2, padding='same')
        # Block 3
        for i in range(3):
            net = self.conv_bn_relu(input_tensor=net, depth=256, size=3, stride=1, 
                                    is_training=self.is_training, bn=self.bn)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2, padding='same')
        # Block4
        for i in range(3):
            net = self.conv_bn_relu(input_tensor=net, depth=512, size=3, stride=1, 
                                    is_training=self.is_training, bn=self.bn)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2, padding='same')
        # Block5
        for i in range(3):
            net = self.conv_bn_relu(input_tensor=net, depth=512, size=3, stride=1, 
                                    is_training=self.is_training, bn=self.bn)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2, padding='same')
        # fully-connected layers Block
        net = tf.layers.flatten(net)
        for i in range(2):
            net = tf.layers.dense(inputs=net,
                                  units=1024,
                                  activation=activation,
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_init
                                  )
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, self.prob)
        net = tf.layers.dense(inputs=net,
                              units=self.nums_classes,
                              activation=activation,
                              kernel_initializer=kernel_init,
                              bias_initializer=bias_init
                              )
        return net

    def loss_fn(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        return tf.reduce_mean(loss)

    def accuracy_fn(self):
        nums_correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        return tf.reduce_mean(tf.cast(nums_correct, "float"))

    def build_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.025, momentum=0.9, use_nesterov=True)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5) for gradient in gradients]
            train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            return train_op


if __name__ == '__main__':
    main()

    
    
