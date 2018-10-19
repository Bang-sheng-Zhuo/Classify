import numpy as np
import pickle
import os
import pdb
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


# CIFAR-100-dataLoad
class cifarDataLoader():
    def __init__(self):
        self.fine_label_names = None
        self.coarse_label_names = None
        self.X = None
        self.Y = None
        self.YY = None

    def load_CIFAR_Data(self, filename):
        with open(filename, 'rb') as file:
            dataset = pickle.load(file, encoding='bytes')
            # print(dataset.keys())
            X = dataset[b'data']            # 数据
            Y = dataset[b'fine_labels']     # 小范围标签
            YY = dataset[b'coarse_labels']  # 大范围标签
            self.X = np.reshape(X, (-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype("uint8")
            self.Y = np.array(Y)
            self.YY = np.array(YY)

    def load_CIFAR_labels(self, filename):
        with open(filename, 'rb') as file:
            labels = pickle.load(file, encoding='bytes')
            # print(labels.keys())
            self.fine_label_names = labels[b'fine_label_names']
            self.coarse_label_names = labels[b'coarse_label_names']

    def load(self, imgfile, labelfile):
        self.load_CIFAR_Data(imgfile)
        self.load_CIFAR_labels(labelfile)

    def get_next_batch(self, batch_size, one_hot=True):
        idx = np.random.choice(len(self.X), batch_size)
        # idx = range(50)
        batch_x = self.X[idx]
        batch_y = self.Y[idx]
        if one_hot:
            batch_onehot = np.eye(100)[batch_y]
            return batch_x.astype(np.float32), batch_onehot.astype(np.float32)
        else:
            return batch_x.astype(np.float32), batch_y.astype(np.float32)

    def show(self):
        example_nums = self.X.shape[0]
        for i in range(example_nums):
            img = self.X[i]
            print("fine_label:", self.fine_label_names[self.Y[i]],
                  " coarse_label:", self.coarse_label_names[self.YY[i]])
            plt.imshow(img)
            plt.show()


# BaseModel
class ClassifyModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_num(self):
        return self.num_classes

    def prepocess(self, raw_input):
        pass

    def predict(self, input):
        pass

    def postprocess(self, output):
        pass


# VGG16 network
class VGG16(ClassifyModel):
    def __init__(self, is_training=True, num_classes=1000):
        self.is_training = is_training
        self.num_classes = num_classes

    def get_num(self):
        return self.num_classes

    def prepocess(self, raw_input):
        inputs = raw_input / 255
        return inputs

    def predict(self, net):
        net = self.Block1(net)
        net = self.Block2(net)
        # net = self.Block3(net)
        # net = self.Block4(net)
        # net = self.Block5(net)
        # pdb.set_trace()
        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net,
                              units=4096,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.variance_scaling_initializer(),
                              bias_initializer=tf.zeros_initializer(),
                              name='fc1')
        net = tf.layers.dense(net, 4096, activation=tf.nn.relu,
                              kernel_initializer=tf.variance_scaling_initializer(),
                              name='fc2')
        net = tf.layers.dense(net, 100, activation=tf.nn.relu,
                              kernel_initializer=tf.variance_scaling_initializer(),
                              name='fc3')
        # net = tf.nn.softmax(net)
        return net

    def postprocess(self, output):
        pass

    def loss(self, predict, labels):
        pass

    def Block1(self, net):
        for i in range(2):
            net = tf.layers.conv2d(inputs=net,
                                   filters=64,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   use_bias=True,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='block1_conv%d' % (i+1))
        net = tf.layers.max_pooling2d(inputs=net,
                                      pool_size=2,
                                      strides=2,
                                      padding='valid',
                                      name='maxpool1')
        return net

    def Block2(self, net):
        for i in range(2):
            net = tf.layers.conv2d(net, 128, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   name='block2_conv%d' % (i+1))
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='maxpool2')
        return net

    def Block3(self, net):
        for i in range(3):
            net = tf.layers.conv2d(net, 256, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   name='block3_conv%d' % (i+1))
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='maxpool3')
        return net

    def Block4(self, net):
        for i in range(3):
            net = tf.layers.conv2d(net, 512, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   name='block4_conv%d' % (i+1))
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='maxpool4')
        return net

    def Block5(self, net):
        for i in range(3):
            net = tf.layers.conv2d(net, 512, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   name='block5_conv%d' % (i+1))
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='maxpool5')
        return net

    def loss(self):
        logits = self.predict()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.int_logits, labels=self.y)


def loss_fn(logits, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(loss)


def build_model(loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, 50) for gradient
                     in gradients]
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op


def main():
    cifar = cifarDataLoader()
    cifar.load('cifar-100-python/test', 'cifar-100-python/meta')
    # cifar.load_CIFAR_Data('cifar-100-python/train', onehot=True)
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='inputs')
    labels = tf.placeholder(tf.float32, shape=[None, 100])
    # Build Graph
    vgg = VGG16(is_training=True, num_classes=100)
    processed = vgg.prepocess(inputs)
    logits = vgg.predict(processed)
    loss = loss_fn(logits, labels)
    train_op = build_model(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_x, batch_y = cifar.get_next_batch(100)
        batch_loss, _ = sess.run([loss, train_op], feed_dict={inputs: batch_x, labels: batch_y})
        if i % 20 == 0:
            # train_loss = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step ", i, "train_loss ", batch_loss)
    """
    # pdb.set_trace()
    # logits = tf.reshape(logits, (-1, 100))
    # cross_entropy_loss
    # cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    batch_loss = tf.reduce_sum(cross_entropy)
    # nums of correct predictions
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # global_step = tf.Variable(0, trainable=False)
    # lr = tf.train.exponential_decay(0.005, global_step, 150, 0.9)
    optimizer = tf.train.AdamOptimizer(0.01)
    train_step = optimizer.minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
     # session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            batch_img, batch_y = cifar.get_next_batch(50)
            train_dict = {inputs: batch_img, labels: batch_y}
            sess.run(train_step, feed_dict=train_dict)
            if i % 20 == 0:
                accu = accuracy.eval(session=sess, feed_dict=train_dict)
                loss = batch_loss.eval(session=sess, feed_dict=train_dict)
                print("step %d, train_accuracy %g" % (i, accu), ", train_loss: ", loss)
    """


if __name__ == '__main__':
    main()

