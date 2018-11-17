from VGG16 import VGG16
from ResNet50 import ResNet
from DenseNet import DenseNet
from cifarDataLoader import cifarDataLoader
from data_augmentation import random_erase
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def one_hot_encode(label, nums_classes):
    onehot = np.eye(nums_classes)[label]
    return onehot.astype(np.float32)

def main():
    cifar = cifarDataLoader()
    cifar.load('../cifar-100-python/train', '../cifar-100-python/meta')
    # mnist = input_data.read_data_sets("../mnist_data/", one_hot=True)
    train_generator = ImageDataGenerator(rotation_range=20,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         rescale=1.,
                                         zoom_range=0.2,
                                         fill_mode='nearest',
                                         cval=0)
    batch_input = train_generator.flow(cifar.X, cifar.Y, 200)
    with tf.device('/device:GPU:0'):
        # CNN = ResNet(height=32, width=32, depth=3, nums_classes=100, lr=0.00025)
        CNN = DenseNet(height=32, width=32, depth=3, nums_classes=100, lr=0.001, prob=0.2)
        # CNN = VGG16()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir/'))
        # saver.restore(sess, './checkpoint_dir/MyModel-12000')
        # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir_densenet/'))
        # saver.restore(sess, './checkpoint_dir_densenet/MyModel-20000')
        # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir_vgg/'))
        # saver.restore(sess, './checkpoint_dir_vgg/MyModel-5000')
        for x, y in batch_input:
            batch_x, batch_y = batch_input.next()
            for i in range(batch_x.shape[0]):
                batch_x[i] = random_erase(batch_x[i], prob=0.5, min=0.1, max=0.4)
            batch_y = one_hot_encode(batch_y, 100)
            batch_loss, batch_accuracy, _, train_step = sess.run([CNN.loss, CNN.accuracy, CNN.train_op, CNN.global_step],
                                                                feed_dict={CNN.X: batch_x, CNN.Y: batch_y})
            if train_step % 25 == 0:
                print("traing_step: ", train_step, "training_loss: ", batch_loss, "batch_accuracy: ", batch_accuracy)
            if train_step % 1000 == 0:
                # saver.save(sess, './checkpoint_dir/MyModel', global_step=train_step)
                saver.save(sess, './checkpoint_dir_densenet/MyModel', global_step=train_step)
                # saver.save(sess, './checkpoint_dir_vgg/MyModel', global_step=train_step)
            if train_step == 30000:
                break


if __name__ == '__main__':
    main()