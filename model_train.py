from VGG16 import VGG16
from ResNet50 import ResNet
from DenseNet import DenseNet
from cifarDataLoader import cifarDataLoader
from data_augmentation import random_crop, random_erase, one_hot_encode
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    cifar = cifarDataLoader()
    cifar.load('../cifar-100-python/train', '../cifar-100-python/meta')
    train_generator = ImageDataGenerator(# rotation_range=30,
                                         # width_shift_range=0.2,
                                         # height_shift_range=0.2,
                                         rescale=1.,
                                         # zoom_range=0.2,
                                         # horizontal_flip=True,
                                         fill_mode='nearest',
                                         cval=0)
    batch_input = train_generator.flow(cifar.X, cifar.Y, 64)
    with tf.device('/device:GPU:1'):
        # CNN = ResNet(height=32, width=32, depth=3, nums_classes=100, lr=0.1)
        CNN = DenseNet(height=32, width=32, depth=3, nums_classes=100, lr=0.001, prob=0.)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess, tf.train.latest_checkpoint('./ckpt_resnet_rp/'))
        # saver.restore(sess, './ckpt_resnet_rp/MyModel-50000')
        # saver.restore(sess, tf.train.latest_checkpoint('./ckpt_densenet_rp/'))
        saver.restore(sess, './ckpt_densenet_rp/MyModel-32000')
        for x, y in batch_input:
            batch_x, batch_y = batch_input.next()
            for i in range(batch_x.shape[0]):
                batch_x[i] = random_crop(batch_x[i], padding=4, is_flip=True, prob=0.5, is_crop=True)
                batch_x[i] = random_erase(batch_x[i], prob=0.5, min=0.1, max=0.4)
            batch_y = one_hot_encode(batch_y, 100)
            batch_loss, batch_accuracy, _, train_step = sess.run([CNN.loss, CNN.accuracy, CNN.train_op, CNN.global_step],
                                                                feed_dict={CNN.X: batch_x, CNN.Y: batch_y})
            if train_step % 50 == 0:
                print("traing_step: ", train_step, "training_loss: ", batch_loss, "batch_accuracy: ", batch_accuracy)
            if train_step % 2000 == 0:
                # saver.save(sess, './ckpt_resnet_rp/MyModel', global_step=train_step)
                saver.save(sess, './ckpt_densenet_rp/MyModel', global_step=train_step)
            if train_step == 100000:
                break


if __name__ == '__main__':
    main()
