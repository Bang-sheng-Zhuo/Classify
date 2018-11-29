from ResNet50 import ResNet
from DenseNet import DenseNet
from cifarDataLoader import cifarDataLoader
import numpy as np
import pdb
import tensorflow as tf
from data_augmentation import random_crop, random_erase, one_hot_encode


def main():
    cifar = cifarDataLoader()
    cifar.load('../cifar-100-python/test', '../cifar-100-python/meta')
    batch_size = 250
    data_size = cifar.X.shape[0]
    with tf.device('/device:GPU:1'):
        # CNN = ResNet(height=32, width=32, depth=3, nums_classes=100, trainable=False)
        CNN = DenseNet(height=32, width=32, depth=3, nums_classes=100, trainable=False)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        # saver.restore(sess, tf.train.latest_checkpoint('./ckpt_resnet_rp/'))
        # saver.restore(sess, './ckpt_resnet_rp/MyModel-100000')
        # saver.restore(sess, tf.train.latest_checkpoint('./ckpt_densenet_rp/'))
        saver.restore(sess, './ckpt_densenet_rp/MyModel-36000')
        total_loss = []
        total_accuracy = []
        for i in range(data_size//batch_size):
            batch_x = cifar.X[i*batch_size:(i+1)*batch_size]
            batch_x = batch_x.astype(np.float32)
            for j in range(batch_x.shape[0]):
                batch_x[j] = random_crop(batch_x[j], padding=4, is_flip=False, is_crop=False)
                # batch_x[i] = random_erase(batch_x[i], prob=0.5, min=0.1, max=0.4)
            batch_y = cifar.Y[i*batch_size:(i+1)*batch_size]
            batch_y = one_hot_encode(batch_y, 100)
            batch_loss, batch_accuracy = sess.run([CNN.loss, CNN.accuracy],
                                                  feed_dict={CNN.X: batch_x, CNN.Y: batch_y})
            print("test_step: ", i+1, " test_loss: ", batch_loss, " test_accuracy: ", batch_accuracy)
            total_loss.append(batch_loss)
            total_accuracy.append(batch_accuracy)
        print("Finally, final_test_loss:", np.mean(total_loss), " final_test_accuracy:", np.mean(total_accuracy))


if __name__ == '__main__':
    main()
