from ResNet50 import ResNet
from DenseNet import DenseNet
from cifarDataLoader import cifarDataLoader
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
    cifar.load('../cifar-100-python/test', '../cifar-100-python/meta')
    batch_size = 250
    data_size = cifar.X.shape[0]
    with tf.device('/device:GPU:0'):
        # CNN = VGG16()
        # CNN = ResNet(height=32, width=32, depth=3, nums_classes=100, trainable=False)
        CNN = DenseNet(height=32, width=32, depth=3, nums_classes=100, trainable=False)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir/'))
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir_densenet/'))
        total_loss = []
        total_accuracy = []
        i = 0
        for i in range(data_size//batch_size):
            batch_x = cifar.X[i*batch_size:(i+1)*batch_size]
            batch_x = batch_x.astype(np.float32)
            batch_y = cifar.Y[i*batch_size:(i+1)*batch_size]
            i += 1
            batch_y = one_hot_encode(batch_y, 100)
            batch_loss, batch_accuracy = sess.run([CNN.loss, CNN.accuracy],
                                                  feed_dict={CNN.X: batch_x, CNN.Y: batch_y})
            print("test_step: ", i, " test_loss: ", batch_loss, " test_accuracy: ", batch_accuracy)
            total_loss.append(batch_loss)
            total_accuracy.append(batch_accuracy)
        print("Finally, final_test_loss:", np.mean(total_loss), " final_test_accuracy:", np.mean(total_accuracy))


if __name__ == '__main__':
    main()