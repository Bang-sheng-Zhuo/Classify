# assemble test
from ResNet50 import ResNet
from DenseNet import DenseNet
from cifarDataLoader import cifarDataLoader
import numpy as np
import pdb
import tensorflow as tf
from scipy import stats
from data_augmentation import random_crop, random_erase, one_hot_encode

def main():
    cifar = cifarDataLoader()
    cifar.load('../cifar-100-python/test', '../cifar-100-python/meta')
    batch_size = 50
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
        total_assemble_ac = []
        total_accuracy = []
        for i in range(data_size//batch_size):
            batch_x = cifar.X[i*batch_size:(i+1)*batch_size]
            batch_x = np.repeat(batch_x, 4, axis=0)
            for j in range(batch_size):
                batch_x[j*4+1] = random_crop(batch_x[j*4+1], padding=0, is_flip=True, prob=0., is_crop=False)
                batch_x[j*4+2] = random_crop(batch_x[j*4+2], padding=4, is_flip=False, is_crop=True)
                batch_x[j*4+3] = random_crop(batch_x[j*4+3], padding=4, is_flip=True, prob=0., is_crop=True)
            batch_x = np.asarray(batch_x, np.float32) 
            batch_y = cifar.Y[i*batch_size:(i+1)*batch_size]
            batch_y = one_hot_encode(batch_y, 100)
            # pdb.set_trace()
            batch_y = np.repeat(batch_y, 4, axis=0)
            batch_pred, batch_accuracy = sess.run([CNN.pred, CNN.accuracy],
                                                      feed_dict={CNN.X: batch_x, CNN.Y: batch_y})
            ensemble_labels = []
            for j in range(batch_size):
                ensemble_labels.append(stats.mode(batch_pred[4*j:4*(j+1)])[0][0])
            ensemble_labels = np.asarray(ensemble_labels, np.int32)
            # pdb.set_trace()
            nums_correct = np.equal(ensemble_labels, cifar.Y[i*batch_size:(i+1)*batch_size])
            batch_ensemble_ac = np.mean(nums_correct)
            print("test_step: ", i+1, " test_loss: ", batch_ensemble_ac, " test_accuracy: ", batch_accuracy)
            total_ensemble_ac.append(batch_ensemble_ac)
            total_accuracy.append(batch_accuracy)
        print("Finally, final_ensemble_ac:", np.mean(total_ensemble_ac), " final_test_accuracy:", np.mean(total_accuracy))


if __name__ == '__main__':
    main()