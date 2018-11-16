from cifarDataLoader import cifarDataLoader
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)
# DenseNet

class DenseNet(object):
	def __init__(self, height=28, width=28, depth=1, nums_classes=10, trainable=True, lr=0.001, theta=0.5):
		self.height = height
		self.width = width
		self.depth = depth
		self.nums_classes = nums_classes
		self.is_training = trainable
		self.lr = lr
		self.theta = theta
		self.X = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.depth], name='inputs')
		self.Y = tf.placeholder(tf.float32, shape=[None, self.nums_classes], name="labels")
		self.logits = self.build_network()
		self.loss = self.loss_fn()
		self.accuracy = self.accuracy_fn()
		self.global_step = tf.get_variable("global", initializer=tf.constant(0), trainable=False)
		self.train_op = self.build_optimizer()
	
	def bn_activation_conv(self, input_tensor, depth, size, stride, padding='same', activation=tf.nn.relu, is_training=True, bn=True, bn_scale=True):
		net = input_tensor
		if bn:
			net = tf.layers.batch_normalization(inputs=net,
												center=True,
												scale=bn_scale,
												momentum=0.96,
												training=is_training
												)
		if activation is not None:
			net = activation(net)
		net = tf.layers.conv2d(inputs=input_tensor, filters=depth, kernel_size=size, strides=stride, padding=padding)
		return net
	
	def transition_layer(self, input_tensor, theta, size, stride, padding='same', is_training=True, bn=True, bn_scale=True):
		# batch_normalization->1x1 conv->2x2 ave_pooling
		input_depth = slim.utils.last_dimension(input_tensor.get_shape(), min_rank=4)
		net = self.bn_activation_conv(input_tensor, depth=int(theta*input_depth), size=size, stride=stride, padding=padding, is_training=self.is_training, bn=bn, bn_scale=bn_scale)
		net = tf.layers.average_pooling2d(inputs=net,
									  pool_size=2,
									  strides=2,
									  padding='same'
									  )
		return net
	
	def dense_block(self, input_tensor, depth, activation=tf.nn.relu, is_training=True, bn=True, bn_scale=True):
		net = self.bn_activation_conv(input_tensor, depth=depth, size=1, stride=1, padding='same', activation=activation, is_training=is_training, bn=bn, bn_scale=True)
		net = self.bn_activation_conv(net, depth=depth, size=3, stride=1, padding='same', activation=activation, is_training=is_training, bn=bn, bn_scale=True)
		return net
	
	def build_network(self, activation=tf.nn.relu,
					  kernel_init=None,    # tf.variance_scaling_initializer()
					  bias_init=None,    # tf.constant_initializer()
					  kernel_reg=None,    # tf.contrib.layers.l2_regularizer(scale=0.001)
					  bias_reg=None
					  ):
		# Preprocess Block, input_size = 32
		net = self.X
		# net = self.bn_activation_conv(net, depth=64, size=7, stride=2, padding='same', is_training=self.is_training, bn=True, bn_scale=True)
		net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=7, strides=2, padding='same')
		net = tf.nn.relu(net)
		net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, padding='same')
		# pdb.set_trace()
		# Dense Block1, k = 32, l = 6, input_size = 8
		tmp = net
		for i in range(6):
			net = self.dense_block(input_tensor=tmp, depth=32, is_training=self.is_training)
			tmp = tf.concat(values=[tmp, net], axis=3)
		# Transition Layer
		# pdb.set_trace()
		tmp = self.transition_layer(input_tensor=tmp, theta=self.theta, size=1, stride=1, padding='same', is_training=self.is_training, bn=True, bn_scale=True)
		# Dense Block2, k = 32, l = 12, input_size = 4
		# pdb.set_trace()
		for i in range(12):
			net = self.dense_block(input_tensor=tmp, depth=32, is_training=self.is_training)
			tmp = tf.concat(values=[tmp, net], axis=3)
		# tmp = self.transition_layer(input_tensor=tmp, theta=self.theta, size=1, stride=1, padding='same', is_training=True, bn=True, bn_scale=True)
		# Dense Block3, k = 32, l = 48, input_size = 2
		# for i in range(32)
		# net = self.dense_block(input_tensor=tmp, depth=32, is_training=self.is_training)
			# tmp = tf.concat(values=[tmp, net], axis=3)
		# Postprocess Block, input_size = 4
		# pdb.set_trace()
		net = tf.layers.average_pooling2d(inputs=tmp,
										  pool_size=2,
										  strides=2,
										  padding='same'
										  )
		net = tf.layers.flatten(net)
		net = tf.layers.dense(inputs=net,
							  units=self.nums_classes,
							  activation=None
							  )
		# pdb.set_trace()
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
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			# self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
			# self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.025, momentum=0.9, use_nesterov=True)
			gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
			gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5) for gradient in gradients]
			train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
			return train_op

	
	
if __name__ == '__main__':
	main()			