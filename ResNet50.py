from cifarDataLoader import cifarDataLoader
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import rotate

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)
# ResNet-50

class ResNet(object):
	def __init__(self, height=28, width=28, depth=1, nums_classes=10, trainable=True, lr=0.001):
		self.height = height
		self.width = width
		self.depth = depth
		self.nums_classes = nums_classes
		self.is_training = trainable
		self.lr = lr
		# self.prob = tf.placeholder(tf.float32, name='keep_prob')
		self.X = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.depth], name='inputs')
		self.Y = tf.placeholder(tf.float32, shape=[None, self.nums_classes], name="labels")
		self.logits = self.build_network()
		self.loss = self.loss_fn()
		self.accuracy = self.accuracy_fn()
		self.global_step = tf.get_variable("global", initializer=tf.constant(0), trainable=False)
		self.train_op = self.build_optimizer()
	
	def conv_bn_activation(self, input_tensor, depth, size, stride, padding='same', activation=tf.nn.relu, is_training=True, bn=True, bn_scale=True):
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
	
	def residual_connect(self, input_tensor, output_depth, is_training=True):
		net = input_tensor
		input_depth = slim.utils.last_dimension(input_tensor.get_shape(), min_rank=4)
		if input_depth == output_depth:
			short_cut = input_tensor
		else:
			short_cut = self.conv_bn_activation(input_tensor=net, depth=output_depth, size=1, stride=1, is_training=self.is_training)
		net = self.conv_bn_activation(input_tensor=net, depth=output_depth/4, size=1, stride=1, is_training=self.is_training)
		net = self.conv_bn_activation(input_tensor=net, depth=output_depth/4, size=3, stride=1, is_training=self.is_training)
		net = self.conv_bn_activation(input_tensor=net, depth=output_depth, size=1, stride=1, is_training=self.is_training)
		net = net + short_cut
		return net
		
	def downsampling(self, input_tensor):
		net = tf.layers.max_pooling2d(inputs=input_tensor,
									  pool_size=1,
									  strides=2,
									  padding='same'
									  )
		return net
	
	def build_network(self, activation=tf.nn.relu,
					  kernel_init=None,    # tf.variance_scaling_initializer()
					  bias_init=None,    # tf.constant_initializer()
					  kernel_reg=None,    # tf.contrib.layers.l2_regularizer(scale=0.001)
					  bias_reg=None
					  ):
		net = self.X
		# 32x32
		net = self.conv_bn_activation(input_tensor=net, depth=64, size=7, stride=2, is_training=self.is_training)
		# 16x16
		net = tf.layers.max_pooling2d(inputs=net,
									  pool_size=3,
									  strides=2,
									  padding='same'
									  )
		# Block1 8x8
		for i in range(3):
			net = self.residual_connect(input_tensor=net, output_depth=256, is_training=self.is_training)
		net = self.downsampling(input_tensor=net)
		# Block2 4x4
		for i in range(4):
			net = self.residual_connect(input_tensor=net, output_depth=512, is_training=self.is_training)
		# net = self.downsampling(input_tensor=net)
		# Block3 2x2
		# for i in range(4):
			# net = self.residual_connect(input_tensor=net, output_depth=1024, is_training=self.is_training)
		# net = self.downsampling(input_tensor=net)
		# # Block4 1x1
		# for i in range(3):
			# net = self.residual_connect(input_tensor=net, output_depth=2048, is_training=self.is_training)
		# ave-pooling
		net = tf.layers.average_pooling2d(inputs=net,
										  pool_size=2,
										  strides=2,
										  padding='same'
										  )
		net = tf.layers.flatten(net)
		net = tf.layers.dense(inputs=net,
							  units=self.nums_classes,
							  activation=None
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
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			# self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
			# self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.025, momentum=0.9, use_nesterov=True)
			gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
			gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5) for gradient in gradients]
			train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
			return train_op

	
	
if __name__ == '__main__':
	main()			
