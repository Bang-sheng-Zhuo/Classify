from cifarDataLoader import cifarDataLoader
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# CIFAR-100-dataLoad
			
class VGG16(object):
	def __init__(self, height=28, width=28, depth=1, nums_classes=10):
		self.height = height
		self.width = width
		self.depth = depth
		self.nums_classes = nums_classes
		self.prob = tf.placeholder(tf.float32, name='keep_prob')
		self.X = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.depth], name='inputs')
		self.Y = tf.placeholder(tf.float32, shape=[None, self.nums_classes], name="labels")
		self.logits = self.build_network()
		self.loss = self.loss_fn()
		self.accuracy = self.accuracy_fn()
		self.global_step = tf.get_variable("global", initializer=tf.constant(0), trainable=False)
		self.train_op = self.build_optimizer()
	
	def build_network(self, activation=None, 
					  kernel_init=None,    # tf.variance_scaling_initializer()
					  bias_init=tf.constant_initializer(0),    # tf.constant_initializer()
					  kernel_reg=tf.contrib.layers.l2_regularizer(scale=5e-4),    # tf.contrib.layers.l2_regularizer(scale=0.001)
					  bias_reg=None
					  ):
		# Block 1
		net = self.X
		for i in range(2):
			net = tf.layers.conv2d(inputs=net,
								   filters=64,
								   kernel_size=3,
								   padding='same',
								   kernel_initializer=kernel_init,
								   activation=activation,
								   bias_initializer=bias_init,
								   kernel_regularizer=kernel_reg,
								   bias_regularizer=bias_reg
								   )
			net = tf.nn.relu(net)
			# net = tf.nn.sigmoid(net)
		net = tf.layers.max_pooling2d(inputs=net,
									  pool_size=2,
									  strides=2,
									  padding='valid'
									  )
		# Block2
		for i in range(3):
			net = tf.layers.conv2d(inputs=net,
								   filters=128,
								   kernel_size=3,
								   padding='same',
								   kernel_initializer=kernel_init,
								   activation=activation,
								   bias_initializer=bias_init,
								   kernel_regularizer=kernel_reg,
								   bias_regularizer=bias_reg
								   )
			net = tf.nn.relu(net)
		net = tf.layers.max_pooling2d(inputs=net,
									  pool_size=2,
									  strides=2,
									  padding='valid'
									  )
		# Block 3
		for i in range(3):
			net = tf.layers.conv2d(inputs=net, 
								   filters=256,
								   kernel_size=3,
								   padding='same',
								   kernel_initializer=kernel_init,
								   activation=activation,
								   bias_initializer=bias_init,
								   kernel_regularizer=kernel_reg,
								   bias_regularizer=bias_reg
								   )
			net = tf.nn.relu(net)
		net = tf.layers.max_pooling2d(inputs=net,
									  pool_size=2,
									  strides=2,
									  padding='valid'
									  )
		# Block4
		"""
		for i in range(3):
			net = tf.layers.conv2d(inputs=net,
								   filters=512,
								   kernel_size=3,
								   padding='same',
								   kernel_initializer=kernel_init,
								   activation=activation,
								   bias_initializer=bias_init,
								   kernel_regularizer=kernel_reg,
								   bias_regularizer=bias_reg
								   )
			net = tf.nn.relu(net)
		net = tf.layers.max_pooling2d(inputs=net,
									  pool_size=2,
									  strides=2,
									  padding='valid'
									  )
		# Block5
		
		for i in range(3):
			net = tf.layers.conv2d(inputs=net,
								   filters=512,
								   kernel_size=3,
								   padding='same',
								   kernel_initializer=kernel_init,
								   activation=activation,
								   bias_initializer=bias_init,
								   kernel_regularizer=kernel_reg,
								   bias_regularizer=bias_reg
								   )
			net = tf.nn.relu(net)
		net = tf.layers.max_pooling2d(inputs=net,
									  pool_size=2,
									  strides=2,
									  padding='valid'
									  )
		"""
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
	
			
def rgb_mean(batch_img):
	per_img_r = []
	per_img_g = []
	per_img_b = []
	batch_size = np.shape(batch_img)[0]
	for i in range(batch_size):
		per_img_r.append(np.mean(batch_img[i,:,:,0]))
		per_img_g.append(np.mean(batch_img[i,:,:,1]))
		per_img_b.append(np.mean(batch_img[i,:,:,2]))
	r_mean = np.mean(per_img_r)
	g_mean = np.mean(per_img_g)
	b_mean = np.mean(per_img_b)
	batch_img[:,:,:,0] -= r_mean
	batch_img[:,:,:,1] -= g_mean
	batch_img[:,:,:,2] -= b_mean
	return batch_img 

def train_model():
	cifar = cifarDataLoader()
	cifar.load('../cifar-100-python/train', '../cifar-100-python/meta')
	# mnist = input_data.read_data_sets("../mnist_data/", one_hot=True)
	with tf.device('/device:GPU:0'):
		# CNN = VGG16()
		CNN = VGG16(height=32, width=32, depth=3, nums_classes=100)
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	saver = tf.train.Saver(max_to_keep=20)
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	for i in range(5000):
		# batch = mnist.train.next_batch(256)
		# batch_x = np.reshape(batch[0], [-1, 28, 28, 1])
		batch = cifar.get_next_batch(256)
		batch_x = batch[0]
		batch_x = rgb_mean(batch_x)
		batch_loss, batch_accuracy, _, train_step = sess.run([CNN.loss, CNN.accuracy, CNN.train_op, CNN.global_step],
															 feed_dict={CNN.X: batch_x, CNN.Y: batch[1], CNN.prob: 0.75})
		
		print("traing_step: ", train_step, "training_loss: ", batch_loss, "batch_accuracy: ", batch_accuracy)
		# test during train
		if train_step % 100 == 0:
			batch_test = test.get_next_batch(200)
			x_test = batch_test[0]
			test_loss, test_accuracy = sess.run([CNN.loss, CNN.accuracy], 
												feed_dict={CNN.X: x_test, CNN.Y:batch_test[1], CNN.prob: 1.0})
			print("test_step: ", train_step/100, "training_loss: ", test_loss, "batch_accuracy: ", test_accuracy)
		# save model
		if train_step % 500 == 0:
			saver.save(sess,"./checkpoint_dir/MyModel", global_step=train_step)
	
def test_model():
	test = cifarDataLoader()
	test.load('../cifar-100-python/test', '../cifar-100-python/meta')
	with tf.device('/device:GPU:0'):
		# CNN = VGG16()
		CNN = VGG16(height=32, width=32, depth=3, nums_classes=100)
		with tf.Session() as sess:
			saver=tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
			new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
			graph = tf.get_default_graph()
			total_loss = []
			total_accuracy = []
			for i in range(200):
				batch_test = test.get_next_batch(250)
				x_test = batch_test[0]
				test_loss, test_accuracy = sess.run([CNN.loss, CNN.accuracy], 
												feed_dict={CNN.X: x_test, CNN.Y:batch_test[1], CNN.prob: 1.0})
				total_loss.append(test_loss)
				total_accuracy.append(test_accuracy)
				print("test_step: ", i+1, "training_loss: ", test_loss, "batch_accuracy: ", test_accuracy)
			print("final_test, training_loss: ", np.mean(total_loss), "batch_accuracy: ", np.mean(total_accuracy))
			
def main():
	
	
	
	
		
	
	
	
if __name__ == '__main__':
	main()
			
			
			
