import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------initialize global var
training_num = 0
centroids_num = 0
train = None
valid = None
data_dim = 0
centroids = None
cluster = None
centroid = None


def build(inputFile, K):
	global cluster, centroid
	global train, valid, centroids, training_num, data_dim, centroids_num
	data = np.float32(np.load(inputFile))
	data = (data - data.mean()) / data.std()
	data_num, data_dim = data.shape
	centroids_num = K
	training_num = int(2. / 3 * data_num)
	train = data[:training_num]
	valid = data[training_num:]
	centroids = tf.truncated_normal(shape=[centroids_num, data_dim])
	cluster = tf.placeholder(tf.float32, shape=[None, data_dim])
	centroid = tf.Variable(tf.convert_to_tensor(centroids, dtype=tf.float32))
	tf_train_dist = euclidean_dist(cluster, centroid, training_num, centroids_num)
	tf_train_min_index = tf.argmin(tf_train_dist, dimension=1)
	tf_train_loss = tf.reduce_sum(tf.reduce_min(euclidean_dist(cluster, centroid, training_num, centroids_num), 1, keep_dims=True))
	tf_train_opt = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.9, beta2=0.9, epsilon=1e-4).minimize(tf_train_loss)
	tf_valid_dist = euclidean_dist(cluster, centroid, (data_num - training_num), centroids_num)
	tf_valid_loss = tf.reduce_sum(
		tf.reduce_min(euclidean_dist(cluster, centroid, (data_num - training_num), centroids_num), 1, keep_dims=True))
	return tf_train_min_index, tf_train_loss, tf_train_opt, tf_valid_loss


def model_train(K):
	graph = tf.Graph()
	fileRead = 'data2D.npy'
	with graph.as_default():
		tf_train_min_index, tf_train_loss, tf_train_opt, tf_valid_loss = build(fileRead, K)
	with tf.Session(graph=graph) as sess:
		init = tf.global_variables_initializer()
		init.run()
		epoch = 1000
		train_losses = []
		valid_losses = []
		for i in range(epoch):
			feed_dict = {cluster: train}
			opt = sess.run([tf_train_opt], feed_dict=feed_dict)
			train_loss, train_min_index, centroids = sess.run([tf_train_loss, tf_train_min_index, centroid], feed_dict=feed_dict)
			feed_dict = {cluster: valid}
			valid_loss = sess.run(tf_valid_loss, feed_dict=feed_dict)
			train_losses.append(train_loss)
			valid_losses.append(valid_loss)
		epoch_update_plot(np.array(range(epoch)), train_losses, K, 'training loss')
		epoch_update_plot(np.array(range(epoch)), valid_losses, K, 'validation loss')
	return train_loss, valid_loss



def epoch_update_plot(updates, loss, K, data_type):
	assert len(updates) == len(loss)
	plt.figure(1)
	plt.plot(updates, loss)
	plt.ylabel('loss'), plt.xlabel('updates #')
	plt.title(data_type + ': ' + str(K) + ' clusters')
	plt.show()


def euclidean_dist(a, b, x, y):
	a_reduce = tf.reduce_sum(tf.square(a), 1)
	b_reduce = tf.reduce_sum(tf.square(b), 1)
	matmul = tf.matmul(a, b, transpose_b=True)
	returnVal = -2 * matmul + b_reduce
	returnVal = tf.transpose(returnVal)
	tf_data_2 = tf.reshape(a_reduce, [1, x])
	returnVal += tf_data_2
	returnVal = tf.transpose(returnVal)
	return returnVal


if __name__ == '__main__':
	N = range(1, 6)
	train_losses = []
	valid_losses = []
	for num in N:
		train_loss, valid_loss = model_train(num)
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

