import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import math

data2D = np.float32(np.load('data2D.npy'))
data = (data2D - data2D.mean()) / data2D.std()
k = 5

num_sample = data.shape[0]
dimension = data.shape[1]
optimizer = None
tf_loss = None
result = None
var_mean = None
placeholder = None




#build graph
var_mean = tf.Variable(tf.random_normal([k, dimension], mean=0.0, stddev=1.0, dtype=tf.float32))
tf_covariance = tf.Variable(0.5 * tf.exp(tf.random_normal([k], mean=0.0, stddev=1.0, dtype=tf.float32)))
phi = tf.Variable(tf.truncated_normal([1, k], mean=0.0, stddev=1.0, dtype=tf.float32))
log_pi = utils.logsoftmax(phi)
placeholder = tf.placeholder(tf.float32, shape=(num_sample, dimension))
expanded_data = tf.expand_dims(placeholder, 0)
tf_expanded_mean = tf.expand_dims(var_mean, 1)
tf_sub = tf.sub(expanded_data, tf_expanded_mean)
sub_square = tf.square(tf_sub)
sub_square_sum = tf.reduce_sum(sub_square, 2, True)
sub_square_sum_02 = tf.squeeze(tf.transpose(sub_square_sum))
index = (-0.5) * tf.div(sub_square_sum_02, tf_covariance)
tf_log_first_term = (-0.5 * dimension) * tf.log(2 * math.pi * tf_covariance)
tf_log_x_gan_z = tf.add(tf_log_first_term, index)
tf_log_pro_z_x_gan_z = tf.add(log_pi, tf_log_x_gan_z)
sum_prob = utils.reduce_logsumexp(tf_log_pro_z_x_gan_z, 1)
tf_log_like = tf.reduce_sum(sum_prob)
loss = -1 * tf_log_like
optimizer = tf.train.AdamOptimizer(0.1, 0.9, 0.99, 1e-5).minimize(loss)
trans_op = tf.transpose(tf.expand_dims(tf.reduce_sum(tf_log_pro_z_x_gan_z, 1), 0))
sub = tf.sub(tf_log_pro_z_x_gan_z, trans_op)
result = tf.argmax(sub, 1)


def scatter_plot(data, k, assignments):
    num_sample, dim = data.shape
    mark = ['or', 'ob', 'og', 'ok', '^r']
    for i in range(num_sample):
        plt.plot(data[i, 0], data[i, 1], mark[assignments[i]])


def model_train():
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    init.run()
    loss_list = []
    for i in range(10000):
        feed_dict = {placeholder: data}
        _, loss, assignments, mean = sess.run([optimizer, tf_loss, result, var_mean], feed_dict = feed_dict)
        loss_list.append(loss)

    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    scatter_plot(data, k, assignments)
    plt.show()

if __name__ == '__main__':
	model_train()
