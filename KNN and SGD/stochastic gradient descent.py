from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random


#---------------------------------------------hyper params----------------------------------------
learning_rate = 0.11
training_epochs = 14
size = 50
weight_decay= 0.1


#---------------------------------------------Initializing Data-----------------------------------------
with numpy.load ("tinymnist.npz") as data :
	train_X, train_Y = data ["x"], data["y"]
	validData, validTarget = data ["x_valid"], data ["y_valid"]
	testData, testTarget = data ["x_test"], data ["y_test"]

i = 0
resultx = []
resulty = []
while i < len(train_X):
    resultx.append(train_X[i:i+size])
    resulty.append(train_Y[i:i + size])
    i += size

n_samples = resultx[0].shape[0]


temp = tf.constant(rng.rand(1,64))
temp = tf.cast(temp, tf.float32)
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(temp, name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.mul(X, W), b)



#--------------------------------------------main logic------------------------------
# Mean squared error
cost = tf.reduce_mean(tf.reduce_mean(tf.square(pred - Y), reduction_indices=1, name='squared_error'), name='mean_squared_error')+ 0.4*weight_decay * tf.reduce_sum(pow(W,2))
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()


#--------------------------------------------batch running----------------------------
with tf.Session() as sess:
    sess.run(init)
    cost_list = []
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(resultx[epoch], resulty[epoch]):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        #if (epoch+1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: resultx[epoch], Y:resulty[epoch]})
        cost_list.append(c)
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimized")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, '\n')

#------------------------------------plotting loss vs updates---------------------------------------------
    plt.plot(cost_list)
    plt.title('Total Loss Function vs Number of Updates')
    plt.show

#------------------------------------raw prediction list-----------------------------------------------
    Y_prediction = []
    for item_x in train_X:
        Y_prediction.extend(numpy.inner(sess.run(W), item_x) + sess.run(b))

#-------------------------------------------computing lost using test/validation data----------------------

    test_X = testData
    test_Y = testTarget

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(cost, feed_dict={X: testData, Y: testTarget}) # same function as cost above
    print("Testing cost=", testing_cost)


#---------------------------------computing prediction result, accuracy and graphing.----------------------------
    sum=0
    test_pred = []
    guess = []
    for item_x in test_X:
	    test_pred.extend(numpy.inner(sess.run(W), item_x) + sess.run(b))
    for item in test_pred:
	    if item > 0.5:
		    guess.append(1)
	    else:
		    guess.append(0)

    for i in range(400):
	    if test_Y[i] == guess[i]:
		    sum += 1
    print("accuracy = ", sum / 400)
    guess.append(1.1)
    guess.append(-0.1)
    # plt.plot(test_Y, 'ro', label='Testing data')
    # plt.plot(guess, 'b^', label='guess')
    # plt.title('accuracy for weight decay coefficient = 0.1 ')
    plt.show()

