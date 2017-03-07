import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

with np.load("notMNIST.npz") as data:
    Data, Target = data["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx] / 255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
num_classes = 10

print (len(testTarget))

trainData = trainData.reshape((trainData.shape[0], trainData.shape[1] * trainData.shape[2]))
validData = validData.reshape((validData.shape[0], validData.shape[1] * validData.shape[2]))
testData = testData.reshape((testData.shape[0], testData.shape[1] * testData.shape[2]))

num_examples = trainTarget.size
labels_one_hot = np.zeros((num_examples, trainTarget.max() - trainTarget.min() + 1))
labels_one_hot[np.arange(num_examples), trainTarget.ravel()] = 1
trainTarget = labels_one_hot

num_examples = testTarget.size
labels_one_hot = np.zeros((num_examples, testTarget.max() - testTarget.min() + 1))
labels_one_hot[np.arange(num_examples), testTarget.ravel()] = 1
testTarget = labels_one_hot

num_examples = validTarget.size
labels_one_hot = np.zeros((num_examples, validTarget.max() - validTarget.min() + 1))
labels_one_hot[np.arange(num_examples), validTarget.ravel()] = 1
validTarget = labels_one_hot


def init_weights(shape, init_method='xavier', xavier_params=(None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else:  # xavier
        (fan_in, fan_out) = xavier_params
        low = -4 * np.sqrt(6.0 / (fan_in + fan_out))  # {sigmoid:4, tanh:1}
        high = 4 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def model(X, num_hidden, droprate):
    w_h = init_weights([784, num_hidden], 'xavier', xavier_params=(1, num_hidden))
    #w_h_2 = init_weights([num_hidden, num_hidden], 'xavier', xavier_params=(1, num_hidden))#
    b_h = tf.Variable(tf.zeros([num_hidden]))
   # b_h_2 = tf.Variable(tf.zeros([num_hidden]))#
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h)
    h_drop = tf.nn.dropout(h, droprate)
   # h_2 = tf.nn.relu(tf.matmul(h, w_h_2) + b_h_2)#
    w_o = init_weights([num_hidden, 10], 'xavier', xavier_params=(num_hidden, 1))
    b_o = tf.Variable(tf.zeros([10]))
    return tf.matmul(h_drop, w_o) + b_o, w_h,  w_o


def buildGraph():

    X = tf.placeholder(tf.float32, shape=[None, 784])
    droprate = tf.placeholder(tf.float32)
    Targets = tf.placeholder(tf.float32, shape=[None, num_classes])
    Y_pred, w_h, w_o = model(X, 1000, droprate)
    learning_rate = 0.001

    weight_decay_loss = 3.0 * tf.exp(-10.0) * (tf.reduce_sum(tf.square(w_h)) + tf.reduce_sum(tf.square(w_o))) / 2.0#

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Targets)) + weight_decay_loss

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    return accuracy, X, Targets, Y_pred, cost, train_op, w_h, droprate


accuracy, X, Targets, Y_pred, cost, train_op, w_h, droprate = buildGraph()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

num_epochs = 200
mini_batch_size = 500
training_size = len(trainTarget)
testing_size = len(testTarget)
epochs_per_evaluation = 1

init = tf.global_variables_initializer()
sess.run(init)
train_accuracy = []
train_cost = []
test_accuracy = []
test_cost = []

train_accuracy = []
valid_accuracy = []
test_accuracy = []
train_cost = []
valid_cost = []
test_cost = []

for epoch in range(num_epochs):
    for i in range(int(training_size / mini_batch_size)):
        batch_x = trainData[i * mini_batch_size: (i + 1) * mini_batch_size]
        batch_y = trainTarget[i * mini_batch_size: (i + 1) * mini_batch_size]
      

        _accuracy, _cost, _ = sess.run([accuracy, cost, train_op], feed_dict={X: batch_x, Targets: batch_y, droprate: 1})

    if 0 == (epoch % epochs_per_evaluation):
        cost_train, accuracy_train, weights = sess.run([cost, accuracy, w_h], feed_dict={X: trainData, Targets: trainTarget, droprate: 1.0})
        cost_valid, accuracy_valid = sess.run([cost, accuracy], feed_dict={X: validData, Targets: validTarget, droprate: 1.0})
        cost_test, accuracy_test = sess.run([cost, accuracy], feed_dict={X: testData, Targets: testTarget, droprate: 1.0})

        train_cost.append(cost_train)
        train_accuracy.append(accuracy_train)
        valid_cost.append(cost_valid)
        valid_accuracy.append(accuracy_valid)
        test_cost.append(cost_test)
        test_accuracy.append(accuracy_test)
        print("epoch: ",epoch)

    if epoch == 50:
	    f1 = plt.figure(1)
	    plt.imshow(weights, cmap='gray')
	    plt.title("25%")
	    f1.show()
    elif epoch == 150:
	    f2 = plt.figure(2)
	    plt.imshow(weights, cmap='gray')
	    plt.title("75%")
	    f2.show()
    elif epoch == 190:
	    f3 = plt.figure(3)
	    plt.imshow(weights, cmap='gray')
	    plt.title("100%")
	    f3.show()



train_error = [1-i for i in train_accuracy]
valid_error = [1-i for i in valid_accuracy]
test_error = [1-i for i in test_accuracy]
Err_train = (1-accuracy_train) * len(trainTarget)
Err_valid = (1-accuracy_valid) * len(validTarget)
Err_test = (1-accuracy_test) * len(testTarget)



plt.imshow(weights, cmap='gray')
plt.title("100%")
plt.show()

# plt.title("Error vs Number of Epochs")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Error")
# plt.plot(train_error)
# plt.plot(valid_error)
# plt.plot(test_error)
# plt.legend(['Training ', 'Validation', 'Test'])
# plt.show()

#
# plt.title("Cost vs Number of Epochs")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Cost")
# plt.plot(train_cost)
# plt.plot(valid_cost)
# plt.plot(test_cost)
# plt.legend(['Training ', 'Validation', 'Test'])
# plt.show()



print("test cost: ", cost_test)
print("test accuracy: ", accuracy_test)
print("validation cost: ", cost_valid)
print("validation accuracy: ", accuracy_valid)
print("train cost: ", cost_train)
print("train accuracy: ", accuracy_train)
print("training errors: ", Err_train)
print("validation errors: ", Err_valid)
print("test errors: ", Err_test)
print("error rates for training, validation and test:", 1-accuracy_train, 1-accuracy_valid, 1-accuracy_test)
