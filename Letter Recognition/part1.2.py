
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


with np.load("notMNIST.npz") as data:
	Data, Target = data ["images"], data["labels"]
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data = Data[randIndx]/255.
	Target = Target[randIndx]
	trainData, trainTarget = Data[:15000], Target[:15000]
	validData, validTarget = Data[15000:16000], Target[15000:16000]
	testData, testTarget = Data[16000:], Target[16000:]
num_classes = 10



trainData = trainData.reshape((trainData.shape[0], trainData.shape[1] * trainData.shape[2]))
validData = validData.reshape((validData.shape[0], validData.shape[1] * validData.shape[2]))
testData = testData.reshape((testData.shape[0],testData.shape[1] * testData.shape[2]))
print(trainData.shape)


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


def buildGraph():
	image_pixels = 28 * 28
	X = tf.placeholder(tf.float32, shape=[None, image_pixels])
	Targets = tf.placeholder(tf.float32, shape=[None, num_classes])

	W = tf.Variable(tf.truncated_normal([image_pixels, num_classes], stddev=0.1))
	b = tf.Variable(tf.zeros([num_classes]))
	# W = tf.Variable(np.random.randn(image_pixels, num_classes).astype("float32"), name="weight")
	# b = tf.Variable(np.random.randn(num_classes).astype("float32"), name="bias")

	logits = tf.add(tf.matmul(X, W), b)
	Y = tf.nn.softmax(logits)
	cost_batch =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Targets))
	Lambda = 0.01
	weight_cost = tf.reduce_sum(W*W) * 0.5 * Lambda
	cost = tf.reduce_sum(cost_batch)+ weight_cost
	learning_rate = 0.0001
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(cost)

	correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Targets,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
	return accuracy, W, b, X, Targets, Y, cost, train_op



accuracy, W, b, X, Targets, Y, cost, train_op = buildGraph()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


num_epochs = 100
mini_batch_size = 500
training_size = trainTarget.size
testing_size = testTarget.size
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

		_accuracy, _cost, _ = sess.run([accuracy, cost, train_op], feed_dict={X: batch_x, Targets: batch_y})

	if 0 == (epoch % epochs_per_evaluation):
		cost_train, accuracy_train= sess.run([cost, accuracy], feed_dict={X: trainData, Targets: trainTarget})
		cost_valid, accuracy_valid = sess.run([cost, accuracy], feed_dict={X: validData, Targets: validTarget})
		cost_test, accuracy_test = sess.run([cost, accuracy], feed_dict={X: testData, Targets: testTarget})

		train_cost.append(cost_train)
		train_accuracy.append(accuracy_train)
		valid_cost.append(cost_valid)
		valid_accuracy.append(accuracy_valid)
		test_cost.append(cost_test)
		test_accuracy.append(accuracy_test)

print("test cost: ", cost_test)
print("test accuracy: ", accuracy_test)

# plt.title("Accuracy vs Number of Epochs")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Accuracy")
# plt.plot(train_accuracy)
# plt.plot(valid_accuracy)
# plt.plot(test_accuracy)
# plt.legend(['Training ', 'Validation', 'Test'])


plt.title("Cost vs Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.plot(train_cost)
plt.plot(valid_cost)
plt.plot(test_cost)
plt.legend(['Training ', 'Validation', 'Test'])
plt.show()