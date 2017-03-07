
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


with np.load("notMNIST.npz") as data :
	Data, Target = data ["images"], data["labels"]
	posClass = 2
	negClass = 9
	dataIndx = (Target==posClass) + (Target==negClass)
	Data = Data[dataIndx]/255.
	Target = Target[dataIndx].reshape(-1, 1)
	Target[Target==posClass] = 1
	Target[Target==negClass] = 0
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data, Target = Data[randIndx], Target[randIndx]
	trainData, trainTarget = Data[:3500], Target[:3500]
	validData, validTarget = Data[3500:3600], Target[3500:3600]
	testData, testTarget = Data[3600:], Target[3600:]
num_classes = 2
trainData = trainData.reshape((trainData.shape[0], trainData.shape[1] * trainData.shape[2]))
validData = validData.reshape((validData.shape[0], validData.shape[1] * validData.shape[2]))
testData = testData.reshape((testData.shape[0],testData.shape[1] * testData.shape[2]))


def buildGraph():
	# Variable creation
	image_pixels = 28 * 28
	# Placeholders
	X = tf.placeholder(tf.float32, shape=[None, image_pixels])
	Targets = tf.placeholder(tf.float32, shape=[None, 1])

	# Variables
	W = tf.Variable(np.random.randn(image_pixels, 1).astype("float32"), name="weight")
	b = tf.Variable(0.0, name="bias")
	logits = tf.add(tf.matmul(X, W), b)
	Y = tf.sigmoid(logits)
	predict = tf.cast(Y > 0.5, tf.float32)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Targets), tf.float32))

	cost_batch = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=Targets)))
	Lambda = 0
	weight_cost = tf.reduce_sum(W*W) * 0.5 * Lambda
	cost = tf.add(cost_batch, weight_cost)
	learning_rate = 0.01 #learning rate for gradient descent optimizer is 0.1, for adam optimizer its 0.01
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(cost)


	return accuracy, W, b, X, Targets, Y, cost, train_op



accuracy, W, b, X, Targets, Y, cost, train_op = buildGraph()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

num_epochs = 200
mini_batch_size = 500
training_size = trainTarget.size
testing_size = testTarget.size
epochs_per_evaluation = 1


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