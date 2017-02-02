#written in python3.5
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#------------------------------------------------Initializing Dataset----------------------------------------
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


#------------------------------------------------FUNCTIONS---------------------------------------------------

#Part 1.3.1 take distance vector and return responsibility vector
def get_responsibility(distance, k):
    value, index = tf.nn.top_k(-distance, k)
    value = -value
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    ishape = distance.get_shape()
    formatInput = [[0, i] for i in sess.run(index)[0]]
    lst = [1.0/k]*k
    result = tf.SparseTensor(indices=formatInput, values=lst, shape=ishape)
    a = tf.to_float(result, name='ToFloat')
    b = tf.sparse_tensor_to_dense(a, validate_indices=False, default_value=0.0)

    return sess.run(b)

def prediction_for_one_point(target_matrix_of_training_set, responsibility):
    return sum([item[0] * item[1] for item in zip(target_matrix_of_training_set, responsibility[0])])

#Part 1.3.2 get prediction given a set of input data
def get_prediction(trainData, trainTarget, input_data, k):
    prediction_matrix = []
    trainTarget = [i[0] for i in trainTarget]
    for i in input_data:
        distanceVector = [abs(tData[0] - i[0]) for tData in trainData]
        distanceVector = tf.constant([distanceVector])
        responsibility = get_responsibility(distanceVector, k)
        prediction_matrix.append(prediction_for_one_point(trainTarget, responsibility))
    return prediction_matrix


def mean_square_error(predictionResult, validTarget):
    temp = 0
    for index in range(len(validTarget)):
        temp += (predictionResult[index] - validTarget[index])**2
    result = temp * (1.0/(2.0*len(validTarget)))
    return result


def k_performance(input_data_set, target):
    k=[1,3,5,50]
    for item in k:
        prediction = get_prediction(trainData, trainTarget, input_data_set, item)
        MSE = mean_square_error(prediction, target)
        print ("for k = ", item, ", mean square error is ", MSE)


#------------------------------------------------Compute Prediction and MSE-------------------------------------
print ("-----Compute Prediction and MSE using Training Data-----")
k_performance(trainData, trainTarget)
print ("-----Compute Prediction and MSE using Validation Data-----")
k_performance(validData, validTarget)
print ("-----Compute Prediction and MSE using Testing Data-----")
k_performance(testData, testTarget)


#------------------------------------------------Graph for the prediction function using differen k-------------
Data = np.linspace(0.0, 11.0, num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1)
# randIdx = np.arange(100)
# np.random.shuffle(randIdx)
# Data, Target = Data[randIdx[:]], Target[randIdx[:]]
print("-------")
plt.subplot(4,1,1)
k = 1
result = get_prediction(trainData, trainTarget, Data, k)
plt.plot(Data, result, 'g')
plt.title("k = 1")
plt.plot(Data, Target, 'bo')
print("running")
plt.subplot(4,1,2)
k = 3
result = get_prediction(trainData, trainTarget, Data, k)
plt.plot(Data, result,'g')
plt.title("k = 3")
plt.plot(Data, Target, 'bo')

print("running")
plt.subplot(4,1,3)
k = 5
result = get_prediction(trainData, trainTarget, Data, k)
plt.plot(Data, result, 'g')
plt.title("k = 5")
plt.plot(Data, Target, 'bo')
print("running")
plt.subplot(4,1,4)
k = 50
result = get_prediction(trainData, trainTarget, Data, k)
plt.plot(Data, result, 'g')
plt.title("k = 50")
plt.plot(Data, Target, 'bo')

plt.show()

