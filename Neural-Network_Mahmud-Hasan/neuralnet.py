# neural nets are essentially trying to fit a fnx onto

# 28 x 28 pixel image
# as the images are greyscale, each pixel has value 0 (black) to 255 (white)

# represent this first as a matrix, each row being an example and having 784 columns
# corresponding to each pixel in the image, brightness of pixels

# transpose this matrix such that each column is an example and each column has 784 rows

# input layer: 784 nodes for each pixel
# hiden layer: 10 nodes
# ouput layey: 10 nodes for each possible digit classification

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# importing training data as pandas dataframe
curr_dir = os.getcwd()
path = os.path.join(curr_dir, '../train.csv')
data = pd.read_csv(path)

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.


def init_params():
    # random starting weights from -0.5 to 0.5 of size 10 by 784 since we are multiplying weight by  input which is a
    # column of the data 784 x 1
    w1 = np.random.rand(10, 784) - 0.5

    # random bais weight 10 x 1 because ouput of ^ mult 784x1 * 10x784 is 10x1
    b1 = np.random.rand(10, 1) - 0.5

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


# forward propegation
# running image through the network to see what network outputs
# A0 = x the input   (input layer 784 x m)
# Z1 = w1 * A0 + b1   (weight (10 x 784) dot input + bias (10 x 1)) => 10 x m output) (think of this as the synapes)
# A1 = ReLU(Z1)   (apply activation function ReLU still 10x1 matrix)
# Z2 = w2 * A1 + b2 (again this is like the connection between neuron/nodes)
# A2 = softmax(Z2) (another activation function, softmax, which helps turn the outputs into a probability)

# goes through each element in Z, returns relu of it
def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))  # values of Z will sum to 1, giving percentage


def forward_prop(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# what are the correct weights and biases? that is determined in
# backwards propegation
# find out how much the prediction is off from the correct result, error, and determine how much the previous weight
# and biases contriubte to this, and then adjust
# dZ2 = A2 - Y    (take prediction and subtract the correct value, note that the node outputs a 10x1 matrix,
#                  the correct matrix is similar execpt every row is 0 except the index for the correct number)
# dw2 = 1/m * dZ2 * A1^T   (derivative of loss fnx with respect to the weights)
# db2 = 1/m * SUM(dZ2)   (average of the errors)
# dZ1 = w2^T * dZ2 * f'()   (this is sortve like propegation in reverse)
# dw1 = 1/m * dZ! * A0^T   (derivative of loss fnx with respect to the weights)
# db1 = 1/m * SUM(dZ1)

def one_hot(Y):
    # create the label matrix for each data and set the correct identifier as 1
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T  # remember to flip because we want columns to be data
    return one_hot_Y


def dydx_ReLU(Z):
    return (Z > 0).astype(int)  # if x less than 0 returns 0 if greater return 1, slope of y=z


def backward_prop(Z1, A1, Z2, A2, w1, w2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dw2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = w2.T.dot(dZ2) * dydx_ReLU(Z1)
    dw1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dw1, db1, dw2, db2


# update parameters usinig learning rate r (hyper parameter)
# w1 = w1 - r * dw1
# b1 = b1 - r * db1
# w2 = w2 - r * dw2
# b2 = b2 - r * db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


# determine accuracy of model
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# describe gradient descent
# iterations how many times updating params

def gradient_descent(X, Y, alpha, iterations):
    w1, b1, w2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = backward_prop(Z1, A1, Z2, A2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, Y))

    return w1, b1, w2, b2


w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.1, 501)


# test model on remaining  data
data_train = data[0:1000].T
Y_test = data_train[0]
X_test = data_train[1:n]
X_test = X_test / 255.

def test_neural_network(w1, b1, w2, b2, X_test, Y_test):
    Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_test)
    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, Y_test)
    print("Test Accuracy: ", accuracy)

    return predictions

# Test the neural network on the provided test dataset
test_predictions = test_neural_network(w1, b1, w2, b2, X_test, Y_test)


# 