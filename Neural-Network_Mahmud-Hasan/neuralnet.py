import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = [np.random.rand(layer_sizes[i + 1], layer_sizes[i]) - 0.5 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.rand(layer_sizes[i + 1], 1) - 0.5 for i in range(len(layer_sizes) - 1)]

    def forward_propagation(self, inputs):
        activations = [inputs]
        weighted_inputs = []

        for i in range(len(self.layer_sizes) - 1):
            weighted_input = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            weighted_inputs.append(weighted_input)
            activation = self.activation_functions[i](weighted_input)
            activations.append(activation)

        return weighted_inputs, activations

    def backward_propagation(self, inputs, targets, weighted_inputs, activations, learning_rate):
        m = inputs.shape[1]
        deltas = [activations[-1] - targets]

        for i in reversed(range(len(self.layer_sizes) - 2)):
            delta = np.dot(self.weights[i + 1].T, deltas[-1]) * self.activation_functions[i](weighted_inputs[i], derivative=True)
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] -= learning_rate * np.dot(deltas[i], activations[i].T) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=1, keepdims=True) / m

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            weighted_inputs, activations = self.forward_propagation(inputs)
            self.backward_propagation(inputs, targets, weighted_inputs, activations, learning_rate)

            if epoch % 100 == 0:
                loss = np.mean(np.square(activations[-1] - targets))
                print(f'Epoch: {epoch}, Loss: {loss}')

    def predict(self, inputs):
        _, predictions = self.forward_propagation(inputs)
        return predictions[-1]


def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(int)
    return np.maximum(0, x)

def sigmoid(x, derivative=False):
    sig = 1 / (1 + np.exp(-x))
    if derivative:
        return sig * (1 - sig)
    return sig

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


if __name__ == "__main__":
    curr_dir = os.getcwd()
    path = os.path.join(curr_dir, '../train.csv')
    data = pd.read_csv(path)

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    data_test = data[0:1000].T
    Y_test = data_test[0]
    X_test = data_test[1:n]
    X_test = X_test / 255.

    # Example: 784 input nodes, 3 hidden layers with 64 nodes each, 10 output nodes
    nn = NeuralNetwork(layer_sizes=[784, 64, 64, 64, 10], activation_functions=[relu, relu, sigmoid])

    # Train the neural network
    nn.train(X_train, Y_train, epochs=500, learning_rate=0.1)

    # Test the neural network on the provided test dataset
    test_predictions = nn.predict(X_test)

    # Visualize predictions for a few test samples
    for i in range(4):
        current_image = X_test[:, i, None]
        prediction = np.argmax(test_predictions[:, i])
        label = Y_test[i]

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.title(f"Prediction: {prediction}, Label: {label}")
        plt.show()
