import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load MNIST data
data = pd.read_csv('../train.csv')
data = np.array(data)
np.random.shuffle(data)

X_train = data[:, 1:] / 255.
Y_train = data[:, 0]

# Autoencoder settings
n_inputs = 784
n_hidden = 64  # codings
n_outputs = n_inputs

def init_params():
    w1 = np.random.randn(n_inputs, n_hidden)
    b1 = np.zeros(n_hidden)
    w2 = np.random.randn(n_hidden, n_outputs)
    b2 = np.zeros(n_outputs)
    return w1, b1, w2, b2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_prop(X, w1, b1, w2, b2):
    Z1 = np.dot(X, w1) + b1
    A1 = sigmoid(Z1)  # codings
    Z2 = np.dot(A1, w2) + b2
    A2 = sigmoid(Z2)  # reconstructed inputs
    return Z1, A1, Z2, A2

def backward_prop(X, Z1, A1, Z2, A2, w1, w2):
    dZ2 = (A2 - X) * A2 * (1 - A2)
    dw2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)
    dZ1 = np.dot(dZ2, w2.T) * A1 * (1 - A1)
    dw1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def train_autoencoder(X, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for iteration in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(X, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward_prop(X, Z1, A1, Z2, A2, w1, w2)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
    return w1, b1, w2, b2

w1, b1, w2, b2 = train_autoencoder(X_train, 0.01, 1000)

def generate_images(w1, b1, w2, b2, n):
    # Generate n random codings
    codings = np.random.normal(size=[n, n_hidden])
    
    # Decode the codings to get images
    _, _, _, images = forward_prop(codings, w1, b1, w2, b2)
    
    # Plot the images
    fig, axes = plt.subplots(1, n, figsize=(n, 1))
    for image, ax in zip(images, axes):
        ax.imshow(image.reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

generate_images(w1, b1, w2, b2, 5)
