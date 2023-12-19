import os
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


# Load and prepare data
curr_dir = os.getcwd()
path = os.path.join(curr_dir, '../train.csv')
data = pd.read_csv(path)

# Convert data to PyTorch tensors and shuffle
data = torch.tensor(data.values).float()
m, n = data.shape
data = data[torch.randperm(m)]

# Split data into training and test sets
data_train = data[1000:m]
Y_train = data_train[:, 0].long()
X_train = data_train[:, 1:n] / 255.

data_test = data[0:1000]
Y_test = data_test[:, 0].long()
X_test = data_test[:, 1:n] / 255.

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# optimizer is an algorithm or method used to adjust the parameters of your model to minimize the error or loss.
# The optimizer updates the model parameters by a rule defined by the optimization algorithm being used. In this case,
# we’re using Stochastic Gradient Descent (SGD)
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# Training a model on the entire dataset at once can be computationally expensive, especially for large datasets. To
# overcome this, we divide the dataset into smaller subsets known as batches. We then use these batches to train the
# model iteratively, updating the model parameters after each batch.
# Shuffling the training data is important to prevent the model from learning the order of the training examples. This
# helps to make sure that the model generalizes well and doesn’t overfit to the training data.
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_data, batch_size=64)


# Training loop
for epoch in range(100):
    start_time = time.time()  # Start timer
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        # Cross-entropy loss, commonly known as log loss, serves as a crucial metric for evaluating the performance
        # of classification models, especially in the realm of neural networks. It quantifies the disparity between
        # predicted probabilities and the actual distribution of labels. The loss is formulated as the negative log
        # probability of the true class, encouraging the model to assign high probabilities to correct classes and
        # penalizing confidently incorrect predictions. During training, the objective is to minimize this
        # cross-entropy loss, essentially aligning the predicted probability distribution with the true distribution.
        # The probabilistic interpretation underscores its role in optimizing the likelihood of observed data given
        # the model's parameters. Gradients of the loss are computed during training, and the model parameters are
        # adjusted through iterative optimization techniques, such as gradient descent, to minimize the loss. In the
        # context of neural networks, cross-entropy loss is often paired with softmax activation in the output layer
        # to ensure interpretable probabilities.
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    # Print accuracy and time taken every 10 iterations
    if epoch % 10 == 0:
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Iteration: {}, Accuracy: {}, Time taken for last 10 iterations: {} seconds'.format(epoch, correct / total, elapsed_time))

# Visualize predictions for a few test samples
def test_prediction(index):
    current_image = X_test[index, :].view(28, 28).numpy() * 255
    current_image = current_image.astype(int)
    label = Y_test[index].item()

    with torch.no_grad():
        inputs = X_test[index, :].view(1, -1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.item()

    print("Prediction: ", prediction)
    print("Label: ", label)

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title(f"Prediction: {prediction}, Label: {label}")
    plt.show()

# Test the neural network on the provided test dataset
for i in range(5):  # Test the first 5 samples
    test_prediction(i)