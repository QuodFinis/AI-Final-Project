import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load the MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)  # equivalent to w1 and b1
        self.fc2 = nn.Linear(10, 10)  # equivalent to w2 and b2

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))  # equivalent to ReLU(Z1)
        x = self.fc2(x)  # equivalent to Z2
        return x

# Create the network
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # equivalent to softmax and the loss calculation
optimizer = optim.SGD(net.parameters(), lr=0.1)  # equivalent to update_params

# Train the network
for epoch in range(501):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / len(labels)
            print(f"Iteration: {epoch}, Accuracy: {accuracy}%")

print('Finished Training')
