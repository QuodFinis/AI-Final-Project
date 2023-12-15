import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Load data
    curr_dir = os.getcwd()
    path = os.path.join(curr_dir, '../train.csv')
    data = pd.read_csv(path)

    # Prepare the data
    data = torch.tensor(data.values).float()
    m, n = data.shape
    data = data[torch.randperm(m)]  # shuffle before splitting into dev and training sets

    X_train = data[1000:, 1:] / 255.
    Y_train = data[1000:, 0]

    X_test = data[:1000, 1:] / 255.
    Y_test = data[:1000, 0]

    # Create dataloaders
    train_dataset = TensorDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Initialize the model and optimizer
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(500):
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

        # Print accuracy
        if epoch % 100 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            print(f"Iteration: {epoch}, Test Accuracy: {accuracy}%")

    print("Completed")

if __name__ == "__main__":
    main()
