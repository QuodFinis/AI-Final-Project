import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable 

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.fc(x)
        
batch_size = 64
lr = 0.0002
z_size = 100  # Size of the random noise vector
hidden_size = 128

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('../data', train=True, download=False, transform=transform)

if not train_dataset:
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

generator = Generator(z_size, hidden_size, 28*28)
discriminator = Discriminator(28*28, hidden_size, 1)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr) 

num_epochs = 10 # 50
for epoch in range(num_epochs):
    for batch, (real_images, _) in enumerate(train_loader):
        optimizer_D.zero_grad()

        real_images = real_images.view(-1, 28*28)
        batch_size = real_images.size(0)  
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        output_real = discriminator(real_images)
        loss_real = nn.BCELoss()(output_real, real_labels)

        noise = Variable(torch.randn(batch_size, z_size))
        fake_images = generator(noise)

        output_fake = discriminator(fake_images.detach())  
        loss_fake = nn.BCELoss()(output_fake, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        output_fake = discriminator(fake_images)
        loss_g = nn.BCELoss()(output_fake, real_labels)  

        loss_g.backward()
        optimizer_G.step()

        if batch % 250 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch}/{len(train_loader)}], '
                  f'D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}')


noise = Variable(torch.randn(16, z_size))
generated_images = generator(noise) 

noise_for_digit_1 = torch.randn(16, z_size)
noise_for_digit_1[:, 0] = 1.0  

generated_images_digit_1 = generator(noise_for_digit_1).detach()

selected_image = generated_images_digit_1[1].reshape(28, 28)

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Image for Digit 1 at Index 1")
plt.imshow(selected_image, cmap='gray')
plt.show()
