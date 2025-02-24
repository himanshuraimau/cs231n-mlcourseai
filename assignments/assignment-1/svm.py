import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

# Define the SVM model
class SVM(nn.Module):
    def __init__(self, input_size, C=1.0):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.C = C

    def forward(self, x):
        return self.linear(x)

    def hinge_loss(self, output, target):
        return torch.mean(torch.clamp(1 - output.t() * target, min=0))

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Prepare the data
X = mnist_dataset.data.numpy()
y = mnist_dataset.targets.numpy()

# Flatten the images
X = X.reshape(X.shape[0], -1)

# Convert labels to -1 and 1
y = np.array([1 if label == 1 else -1 for label in y])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

# SVM parameters
input_size = X_train.shape[1]
learning_rate = 0.001
num_epochs = 20
C = 1.0

# Initialize the SVM model
model = SVM(input_size, C)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = model.hinge_loss(outputs, y_train) + C * torch.sum(model.linear.weight ** 2)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.sign(outputs)
    correct = (predicted == y_test.unsqueeze(1)).sum().item()
    accuracy = correct / len(y_test)
    print(f'Accuracy of the SVM on the test set: {accuracy:.4f}')
