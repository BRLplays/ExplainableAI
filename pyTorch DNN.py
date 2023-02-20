import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

"""
data = tv.datasets.MNIST(root="C:\Python39\Datasets", train=True)
image, label = data[2]
print(image)
plt.imshow(image, cmap="gray")
plt.show()  
"""

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
input_size = 784 # 28x28
hidden_size = 100
num_class = 10
num_epochs = 4
batch_size = 100
learning_rate = 0.01

# MNIST dataset
train_dataset = tv.datasets.MNIST(root="C:\Python39\Datasets", train=True,
                                  transform=transforms.ToTensor(), download=True)

test_dataset = tv.datasets.MNIST(root="C:\Python39\Datasets", train=False,
                                  transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Demo
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap="gray")
#plt.show()

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        # Output is logits
        # No softmax because CrossEntropy already applies it
        return out
    

model = NN(input_size, hidden_size, num_class)

# Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # (100, 1, 28, 28) -> (100, 784)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f"epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}")
        
    
# Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # Value, index
        _, pred = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (pred == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"accuraccy = {acc}")