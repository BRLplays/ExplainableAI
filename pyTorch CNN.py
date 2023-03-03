import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from temperature_scaling import ModelWithTemperature
from ReliabilityDiagram import _calculate_ece, make_model_diagrams



# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 25
batch_size = 4
learning_rate = 0.01

# Transform
transform = transforms.Compose( [transforms.ToTensor(), 
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataset
train_dataset = tv.datasets.CIFAR10(root="C:\Python39\Datasets", train=True,
                                    download=True, transform=transform)

test_dataset = tv.datasets.CIFAR10(root="C:\Python39\Datasets", train=False,
                                    download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck")


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 color channels
        self.bat1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # input size must be equal to previous output size (conv1)
        self.bat2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.pool1(F.relu(self.bat1(self.conv1(x))))
        out = self.pool2(F.relu(self.bat2(self.conv2(out))))
        out = out.view(-1, 16*5*5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    

model = CNN().to(device)

# Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
n_total_epochs = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] -> [4, 3, 1024]
        # input layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(i+1) % 2000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{n_total_epochs}], Loss: {loss.item():.4f}")
        

print("Finished Training")


outs = torch.Tensor()
trues_list = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
            
        # Max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            
            outs = torch.cat((outs, outputs[i]), 0)
            trues_list.append(label.item())
            
            if(label == pred):
                n_class_correct[label] += 1
            
            n_class_samples[label] += 1
            
            
    acc = 100.0 * n_correct / n_samples
    print(f"Before Accuracy of CNN: {acc}%")
        
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc}%")


# Reliability Diagram
outs = outs.view(-1, 10)
trues = torch.Tensor(trues_list)
make_model_diagrams(outs, trues, 15)






# Temperature Scalling
print("\n\n\n")
scaled_model = ModelWithTemperature(model).to(device)
scaled_model.set_temperature(test_loader)

outs = torch.Tensor()
trues_list = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = scaled_model(images)
            
        # Max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            
            outs = torch.cat((outs, outputs[i]), 0)
            trues_list.append(label.item())
            
            if(label == pred):
                n_class_correct[label] += 1
            
            n_class_samples[label] += 1
            
            
    acc = 100.0 * n_correct / n_samples
    print(f"\nAfter Accuracy of CNN: {acc}%")
        
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc}%")


# Reliability Diagram
outs = outs.view(-1, 10)
trues = torch.Tensor(trues_list)
make_model_diagrams(outs, trues, 15)



