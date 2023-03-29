import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from temperature_scaling import ModelWithTemperature
from ReliabilityDiagram import _calculate_ece, make_model_diagrams

from pytorch_ood.detector import EnergyBased, ODIN
from pytorch_ood.utils import OODMetrics, ToUnknown



# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 10
learning_rate = 0.01

# Transform
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor()
    ]),
}

# Dataset
train_dataset = tv.datasets.MNIST(root="C:\Python39\Datasets", train=True,
                                    download=True, transform=data_transforms['train'])

train_data, val_data = torch.utils.data.random_split(train_dataset, [56000, 4000])

ID_test_dataset = tv.datasets.MNIST(root="C:\Python39\Datasets", train=False,
                                    download=True, transform=data_transforms['test'])

OOD_test_dataset = tv.datasets.FashionMNIST(root="C:\Python39\Datasets", train=False,
                                    download=True, transform=ToUnknown())


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

ID_test_loader = torch.utils.data.DataLoader(ID_test_dataset, batch_size=batch_size, shuffle=False)

OOD_test_loader = torch.utils.data.DataLoader(OOD_test_dataset, batch_size=batch_size, shuffle=False)

mixed_test_loader = torch.utils.data.DataLoader(ID_test_dataset + OOD_test_dataset, batch_size=batch_size, shuffle=False)

dataloaders = {
    'train': train_loader,
    'val': val_loader
}

dataset_sizes = {
    'train': batch_size*len(train_loader),
    'val': batch_size*len(val_loader)
}

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# Demo
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape, '\n')

"""
for i in range(4):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap="gray")
#plt.show()
"""


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}\n\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 color channels
        self.bat1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # input size must be equal to previous output size (conv1)
        self.bat2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.pool1(F.relu(self.bat1(self.conv1(x))))
        out = self.pool2(F.relu(self.bat2(self.conv2(out))))
        out = out.view(-1, 16*4*4)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Own model to test
model = CNN().to(device)

# Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 10 epochs
# Learning rate scheduling should be applied after optimizer’s update

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=3)
model.eval()


# Testing
outs = torch.Tensor()
trues_list = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in ID_test_loader:
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
make_model_diagrams(outs, trues, 10)



# Stage 2: Create Detector
detector = EnergyBased(model)

# Stage 3: Evaluate Detectors
metrics = OODMetrics()

for x, y in mixed_test_loader:
    x = x.to(torch.float)
    y = y.to(torch.float)
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())



"""
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
make_model_diagrams(outs, trues, 10)

"""