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
from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import EnergyBased, ODIN, MaxSoftmax, MaxLogit
from pytorch_ood.utils import OODMetrics, ToUnknown
from temperature_scaling import ModelWithTemperature
from ReliabilityDiagram import _calculate_ece, make_model_diagrams




# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 10
learning_rate = 0.01

# Transform
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
    'test': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
}



# Dataset
train_dataset = tv.datasets.CIFAR100(root="C:\Python39\Datasets", train=True,
                                    download=True, transform=data_transforms['train'])


train_data, val_data = torch.utils.data.random_split(train_dataset, [48000, 2000])

ID_test_dataset = tv.datasets.CIFAR100(root="C:\Python39\Datasets", train=False,
                                    download=True, transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

ID_test_loader = torch.utils.data.DataLoader(ID_test_dataset, batch_size=batch_size, shuffle=False)

OOD_test_dataset = Textures(root="data", download=True, transform=data_transforms['test'], target_transform=ToUnknown())

OOD_test_loader = torch.utils.data.DataLoader(OOD_test_dataset, batch_size=batch_size, shuffle=False)

mixed_test_loader = torch.utils.data.DataLoader(OOD_test_dataset + ID_test_dataset, batch_size=batch_size, shuffle=False)


dataloaders = {
    'train': train_loader,
    'val': val_loader
}

dataset_sizes = {
    'train': batch_size*len(train_loader),
    'val': batch_size*len(val_loader)
}


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
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 color channels
        self.bat1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # input size must be equal to previous output size (conv1)
        self.bat2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 100)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.pool1(F.relu(self.bat1(self.conv1(x))))
        out = self.dropout(out)
        out = self.pool2(F.relu(self.bat2(self.conv2(out))))
        out = self.dropout(out)
        out = out.view(-1, 16*5*5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    

# Own model to test
model = CNN().to(device)

# Source: https://github.com/chenyaofo/pytorch-cifar-models
#model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_shufflenetv2_x2_0", pretrained=True).to(device)
#model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True).to(device)

# Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 10 epochs
# Learning rate scheduling should be applied after optimizer’s update

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=150)

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

    acc = 100.0 * n_correct / n_samples
    print(f"\nOOD Accuracy of CNN: {acc}%")


# Using OOD data
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in OOD_test_loader:
        #print(images.shape())
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
            
        # Max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()          
            
    acc = 100.0 * n_correct / n_samples
    print(f"\nOOD Accuracy of CNN: {acc}%")
    
    
    


# Using mixed data
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in mixed_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
            
        # Max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()          
            
    acc = 100.0 * n_correct / n_samples
    print(f"\nMixed Accuracy of CNN: {acc}%\n")






detector = MaxLogit(model)

metrics = OODMetrics()

for x, y in mixed_test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())


# Reliability Diagram
outs = outs.view(-1, 100)
trues = torch.Tensor(trues_list)
make_model_diagrams(outs, trues, 10)



