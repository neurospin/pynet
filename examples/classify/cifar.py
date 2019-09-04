"""
pynet: multi modal slice orientation prediction 
===============================================

Credit: A Grigis
Based on:
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

This practice demonstrates how to perform the pytorch tutorial in the
pynet framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import urllib
import json
from pynet.plotting import plot_data
from pynet.classifier import Classifier
import matplotlib.pyplot as plt


datadir = "/neurospin/nsap/torch/data"
labels_url = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/"
    "imagenet_class_index.json")

with urllib.request.urlopen(labels_url) as response:
    labels = json.loads(response.read().decode())
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(
    root=datadir,
    train=True,
    download=True,
    transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=50000,
    shuffle=True,
    num_workers=2)
trainiter = iter(trainloader)
X_train, y_train = trainiter.next()
X_train = X_train.numpy()
y_train = y_train.numpy()
testset = torchvision.datasets.CIFAR10(
    root=datadir,
    train=False,
    download=True,
    transform=transform)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=10,
    shuffle=False,
    num_workers=2)
testiter = iter(testloader)
X_test, y_test = testiter.next()
X_test = X_test.numpy()
y_test = y_test.numpy()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
classes = (
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
    "truck")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cl = Classifier(
    batch_size=4,
    optimizer_name="SGD",
    learning_rate=0.001,
    loss_name="CrossEntropyLoss",
    metrics=["accuracy"],
    model=Net(),
    momentum=0.9)
cl.fit(X_train, y_train, nb_epochs=2)
y_pred = cl.predict(X_test)
for key1, key2 in zip(y_pred, y_test):
    print(key1, classes[key1], key2, classes[key2])
plot_data(X_test / 2. + 0.5)

plt.show()
