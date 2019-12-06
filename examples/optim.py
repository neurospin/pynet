"""
pynet optim helpers overview
==============================

Credit: A Grigis

pynet is a Python package related to deep learning and its application in
MRI mediacal data analysis. It is accessible to everybody, and is reusable
in various contexts. The project is hosted on github:
https://github.com/neurospin/pynet.

First checks
------------

In order to test if the 'pynet' package is installed on your machine, you can
check the package version.
"""

import pynet
print(pynet.__version__)

#############################################################################
# Now you can run the the configuration info function to see if all the
# dependencies are installed properly.

import pynet.configure
print(pynet.configure.info())

#############################################################################
# Optimisation
# ------------
#
# First load a dataset (the CIFAR10) and a network.
# You may need to change the 'datasetdir' parameter.

import torch.nn as nn
import torch.nn.functional as func
from pynet.datasets import DataManager, fetch_cifar

data = fetch_cifar(datasetdir="/neurospin/nsap/datasets/cifar")
manager = DataManager(
    input_path=data.input_path,
    labels=["label"],
    metadata_path=data.metadata_path,
    number_of_folds=10,
    batch_size=10,
    stratify_label="category",
    test_size=0.1)

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
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

#############################################################################
# Now start the optimisation.

import torch
from pynet.classifier import Classifier

cl = Classifier(
    optimizer_name="SGD",
    momentum=0.9,
    learning_rate=0.001,
    loss_name="CrossEntropyLoss",
    model=net,
    metrics=["accuracy"])
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=3,
    checkpointdir="/tmp/pynet",
    fold_index=0,
    with_validation=True)

#############################################################################
# You can reload the optimization history at any time and any step.

from pprint import pprint
from pynet.history import History
from pynet.plotting import plot_history

history = History.load("/tmp/pynet/train_0_epoch_2.pkl")
print(history)
plot_history(history)

#############################################################################
# And now predict the labels on the test set.

import numpy as np
from sklearn.metrics import classification_report
from pynet.plotting import plot_data

y_pred, X, y_true, loss, values = cl.testing(
    manager=manager,
    with_logit=True,
    predict=True)
pprint(data.labels)
print(classification_report(y_true, y_pred, target_names=data.labels.values()))
titles = ["{0}-{1}".format(data.labels[it1], data.labels[it2])
          for it1, it2 in zip(y_pred, y_true)]
plot_data(X, labels=titles, nb_samples=5)

# import matplotlib.pyplot as plt
# plt.show()
