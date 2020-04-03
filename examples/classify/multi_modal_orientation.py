"""
pynet: multi modal slice orientation prediction
===============================================

Credit: A Grigis

This practice focuses on a meaningless toy example inspired from the MNIST
manuscript numbers classification challenge that is considered as the
'hello world' example for neural networks.

Here, we have to recognize wether a brain slice image is axial, sagittal or
coronal and comes from MRI T1w, MRI T2w or CT modality. So, there are 9
classes.
"""

# Common imports
import pynet
import numpy as np
import os
import glob

#############################################################################
# And of course we import PyTorch

import torch
torch.__version__

#############################################################################
# Read the data
# -------------
#
# We define training + valid sets vs test set using k-fold cross-validations.
#
# A validation step is a useful way to avoid overfitting. At each epoch, the
# neural network is evaluated on the validation set, but not trained on it.
# If the validation loss starts to grow, it means that the network is
# overfitting the training set, and that it is time to stop the training.
#
# The following cell create stratified test, train, and validation loaders.

from pynet.datasets import fetch_orientation
from pynet.datasets import DataManager

data = fetch_orientation(
    datasetdir="/tmp/orientation",
    flatten=True)
manager = DataManager(
    input_path=data.input_path,
    labels=["label"],
    metadata_path=data.metadata_path,
    number_of_folds=10,
    batch_size=1000,
    stratify_label="label",
    test_size=0.1,
    sample_size=(1 if "CI_MODE" not in os.environ else 0.1))


#############################################################################
# Displaying some images of the test dataset.

from pynet.plotting import plot_data

dataset = manager["test"]
sample = dataset.inputs.reshape(-1, data.height, data.width)
sample = np.expand_dims(sample, axis=1)
plot_data(sample, nb_samples=5)


#############################################################################
# Simple neural network
# ---------------------
#
# The simplest way to create, train and test a network is to use Sequential
# container.
# With a sequential container, you can quickly design a linear stack of layers
# and so, many kinds of models (LSTM, CNN, ...).
# Here we create a simple Multilayer Perceptron (MLP) for multi-class softmax
# classification.

import collections
import torch
import torch.nn as nn
image_size = data.height * data.width
nb_neurons = 16


class OneLayerMLP(nn.Module):
    """  Simple one hidden layer percetron.
    """
    def __init__(self, image_size, nb_neurons, nb_classes):
        """ Initialize the instance.

        Parameters
        ----------
        image_size: int
            the number of elemnts in the image.
        nb_neurons: int
            the number of neurons of the hidden layer.
        nb_classes: int
            the number of classes.
        """
        super(OneLayerMLP, self).__init__()
        self.layers = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(image_size, nb_neurons)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(nb_neurons, nb_classes)),
            ("softmax", nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x):
        x = self.layers(x)
        return x


model = OneLayerMLP(image_size, nb_neurons, 9)
print(model)

#############################################################################
# Then we configure the parameters of the training step and train the model.

from pynet.interfaces import DeepLearningInterface

cl = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=1e-4,
    loss_name="NLLLoss",
    metrics=["accuracy"],
    model=model)
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=10,
    checkpointdir="/tmp/orientation",
    fold_index=0,
    with_validation=True)

#############################################################################
# We focus now on test predictions.

import numpy as np
from pprint import pprint
from sklearn.metrics import classification_report
from pynet.plotting import plot_data

y_pred, X, y_true, loss, values = cl.testing(
    manager=manager,
    with_logit=True,
    predict=True)
pprint(data.labels)
X = X.reshape(-1, data.height, data.width)
X = np.expand_dims(X, axis=1)
print(classification_report(y_true, y_pred, target_names=data.labels.values()))
titles = ["{0}-{1}".format(data.labels[it1], data.labels[it2])
          for it1, it2 in zip(y_pred, y_true)]
plot_data(X, labels=titles, nb_samples=5)

#############################################################################
# Watch learning curves.
# During the training, we saved the loss and the accuracy at each iteration in
# the lists losses and accuracies.
# The following lines display the corresponding curves.

from pynet.plotting import plot_history

print(train_history)
plot_history(train_history)

#############################################################################
# Convolutional neural network
# ----------------------------
#
# Now we will create a neural network using convolutional layers.


class My_Net(torch.nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.linear = torch.nn.Linear(16 * 16 * 16, 9)

    def forward(self, X):
        X = X.view(-1, 1, 64, 64)
        X = torch.nn.functional.relu(self.conv1(X))
        X = torch.nn.functional.relu(self.conv2(X))
        X = self.maxpool(X)
        X = self.linear(X.view(-1, 16 * 16 * 16))
        return torch.nn.functional.log_softmax(X, dim=1)

#############################################################################
# Here, we check how the input size changes through each layer.


model = My_Net()

X_train = sample
X = torch.from_numpy(X_train[:10, :]).float()
print(X.shape)
X = torch.nn.functional.relu(model.conv1(X))
print(X.shape)
X = torch.nn.functional.relu(model.conv2(X))
print(X.shape)
X = model.maxpool(X)
print(X.shape)
X = model.linear(X.view(-1, 16 * 16 * 16))
print(X.shape)
X = torch.nn.functional.log_softmax(X, dim=1)
print(X.shape)

#############################################################################
# Then we configure the parameters of the training step and train the model.

cl = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=1e-5,
    loss_name="NLLLoss",
    metrics=["accuracy"],
    model=model)
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=10,
    checkpointdir="/tmp/orientation",
    fold_index=0,
    with_validation=True)

#############################################################################
# We focus now on test predictions.

import numpy as np
from pprint import pprint
from sklearn.metrics import classification_report
from pynet.plotting import plot_data

y_pred, X, y_true, loss, values = cl.testing(
    manager=manager,
    with_logit=True,
    predict=True)
pprint(data.labels)
X = X.reshape(-1, data.height, data.width)
X = np.expand_dims(X, axis=1)
print(classification_report(y_true, y_pred, target_names=data.labels.values()))
titles = ["{0}-{1}".format(data.labels[it1], data.labels[it2])
          for it1, it2 in zip(y_pred, y_true)]
plot_data(X, labels=titles, nb_samples=5)

#############################################################################
# Watch learning curves.
# During the training, we saved the loss and the accuracy at each iteration in
# the lists losses and accuracies.
# The following lines display the corresponding curves.

from pynet.plotting import plot_history

print(train_history)
plot_history(train_history)

#############################################################################
# Compare the fully-connected network with the CNN
# ------------------------------------------------
#
# Below is a comparison in terms of trainable parameters for both models.

cnn_model = My_Net()
print("Number of parameters in the CNN: ",
      sum(p.numel() for p in cnn_model.parameters()))

image_size = data.height * data.width
nb_neurons = 16
model = OneLayerMLP(image_size, nb_neurons, 9)
print("Number of parameters in the fully connected: ",
      sum(p.numel() for p in model.parameters()))

import os
if "CI_MODE" not in os.environ:
    import matplotlib.pyplot as plt
    plt.show()
