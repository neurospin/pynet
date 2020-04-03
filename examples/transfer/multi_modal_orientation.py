"""
pynet: transfer learning
========================

Credit: A Grigis
Based on:
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

In this tutorial, you will learn how to train your network using transfer
learning on the multi modal orientation prediction dataset. Why? Because in
many cases we do not have enough data to train the network from scratch.

We will use a network trained on the imagenet dataset and freeze the weights
for a part/all of the network (except that of the final fully connected layer).

Read the data
-------------
"""

import os
import sys

if "CI_MODE" in os.environ:
    sys.exit()

from pynet.datasets import fetch_orientation
from pynet.datasets import DataManager
from skimage.color import gray2rgb


def prepare(arr):
    arr = gray2rgb(arr.reshape((64, 64)))
    arr = arr.transpose(2, 0, 1)
    return arr


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
    sample_size=(0.1 if "CI_MODE" not in os.environ else 0.1),
    input_transforms=[prepare])

#############################################################################
# Displaying some images of the test dataset.

from pynet.plotting import plot_data
import numpy as np

dataset = manager["test"]
sample = dataset.inputs.reshape(-1, data.height, data.width)
sample = np.expand_dims(sample, axis=1)
plot_data(sample, nb_samples=5)


#############################################################################
# Load the model
# --------------
#
# Load the model and fix all weights.
# Change the last linear layer.

import pynet.interfaces as interfaces
from pynet import NetParameters
from pynet.utils import get_named_layers, freeze_layers, reset_weights
import torch.nn as nn

net_params = NetParameters(
    num_classes=1000)
cl = interfaces.ResNet18Classifier(
    net_params,
    pretrained="/neurospin/nsap/torch/models/resnet18-5c106cde.pth",
    optimizer_name="Adam",
    learning_rate=1e-4,
    loss_name="NLLLoss",
    metrics=["accuracy"])
print(cl.model)
layers = get_named_layers(cl.model, allowed_layers=[nn.Module], resume=True)
print(layers.keys())
to_freeze_layers = [
    "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3",
    "layer4"]
freeze_layers(cl.model, to_freeze_layers)
nb_features = cl.model.fc.in_features
cl.model.fc = nn.Linear(nb_features, 9)
print(cl.model)

#############################################################################
# Retrain the model
# -----------------
#
# Train the model

from pynet.plotting import plot_history


def train(cl, dataset):

    state = dict(
        (key, val)
        for key, val in cl.model.state_dict().items()
        if key.endswith(".weight"))
    test_history, train_history = cl.training(
        manager=manager,
        nb_epochs=5,
        checkpointdir=None,
        fold_index=0,
        with_validation=False)
    train_state = dict(
        (key, val)
        for key, val in cl.model.state_dict().items()
        if key.endswith(".weight"))
    for key, val in state.items():
        if not np.allclose(val, train_state[key]):
            print("--", key)

    idx = 0
    y_pred_prob, X, y_true, loss, values = cl.testing(
        manager=manager,
        with_logit=True,
        predict=False)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print(" ** true label      : ", y_true[idx])
    print(" ** predicted label : ", y_pred[idx])
    titles = ["{0}-{1}".format(data.labels[it1], data.labels[it2])
              for it1, it2 in zip(y_pred, y_true)]
    plot_data(X, labels=titles, nb_samples=5)
    plot_history(train_history)


train(cl, dataset)

#############################################################################
# Test different strategies
# -------------------------
#
# OK it's not working, let's try different transfer learning strategies.

cl = interfaces.ResNet18Classifier(
    net_params,
    pretrained="/neurospin/nsap/torch/models/resnet18-5c106cde.pth",
    optimizer_name="Adam",
    learning_rate=1e-4,
    loss_name="NLLLoss",
    metrics=["accuracy"])
to_freeze_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
freeze_layers(cl.model, to_freeze_layers)
nb_features = cl.model.fc.in_features
cl.model.fc = nn.Linear(nb_features, 9)
train(cl, dataset)

cl = interfaces.ResNet18Classifier(
    net_params,
    pretrained="/neurospin/nsap/torch/models/resnet18-5c106cde.pth",
    optimizer_name="Adam",
    learning_rate=1e-4,
    loss_name="NLLLoss",
    metrics=["accuracy"])
reset_weights(cl.model)
nb_features = cl.model.fc.in_features
cl.model.fc = nn.Linear(nb_features, 9)
train(cl, dataset)

if "CI_MODE" not in os.environ:
    import matplotlib.pyplot as plt
    plt.show()
