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

import copy
import numpy as np
import os
import glob

from pynet.dataset import split_dataset
from pynet.dataset import LoadDataset
from pynet.optim import training
from pynet.utils import get_named_layers
from pynet.utils import freeze_layers
from pynet.utils import reset_weights
import pynet.classifier as classifier

import torch
import torch.nn as nn


# To plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["axes.labelsize"]  = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
def plot_image(image):
    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image.astype(np.int)
    plt.imshow(image, aspect="equal", interpolation="nearest")
    plt.axis("off")

datadir = "/neurospin/nsap/hackathon/deeplearning-hackathon-2019/tp4_data"
dataset_desc = os.path.join(datadir, "dataset.tsv")
height = 64
width  = 64
med_view = {
        "T1-A": 0,
        "T1-S": 1,
        "T1-C": 2,
        "T2-A": 3,
        "T2-S": 4,
        "T2-C": 5,
        "CT-A": 6,
        "CT-S": 7,
        "CT-C": 8
}
rev_med_view = dict((val, key) for key, val in med_view.items())
dataloader_kwargs = {
    "flatten": False,
    "squeeze_channel": False,
    "gray_to_rgb": True,
    "load": True
}
dataset = split_dataset(
    path=dataset_desc,
    dataloader=LoadDataset,
    inputs=["slice"],
    outputs=None,
    label="label",
    number_of_folds=1,
    batch_size=-1,
    transforms=None,
    test_size=0.25,
    validation_size=0.1,
    nb_samples=1000,
    verbose=0,
    **dataloader_kwargs)

fold_index = 0
batch_index = 0
X_train = dataset["train"][fold_index][batch_index]["inputs"]
y_train = dataset["train"][fold_index][batch_index]["labels"]
X_valid = dataset["validation"][fold_index][batch_index]["inputs"]
y_valid = dataset["validation"][fold_index][batch_index]["labels"]
X_test = dataset["test"][batch_index]["inputs"]
y_test = dataset["test"][batch_index]["labels"]
print(X_train.shape , y_train.shape)
print(X_valid.shape, y_valid.shape )
print(X_test.shape, y_test.shape)
print(X_train.dtype, y_train.dtype)

nb_x = 5
nb_y = 5
plt.figure(figsize=(15, 15 * nb_x / nb_y), dpi=100)
for cnt in range(nb_x * nb_y):
    plt.subplot(nb_x, nb_y , cnt + 1)
    plot_image(X_train[cnt])
    plt.title(rev_med_view[y_train[cnt].item()])

#############################################################################
# Load the model
# --------------
#
# Load the model and fix all weights.
# Change the last linear layer.

cl = classifier.ResNet18(
    num_classes=1000,
    pretrained="/neurospin/nsap/torch/models/resnet18-5c106cde.pth",
    batch_size=50,
    optimizer_name="Adam",
    learning_rate = 1e-4,
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

#############################################################################
# Retrain the model
# -----------------
#
# Train the model

def train(cl, dataset):

    state = dict(
        (key, val)
        for key, val in cl.model.state_dict().items() if key.endswith(".weight"))
    test_history, train_history = training(
        model=cl,
        dataset=dataset,
        nb_epochs=5,
        outdir=None,
        verbose=1)
    train_state = dict(
        (key, val)
        for key, val in cl.model.state_dict().items() if key.endswith(".weight"))
    for key, val in state.items():
        if not np.allclose(val, train_state[key]):
            print("--", key)

    i = 0
    test_input = X_test[i]
    test_input = np.expand_dims(test_input, axis=0)
    y_t = cl.predict_proba(test_input)
    print(" ** true label      : ", y_test[i] )
    print(" ** predicted label : ", np.argmax(y_t))
    plt.figure()
    plot_image(X_test[i])
    plt.title("true: " + rev_med_view[int(y_test[i])] +
              ", predict: " + rev_med_view[np.argmax(y_t)])

    _, losses = train_history["loss"]
    _, accuracies = train_history["accuracy"]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.ylabel("Training loss")
    plt.xlabel("Iterations")
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.ylabel("Training accuracy")
    plt.xlabel("Iterations")

train(cl, dataset)

#############################################################################
# Test different strategies
# -------------------------
#
# OK it's not working, let's try different transfer learning strategies.

cl = classifier.ResNet18(
    num_classes=1000,
    pretrained="/neurospin/nsap/torch/models/resnet18-5c106cde.pth",
    batch_size=50,
    optimizer_name="Adam",
    learning_rate = 1e-4,
    loss_name="NLLLoss",
    metrics=["accuracy"])
to_freeze_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
freeze_layers(cl.model, to_freeze_layers)
nb_features = cl.model.fc.in_features
cl.model.fc = nn.Linear(nb_features, 9)
train(cl, dataset)

cl = classifier.ResNet18(
    num_classes=1000,
    pretrained="/neurospin/nsap/torch/models/resnet18-5c106cde.pth",
    batch_size=50,
    optimizer_name="Adam",
    learning_rate = 1e-4,
    loss_name="NLLLoss",
    metrics=["accuracy"])
reset_weights(cl.model)
nb_features = cl.model.fc.in_features
cl.model.fc = nn.Linear(nb_features, 9)
train(cl, dataset)

plt.show()

