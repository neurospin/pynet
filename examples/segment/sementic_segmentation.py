"""
pynet echocardiography segmentation
===================================

Credit: A Grigis

pynet is a Python package related to deep learning and its application in
MRI mediacal data analysis. It is accessible to everybody, and is reusable
in various contexts. The project is hosted on github:
https://github.com/neurospin/pynet.

In this hands-on session, we will use this U-Net architecture to segment
2D echocardiography images. In particular, we will focus on the segmentation
of three adjacent cardiac structures: the left ventricle, the myocardium and
the right ventricle. The segmentation of these ultrasound images is
particularly difficult due to many sources of artifacts, the recognition
of the structures to segment, and the subjective delineation of the contours
(e.g. at the lower part of the myocardial segmentation mask).
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

#############################################################################
# Import the dataset
# ------------------
#
# You may need to change the 'datasetdir' parameter.

import os
import numpy as np
from pynet.datasets import DataManager, fetch_echocardiography
from pynet.plotting import plot_data
from pynet.utils import setup_logging

setup_logging(level="info")

data = fetch_echocardiography(
    datasetdir="/tmp/echocardiography")
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    output_path=data.output_path,
    number_of_folds=2,
    stratify_label="label",
    sampler="random",
    batch_size=10,
    test_size=0.1,
    sample_size=0.2)
dataset = manager["test"]
print(dataset.inputs.shape, dataset.outputs.shape)
data = np.concatenate((dataset.inputs, dataset.outputs), axis=1)
plot_data(data, nb_samples=5)


#############################################################################
# Optimisation
# ------------
#
# From the available models load the UNet, and start the training.
# You may need to change the 'outdir' parameter.

import torch
import torch.nn as nn
from pynet import NetParameters
from pynet.interfaces import DeepLabNetSegmenter, PSPNetSegmenter
from pynet.plotting import plot_history
from pynet.history import History


def my_loss(x, y):
    """ nn.CrossEntropyLoss expects a torch.LongTensor containing the class
    indices without the channel dimension.
    """
    # y = torch.sum(y, dim=1).type(torch.LongTensor)
    device = y.get_device()
    y = torch.argmax(y, dim=1).type(torch.LongTensor)
    if device != -1:
        y = y.to(device)
    criterion = nn.CrossEntropyLoss()
    return criterion(x, y)


outdir = "/tmp/echocardiography"
model = "pspnet"
if model == "pspnet":
    params = NetParameters(
        n_classes=4,
        sizes=(1, 2, 3, 6),
        psp_size=512,
        deep_features_size=256,
        backend="resnet18",
        drop_rate=0)
    net = PSPNetSegmenter(
        params,
        optimizer_name="Adam",
        learning_rate=5e-4,
        metrics=["multiclass_dice"],
        loss=my_loss,
        use_cuda=False)
else:
    params = NetParameters(
        n_classes=4,
        drop_rate=0)
    net = DeepLabNetSegmenter(
        params,
        optimizer_name="Adam",
        learning_rate=5e-4,
        metrics=["multiclass_dice"],
        loss=my_loss,
        use_cuda=False)
print(net.model)
train_history, valid_history = net.training(
    manager=manager,
    nb_epochs=10,
    checkpointdir=None,
    fold_index=0,
    with_validation=True)
print(train_history)
print(valid_history)
plot_history(train_history)


#############################################################################
# Testing
# -------
#
# Finaly use the testing set and check the results.

y_pred, X, y_true, loss, values = net.testing(
    manager=manager,
    with_logit=True,
    predict=True)
print(y_pred.shape, X.shape, y_true.shape)
y_pred = np.expand_dims(y_pred, axis=1)
data = np.concatenate((y_pred, y_true, X), axis=1)
plot_data(data, nb_samples=5, random=False)

if "CI_MODE" not in os.environ:
    import matplotlib.pyplot as plt
    plt.show()
