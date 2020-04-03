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
    sampler="weighted_random",
    batch_size=10,
    test_size=0.1,
    sample_size=(1 if "CI_MODE" not in os.environ else 0.05))
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
from pynet.interfaces import UNetEncoder
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
trained_model = os.path.join(outdir, "model_0_epoch_9.pth")
unet_params = NetParameters(
    num_classes=4,
    in_channels=1,
    depth=5,
    start_filts=16,
    up_mode="upsample",
    merge_mode="concat",
    batchnorm=False,
    dim="2d")
if os.path.isfile(trained_model):
    unet = UNetEncoder(
        unet_params,
        optimizer_name="Adam",
        learning_rate=5e-4,
        metrics=["multiclass_dice"],
        loss=my_loss,
        pretrained=trained_model,
        use_cuda=False)
    train_history = History.load(
        os.path.join(outdir, "train_0_epoch_9.pkl"))
    valid_history = History.load(
        os.path.join(outdir, "validation_0_epoch_9.pkl"))
else:
    unet = UNetEncoder(
        unet_params,
        optimizer_name="Adam",
        learning_rate=5e-4,
        metrics=["multiclass_dice"],
        loss=my_loss,
        use_cuda=False)
    print(unet.model)
    train_history, valid_history = unet.training(
        manager=manager,
        nb_epochs=(10 if "CI_MODE" not in os.environ else 1),
        checkpointdir=outdir,
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

y_pred, X, y_true, loss, values = unet.testing(
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
