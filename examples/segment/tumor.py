"""
pynet metastasis tumor segmentation
===================================

Credit: A Grigis

pynet is a Python package related to deep learning and its application in
MRI mediacal data analysis. It is accessible to everybody, and is reusable
in various contexts. The project is hosted on github:
https://github.com/neurospin/pynet.

In this notebook we will learn how to segment tumors using MRI images from the
Brats dataset. The NvNet proposed by Andriy Myronenko's will be trained. This
network is a combination of a Vnet (3d Unet) and a VAE (variation
auto-encoder).
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

############################################################################
# Inspect the NvNet network
# --------------------------
#
# Inspect some layers of the network.

from pprint import pprint
import pynet.models as models
from pynet import NetParameters
from pynet.utils import get_named_layers
from pynet.utils import setup_logging
# from pynet.plotting.network import plot_net_rescue

setup_logging(level="info")

model = models.NvNet(
    input_shape=(128, 128, 128),
    in_channels=1,
    num_classes=4,
    activation="relu",
    normalization="group_normalization",
    mode="trilinear",
    with_vae=True)
layers = get_named_layers(model)
pprint(layers)
# graph_file = plot_net_rescue(model, (1, 1, 128, 128, 128))

#############################################################################
# Import the brats dataset
# ------------------------
#
# Use the fetcher of the pynet package.

from pynet.datasets import DataManager, fetch_brats
from pynet.plotting import plot_data

data = fetch_brats(
    datasetdir="/neurospin/nsap/datasets/brats")
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    output_path=data.output_path,
    projection_labels=None,
    number_of_folds=10,
    batch_size=1,
    stratify_label="grade",
    # input_transforms=[
    #     RandomFlipDimensions(ndims=3, proba=0.5, with_channels=True),
    #     Offset(nb_channels=4, factor=0.1)],
    sampler="random",
    add_input=True,
    test_size=0.1,
    pin_memory=True)
dataset = manager["test"][:1]
print(dataset.inputs.shape, dataset.outputs.shape)
plot_data(dataset.inputs, channel=1, nb_samples=5)
plot_data(dataset.outputs, channel=1, nb_samples=5)

#############################################################################
# Training
# --------
#
# From the available models load the 3D NvNet, and start the training.

import os
from torch.optim import lr_scheduler
from pynet.losses import NvNetCombinedLoss
from pynet.interfaces import NvNetSegmenter
from pynet.plotting import plot_history
from pynet.history import History

my_loss = NvNetCombinedLoss(
    num_classes=4,
    k1=0.1,
    k2=0.1)
outdir = "/neurospin/nsap/tmp/nvnet"
if not os.path.isdir(outdir):
    os.mkdir(outdir)
trained_model = os.path.join(outdir, "model_0_epoch_99.pth")
nvnet_params = NetParameters(
    input_shape=(150, 190, 135),
    in_channels=4,
    num_classes=4,
    activation="relu",
    normalization="group_normalization",
    mode="trilinear",
    with_vae=True)

if os.path.isfile(trained_model):
    nvnet = NvNetSegmenter(
        nvnet_params,
        optimizer_name="Adam",
        learning_rate=1e-4,
        weight_decay=1e-5,
        loss=my_loss,
        pretrained=trained_model,
        use_cuda=True)
    train_history = History.load(
        os.path.join(outdir, "train_0_epoch_9.pkl"))
    valid_history = History.load(
        os.path.join(outdir, "validation_0_epoch_9.pkl"))
else:
    nvnet = NvNetSegmenter(
        nvnet_params,
        optimizer_name="Adam",
        learning_rate=1e-4,
        weight_decay=1e-5,
        loss=my_loss,
        use_cuda=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=nvnet.optimizer,
        mode="min",
        factor=0.5,
        patience=5)
    train_history, valid_history = nvnet.training(
        manager=manager,
        nb_epochs=100,
        checkpointdir=outdir,
        # fold_index=0,
        scheduler=scheduler,
        with_validation=True)
print(train_history)
print(valid_history)
plot_history(train_history)

#############################################################################
# Testing
# -------
#
# Finaly use the testing set and check the results.

y_pred, X, y_true, loss, values = nvnet.testing(
    manager=manager,
    with_logit=False,
    predict=False)
print(y_pred.shape, X.shape, y_true.shape)
# y_pred = np.expand_dims(y_pred, axis=1)
# data = np.concatenate((y_pred, y_true, X), axis=1)
# plot_data(data, nb_samples=5)

if "CI_MODE" not in os.environ:
    import matplotlib.pyplot as plt
    plt.show()
