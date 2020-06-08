"""
Practical Deep Learning for Image Registration
==============================================

Credit: A Grigis

Load the data
-------------

Load some data.
You may need to change the 'outdir' parameter.
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()
import logging
import numpy as np
from pynet import NetParameters
from pynet.datasets import DataManager, fetch_registration
from pynet.utils import setup_logging
from pynet.interfaces import (
    VoxelMorphNetRegister, ADDNetRegister, VTNetRegister, RCNetRegister)
from pynet.models.voxelmorphnet import FlowRegularizer
from pynet.models.vtnet import ADDNetRegularizer
from torch.optim import lr_scheduler
from pynet.plotting import plot_history
from pynet.history import History
from pynet.losses import MSELoss, NCCLoss, RCNetLoss, PCCLoss
from pynet.plotting import Board, update_board
import matplotlib.pyplot as plt

setup_logging(level="debug")
logger = logging.getLogger("pynet")

outdir = "/neurospin/nsap/tmp/registration"
data = fetch_registration(
    datasetdir=outdir)
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=8,
    sampler="random",
    stratify_label="studies",
    projection_labels={"studies": ["abide"]},
    test_size=0.1,
    add_input=True,
    sample_size=0.1)

#############################################################################
# Training
# --------
#
# From the available models load the VoxelMorphRegister, VTNetRegister or
# ADDNet  and start the training.
# Note that the two first estimate a non linear deformation and require
# the input data to be afinely registered. The ADDNet estimate an affine
# transform. We will see in the next section how to combine them in an
# efficient way.

base_network = "rcnet"  # "vtnet"  # "addnet"

if base_network == "rcnet":
    rcnet_params = NetParameters(
        input_shape=(128, 128, 128),
        in_channels=2,
        base_network="VTNet",
        n_cascades=1,
        rep=1)
    net = RCNetRegister(
        rcnet_params,
        optimizer_name="Adam",
        learning_rate=1e-4,
        loss=RCNetLoss(),
        use_cuda=True)
elif base_network == "addnet":
    addnet_params = NetParameters(
        input_shape=(128, 128, 128),
        in_channels=2,
        kernel_size=3,
        padding=1,
        flow_multiplier=1.)
    net = ADDNetRegister(
        addnet_params,
        optimizer_name="Adam",
        learning_rate=1e-4,
        loss=PCCLoss(concat=True),
        use_cuda=True)
    regularizer = ADDNetRegularizer(k1=0.1, k2=0.1)
    net.add_observer("regularizer", regularizer)
elif base_network == "vtnet":
    vtnet_params = NetParameters(
        input_shape=(128, 128, 128),
        in_channels=2,
        kernel_size=3,
        padding=1,
        flow_multiplier=1.,
        nb_channels=16)
    net = VTNetRegister(
        vtnet_params,
        optimizer_name="Adam",
        learning_rate=1e-4,
        loss=PCCLoss(concat=True),  # MSELoss(concat=True),
        use_cuda=True)
    flow_regularizer = FlowRegularizer(k1=1.)
    net.add_observer("regularizer", flow_regularizer)
else:
    vmnet_params = NetParameters(
        vol_size=(128, 128, 128),
        enc_nf=[16, 32, 32, 32],
        dec_nf=[32, 32, 32, 32, 32, 16, 16],
        full_size=True)
    net = VoxelMorphNetRegister(
        vmnet_params,
        optimizer_name="Adam",
        learning_rate=1e-4,
        # weight_decay=1e-5,
        loss=MSELoss(concat=True),  # NCCLoss,
        use_cuda=False)
    flow_regularizer = FlowRegularizer(k1=0.01)
    net.add_observer("regularizer", flow_regularizer)
print(net.model)
def prepare_pred(y_pred):
    moving = y_pred[0, :, :, :, 64]
    validation_dataset = manager["validation"][0]
    corresponding_index = validation_dataset.indices[0]
    reference = validation_dataset.inputs[corresponding_index, 1:, :, : , 64]
    orginal = validation_dataset.inputs[corresponding_index, :1, :, : , 64]
    moving = np.expand_dims(moving, axis=1)
    reference = np.expand_dims(reference, axis=1)
    orginal = np.expand_dims(orginal, axis=1)
    moving = (moving / moving.max())
    moving = moving * 255
    reference = (reference / reference.max())
    reference = reference * 255
    orginal = (orginal / reference.max())
    orginal = orginal * 255
    return np.concatenate((moving, orginal, reference), axis=0)
net.board = Board(port=8097, host="http://localhost",
                  env=base_network, display_pred=True,
                  prepare_pred=prepare_pred)
net.add_observer("after_epoch", update_board)

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=net.optimizer,
    mode="min",
    factor=0.5,
    patience=4,
    verbose=True,
    min_lr=1e-7)
train_history, valid_history = net.training(
    manager=manager,
    nb_epochs=(1 if "CI_MODE" in os.environ else 150000),
    checkpointdir=None,  # outdir,
    fold_index=0,
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

y_pred, X, y_true, loss, values = net.testing(
    manager=manager,
    with_logit=False,
    predict=False,
    concat_layer_outputs=["flow"])
print(y_pred.shape, X.shape, y_true.shape)
# y_pred = np.expand_dims(y_pred, axis=1)
# data = np.concatenate((y_pred, y_true, X), axis=1)
# plot_data(data, nb_samples=5)

if "CI_MODE" not in os.environ:
    plt.show()
