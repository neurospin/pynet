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
from pynet.datasets import DataManager, fetch_registration
from pynet.utils import setup_logging
from pynet.registration import VoxelMorphRegister
from torch.optim import lr_scheduler
from pynet.plotting import plot_history
from pynet.history import History
from pynet.losses import mse_loss, ncc_loss, gradient_loss
import matplotlib.pyplot as plt

setup_logging(level="info")
logger = logging.getLogger("pynet")

outdir = "/neurospin/nsap/tmp/registration_ncc"
data = fetch_registration(
    datasetdir=outdir)
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    number_of_folds=10,
    batch_size=1,
    sampler="random",
    #stratify_label="centers",
    test_size=0.1,
    add_input=True,
    sample_size=1)

#############################################################################
# Training
# --------
#
# From the available models load the VoxelMorphRegister, and start the
# training.

net = VoxelMorphRegister(
    vol_size=(128, 128, 128),
    enc_nf=[16, 32, 32, 32],
    dec_nf=[32, 32, 32, 32, 32, 16, 16],
    full_size=True,
    optimizer_name="Adam",
    learning_rate=1e-4,
    # weight_decay=1e-5,
    loss=mse_loss, # ncc_loss,
    use_cuda=True)
print(net.model)

def flow_regularizer(signal):
    logger.debug("Compute flow regularizattion...")
    lambda1 = 0.01  # recommend 1.0 for ncc, 0.01 for mse
    flow = signal.layer_outputs["flow"]
    logger.debug("  lambda: {0}".format(lambda1))
    logger.debug("  flow: {0} - {1} - {2}".format(
        flow.shape, flow.get_device(), flow.dtype))
    grad_regularization = lambda1 * gradient_loss(flow)
    logger.debug("Done.")
    return grad_regularization
net.add_observer("regularizer", flow_regularizer)

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=net.optimizer,
    mode="min",
    factor=0.5,
    patience=5)
train_history, valid_history = net.training(
    manager=manager,
    nb_epochs=(1 if "CI_MODE" in os.environ else 150000),
    checkpointdir=outdir,
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
#y_pred = np.expand_dims(y_pred, axis=1)
#data = np.concatenate((y_pred, y_true, X), axis=1)
#plot_data(data, nb_samples=5)


if "CI_MODE" not in os.environ:
    plt.show()
