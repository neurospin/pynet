"""
pynet: predicting autism
========================

Credit: A Grigis

This practice is based on
https://www2.cs.sfu.ca/~hamarneh/ecopy/neuroimage2017.pdf.

We assessed BrainNetCNN ability to learn and discriminate between differing
network topologies using sets of synthetically generated networks. We first
examined the performance of BrainNetCNN on data with increasing levels of
noise and then compared BrainNetCNN to a fully-connected neural network with
the same number of model parameters.
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()
import logging
import shutil
import pynet
from pynet.datasets import DataManager, get_fetchers
from pynet.utils import setup_logging
from pynet.metrics import SKMetrics
from pynet.plotting import Board, update_board
from mne.viz import circular_layout, plot_connectivity_circle
import collections
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import scipy
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

setup_logging(level="info")
logger = logging.getLogger("pynet")

# Load the data
outdir = "/tmp/graph_connectome"
(injury, x_train, y_train, x_test, y_test, x_valid,
 y_valid) = get_fetchers()["fetch_connectome"](outdir)
labels = [str(idx) for idx in range(1, x_train.shape[-1] + 1)]
for name, (x, y) in (("train", (x_train, y_train)),
                     ("test", (x_test, y_test)),
                     ("validation", (x_valid, y_valid))):
    print("{0}: x {1} - y {2}".format(name, x.shape, y.shape))

# View the realistic base connectome and the injury signatures.
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
plt.imshow(injury.X_mn, interpolation="None")
plt.colorbar()
plt.title("base connectome")
plt.subplot(1, 3, 2)
plt.imshow(injury.sigs[0], interpolation="None")
plt.colorbar()
plt.title("signature 1")
plt.subplot(1, 3, 3)
plt.imshow(injury.sigs[1], interpolation="None")
plt.colorbar()
plt.title("signature 2")

# Show example noisy training data that have the signatures applied.
# It's not obvious to the human eye the subtle differences, but the cross
# row and column above perturbed the below matrices with the y weights.
# Show in the title how much each signature is weighted by.
plt.figure(figsize=(16, 4))
for idx in range(3):
    plt.subplot(1, 3, idx + 1)
    plt.imshow(np.squeeze(x_train[idx]), interpolation="None")
    plt.colorbar()
    plt.title(y_train[idx])

manager = DataManager.from_numpy(
    train_inputs=x_train, train_labels=y_train,
    validation_inputs=x_valid, validation_labels=y_valid,
    test_inputs=x_test, test_labels=y_test,
    batch_size=128, continuous_labels=True)
interfaces = pynet.get_interfaces()["graph"]
net_params = pynet.NetParameters(
    input_shape=(90, 90),
    in_channels=1,
    num_classes=2,
    nb_e2e=32,
    nb_e2n=64,
    nb_n2g=30,
    dropout=0.5,
    leaky_alpha=0.1,
    twice_e2e=False,
    dense_sml=True)
my_loss = pynet.get_tools()["losses"]["MSELoss"]()
model = interfaces["BrainNetCNNGraph"](
    net_params,
    optimizer_name="Adam",
    learning_rate=0.01,
    weight_decay= 0.0005,
    loss_name="MSELoss")
model.board = Board(port=8097, host="http://localhost", env="main")
model.add_observer("after_epoch", update_board)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=model.optimizer,
    mode="min",
    factor=0.1,
    patience=5,
    verbose=True,
    eps=1e-8)
test_history, train_history = model.training(
    manager=manager,
    nb_epochs=15,
    checkpointdir=None,
    fold_index=0,
    scheduler=scheduler,
    with_validation=True)
y_pred, X, y_true, loss, values = model.testing(
    manager=manager,
    with_logit=True,
    predict=False)
y_pred_0, y_pred_1 = y_pred.T
y_true_0, y_true_1 = y_true.T
result = pd.DataFrame.from_dict(collections.OrderedDict([
    ("pred_0", y_pred_0),
    ("truth_0", y_true_0),
    ("pred_1", y_pred_1),
    ("truth_1", y_true_1)]))
print(result)


def regression_metrics(pred_labels, true_labels):
    """ Regression metrics as deefined is the tutorial.
    """
    met = {}
    met["mad"] = np.mean((abs(pred_labels - true_labels)))
    met["std_mad"] = np.std(abs(pred_labels - true_labels))
    # There's multiple labels.
    if np.shape(np.squeeze(pred_labels).shape)[0] > 1:
        n_labels = pred_labels.shape[1]
        for idx in range(n_labels):
            pred_values = pred_labels[:, idx]
            actual_values = true_labels[:, idx]
            r, p = pearsonr(pred_values, actual_values)
            met["corr_" + str(idx)] = r
            met["p_" + str(idx)] = p
    # Only 1 label.
    else:
        r, p = pearsonr(pred_labels, true_labels)
        met["corr_0"] = r
        met["p_0"] = p
    return met

print("E2E prediction results:")
test_metrics_0 = regression_metrics(y_pred_0, y_true_0)
print("class 0: {0}".format(test_metrics_0))
test_metrics_1 = regression_metrics(y_pred_1, y_true_1)
print("class 1: {0}".format(test_metrics_1))

# Saliency map is the gradient of the maximum score value with respect to
# the input image.
model.model.eval()
X = torch.from_numpy(x_test)
X.requires_grad_()
scores = model.model(X)
scores.backward(torch.ones(scores.shape, dtype=torch.float32))
saliency, _ = torch.max(X.grad.data.abs(), dim=1)
saliency = np.mean(saliency.numpy(), axis=0)

hemi_size = len(labels) // 2
node_order = labels[:hemi_size]
node_order.extend(labels[hemi_size:][::-1])
node_angles = circular_layout(labels, node_order, start_pos=90,
                              group_boundaries=[0, hemi_size])
plot_connectivity_circle(saliency, labels, n_lines=300,
                         node_angles=node_angles,
                         title="Partial derivatives mapped on a circle plot")

plt.show()
