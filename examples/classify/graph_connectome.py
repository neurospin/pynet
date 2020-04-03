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
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import collections
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import scipy
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
for data in (x_train, y_train, x_test, y_test, x_valid, y_valid):
    print(data.shape)

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
    test_inputs=x_test, test_labels=y_valid,
    batch_size=5)
interfaces = pynet.get_interfaces()["graph"]
net_params = pynet.NetParameters(
    input_shape=(90, 90),
    in_channels=1,
    num_classes=2)
loss = pynet.get_tools()["losses"]["MSELoss"]()
model = interfaces["BrainNetCNNGraph"](
    net_params,
    optimizer_name="Adam",
    learning_rate=1e-4,
    weight_decay=1.1e-4,
    loss=loss)

def my_loss(x, y):
    logger.debug("Binary cross-entropy loss...")
    device = y.get_device()
    criterion = nn.BCEWithLogitsLoss()
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    y = y.type(torch.float32)
    if device != -1:
        y = y.to(device)
    logger.debug("  x: {0} - {1}".format(x.shape, x.dtype))
    logger.debug("  y: {0} - {1}".format(y.shape, y.dtype))
    return criterion(x, y)

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
    nb_epochs=100,
    checkpointdir=None,
    fold_index=0,
    scheduler=scheduler,
    with_validation=True)
y_pred, X, y_true, loss, values = model.testing(
    manager=manager,
    with_logit=True,
    predict=False)
y_pred = y_pred[:, 1]
y_true = y_true[:, 1]
result = pd.DataFrame.from_dict(collections.OrderedDict([
    ("pred", (y_pred > 0.5).astype(int)),
    ("truth", y_true),
    ("prob", y_pred)]))
print(result)
print(classification_report(y_true, y_pred > 0.5))

fig, ax = plt.subplots()
cmap = plt.get_cmap('Blues')
cm = SKMetrics("confusion_matrix", with_logit=False)(y_pred, y_true)
sns.heatmap(cm, cmap=cmap, annot=True, fmt="g", ax=ax)
ax.set_xlabel("predicted values")
ax.set_ylabel("actual values")
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")

plt.show()
