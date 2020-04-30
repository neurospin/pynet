"""
pynet: predicting autism
========================

Credit: A Grigis

This practice is based on the IMPAC challenge,
https://paris-saclay-cds.github.io/autism_challenge.

Autism spectrum disorder (ASD) is a severe psychiatric disorder that affects
1 in 166 children. In the IMPAC challenge ML models were trained using the
database's derived anatomical and functional features to diagnose a subject
as autistic or healthy. We propose here to implement the best neural network
to achieve this task and proposed in
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6859452, ie. a dense feedforward
network.
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()
import logging
import pynet
from pynet.datasets import fetch_impac
from pynet.datasets import DataManager
from pynet.utils import setup_logging
from pynet.plotting import Board, update_board
from pynet.interfaces import DeepLearningInterface
from pynet.metrics import SKMetrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import collections
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

setup_logging(level="info")
logger = logging.getLogger("pynet")

use_toy = False
dtype = "all"

data = fetch_impac(
    datasetdir="/tmp/impac",
    mode="train",
    dtype=dtype)
nb_features = data.nb_features
manager = DataManager(
    input_path=data.input_path,
    labels=["participants_asd"],
    metadata_path=data.metadata_path,
    number_of_folds=3,
    batch_size=128,
    sampler="random",
    test_size=2,
    sample_size=1)

if use_toy:
    toy_data = {}
    nb_features = 50
    for name, nb_samples in (("train", 1000), ("test", 2)):
        x1 = torch.randn(nb_samples, 50)
        x2 = torch.randn(nb_samples, 50) + 1.5
        x = torch.cat([x1, x2], dim=0)
        y1 = torch.zeros(nb_samples, 1)
        y2 = torch.ones(nb_samples, 1)
        y = torch.cat([y1, y2], dim=0)
        toy_data[name] = (x, y)
        if name == "train":
            plt.figure()
            plt.scatter(x1[:, 0], x1[:, 1], color="b")
            plt.scatter(x2[:, 0], x2[:, 1], color="r")
    manager = DataManager.from_numpy(
        train_inputs=toy_data["train"][0], train_labels=toy_data["train"][1],
        batch_size=50, test_inputs=toy_data["test"][0],
        test_labels=toy_data["test"][1])


class DenseFeedForwardNet(nn.Module):
    def __init__(self, nb_features):
        """ Initialize the instance.

        Parameters
        ----------
        nb_features: int
            the size of the feature vector.
        """
        super(DenseFeedForwardNet, self).__init__()
        self.layers = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(nb_features, 64)),
            ("relu1", nn.LeakyReLU(negative_slope=0.01)),
            ("dropout", nn.Dropout(0.13)),
            ("linear2", nn.Linear(64, 64)),
            ("relu2", nn.LeakyReLU(negative_slope=0.01)),
            ("linear3", nn.Linear(64, 1))
        ]))
        self.layers_alt = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(nb_features, 50)),
            ("relu1", nn.ReLU()),
            ("dropout", nn.Dropout(0.2)),
            ("linear2", nn.Linear(50, 100)),
            ("relu2", nn.PReLU(1)),
            ("linear3", nn.Linear(100, 1))
        ]))

    def forward(self, x):
        return self.layers(x)


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


def plot_metric_rank_correlations(metrics):
    """ Display rank correlations for all numerical metrics calculated over N
    experiments.

    Parameters
    ----------
    metrics: DataFrame
        a data frame with all computedd metrics as columns and N rows.
    """
    fig, ax = plt.subplots()
    labels = metrics.columns
    sns.heatmap(spearmanr(metrics)[0], annot=True, cmap=plt.get_cmap("Blues"),
                xticklabels=labels, yticklabels=labels, ax=ax)


model = DenseFeedForwardNet(nb_features)
print(model)
cl = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=1e-4,
    weight_decay=1.1e-4,
    metrics=["binary_accuracy", "sk_roc_auc_score"],
    loss=my_loss,
    model=model)
cl.board = Board(port=8097, host="http://localhost", env="main")
cl.add_observer("after_epoch", update_board)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=cl.optimizer,
    mode="min",
    factor=0.1,
    patience=5,
    verbose=True,
    eps=1e-8)
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=200,
    checkpointdir=None,
    fold_index=0,
    scheduler=scheduler,
    with_validation=(not use_toy))

if not use_toy:
    data = fetch_impac(
        datasetdir="/neurospin/nsap/datasets/impac",
        mode="test",
        dtype=dtype)
    manager = DataManager(
        input_path=data.input_path,
        labels=["participants_asd"],
        metadata_path=data.metadata_path,
        number_of_folds=3,
        batch_size=10,
        sampler=None,
        test_size=1,
        sample_size=1)
y_pred, X, y_true, loss, values = cl.testing(
    manager=manager,
    with_logit=True,
    logit_function="sigmoid",
    predict=False)
result = pd.DataFrame.from_dict(collections.OrderedDict([
    ("pred", (y_pred.squeeze() > 0.5).astype(int)),
    ("truth", y_true.squeeze()),
    ("prob", y_pred.squeeze())]))
print(result)
fig, ax = plt.subplots()
cmap = plt.get_cmap('Blues')
cm = SKMetrics("confusion_matrix", with_logit=False)(y_pred, y_true)
sns.heatmap(cm, cmap=cmap, annot=True, fmt="g", ax=ax)
ax.set_xlabel("predicted values")
ax.set_ylabel("actual values")
metrics = {}
sk_metrics = dict(
    (key, val) for key, val in pynet.get_tools()["metrics"].items()
    if key.startswith("sk_"))
for name, metric in sk_metrics.items():
    metric.with_logit = False
    value = metric(y_pred, y_true)
    metrics.setdefault(name, []).append(value)
metrics = pd.DataFrame.from_dict(metrics)
print(classification_report(y_true, y_pred >= 0.4))
print(metrics)
# plot_metric_rank_correlations(metrics)
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
