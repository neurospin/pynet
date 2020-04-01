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
import pynet
import logging
from pynet.datasets import fetch_impac
from pynet.datasets import DataManager
from pynet.utils import setup_logging
from pynet.plotting import Board, update_board
from pynet.interfaces import DeepLearningInterface
from sklearn.metrics import classification_report
from pynet.metrics import SKMetrics, BINARY_METRICS, METRICS
import collections
import torch
import torch.nn as nn
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

setup_logging(level="info")
logger = logging.getLogger("pynet")

use_toy = True

data = fetch_impac(
    datasetdir="/neurospin/nsap/datasets/impac",
    mode="train",
    dtype="all")
nb_features = 7710
manager = DataManager(
    input_path=data.input_path,
    labels=["participants_asd"],
    metadata_path=data.metadata_path,
    number_of_folds=3,
    batch_size=10,
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
        print(x.shape, y.shape)
        toy_data[name] = (x, y)
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
    labels =  metrics.columns
    sns.heatmap(spearmanr(metrics)[0], annot=True, cmap=plt.get_cmap("Blues"),
                xticklabels=labels, yticklabels=labels, ax=ax)

model = DenseFeedForwardNet(nb_features)
print(model)
extra_metric = METRICS["binary_accuracy"]
extra_metric.thr = 0.4
cl = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=5e-5,
    weight_decay=1.1e-4,
    metrics=[extra_metric],
    loss=my_loss,
    model=model)
cl.board = Board(port=8097, host="http://localhost", env="main") 
cl.add_observer("after_epoch", update_board)
outdir = "/tmp/impac"
if not os.path.isdir(outdir):
    os.mkdir(outdir)
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=60,
    checkpointdir=outdir,
    fold_index=0,
    with_validation=(not use_toy))

data = fetch_impac(
    datasetdir="/neurospin/nsap/datasets/impac",
    mode="test",
    dtype="all")
if not use_toy:
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
print(y_pred)
print(y_true)
fig, ax = plt.subplots()
cmap = plt.get_cmap('Blues')
cm = SKMetrics("confusion_matrix")(y_pred, y_true)
sns.heatmap(cm, cmap=cmap, annot=True, fmt="g", ax=ax)
ax.set_xlabel("predicted values")
ax.set_ylabel("actual values")
metrics = {}
for name, metric in BINARY_METRICS.items():
    value = metric(y_pred, y_true)
    metrics.setdefault(name, []).append(value)
    metrics.setdefault(name, []).append(value)
metrics = pd.DataFrame.from_dict(metrics)
metrics["brier_loss"] *= -1.0
metrics["log_loss"] *= -1.0
metrics["false_discovery_rate"] *= -1.0
metrics["false_negative_rate"] *= -1.0
metrics["false_positive_rate"] *= -1.0
print(classification_report(y_true, y_pred >= 0.4))
print(metrics)
#plot_metric_rank_correlations(metrics)

#plt.show()
