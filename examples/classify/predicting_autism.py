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
from pynet.interfaces import DeepLearningInterface
from sklearn.metrics import classification_report
import collections
import torch
import torch.nn as nn

setup_logging(level="info")
logger = logging.getLogger("pynet")

data = fetch_impac(
    datasetdir="/neurospin/nsap/datasets/impac",
    mode="train")
manager = DataManager(
    input_path=data.input_path,
    labels=["participants_asd"],
    metadata_path=data.metadata_path,
    number_of_folds=10,
    batch_size=10,
    test_size=2,
    sample_size=1)

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
            ("relu1", nn.ReLU()),
            ("dropout", nn.Dropout(0.13)),
            ("linear2", nn.Linear(64, 64)),
            ("relu2", nn.ReLU()),
            ("linear2", nn.Linear(64, 2)),
            ("softmax", nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x): 
        x = self.layers(x)
        return x

def my_loss(x, y):
    logger.debug("NLLLoss...")
    criterion = nn.NLLLoss()
    logger.debug("  x: {0}".format(x.shape))
    logger.debug("  y: {0}".format(y.shape))
    return criterion(x, y)

model = DenseFeedForwardNet(nb_features=7710)
print(model)
cl = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=1e-4,
    weight_decay=1.1e-4,
    loss=my_loss,
    metrics=["accuracy"],
    model=model)
outdir = "/tmp/impac"
if not os.path.isdir(outdir):
    os.mkdir(outdir)
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=10,
    checkpointdir=outdir,
    fold_index=0,
    with_validation=True)


data = fetch_impac(
    datasetdir="/neurospin/nsap/datasets/impac",
    mode="test")
manager = DataManager(
    input_path=data.input_path,
    labels=["participants_asd"],
    metadata_path=data.metadata_path,
    number_of_folds=10,
    batch_size=10,
    test_size=1,
    sample_size=1)
y_pred, X, y_true, loss, values = cl.testing(
    manager=manager,
    with_logit=False,
    predict=True)
print(classification_report(y_true, y_pred))

