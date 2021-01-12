"""
pynet: self supervised clustering
=================================

Credit: A Grigis
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

# Imports
import collections
import logging
import pynet
from pynet.metrics import SKMetrics
from pynet.datasets import DataManager
from pynet.interfaces import DeepLearningInterface
from pynet.interfaces import DeepClusterClassifier
from pynet.models import BrainNetCNN
from pynet.utils import setup_logging
from pynet.plotting import Board, update_board
from pynet.models.deepcluster import update_pseudo_labels
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
try:
    import faiss
except:
    pass


# Global Parameters
OUTDIR = "/tmp/graph_connectome"
BATCH_SIZE = 5
N_EPOCHS = 20
N_CLUSTERS = 2
N_SAMPLES = 40
AVOID_EMPTY_CLUSTERS = True
UNIFORM_SAMPLING = False
setup_logging(level="info")


# Load the data
data = []
labels = []
for idx in range(N_CLUSTERS):

    x_train = np.ones((N_SAMPLES, 1, 90, 90)) * idx
    x_train += (np.random.rand(*x_train.shape) - 0.5) * 0.01
    y_train = np.asarray([idx] * N_SAMPLES)
    print("sub data: x {0} - y {1}".format(x_train.shape, y_train.shape))
    data.append(x_train)
    labels.extend([idx] * len(x_train))

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

data = np.concatenate(data, axis=0)
labels = np.asarray(labels)
print("dataset: x {0} - y {1}".format(data.shape, labels.shape))


# Create data manager
manager = DataManager.from_numpy(
    train_inputs=data, train_labels=np.zeros(labels.shape),
    batch_size=BATCH_SIZE)


class FKmeans(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, data):
        n_data, d = data.shape
        self.clus = faiss.Kmeans(d, self.n_clusters)
        self.clus.seed = np.random.randint(1234)
        self.clus.niter = 20
        self.clus.max_points_per_centroid = 10000000
        self.clus.train(data)

    def predict(self, data):
        _, I = self.clus.index.search(data, 1)
        losses = self.clus.obj
        print("k-means loss evolution: {0}".format(losses))
        return np.asarray([int(n[0]) for n in I])


class UniformLabelSampler(Sampler):
    """ Samples elements uniformely accross pseudo labels.
    """
    def __init__(self, data_loader):
        """ Init class.

        Parameters
        ----------
        data_loader: DataLoader
            the train data loader that contains the pseudo labels.
        """
        self.n_samples = len(train_loader)
        self.data_loader = data_loader
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        """ Generate sampling indexes.
        """
        labels = self.data_loader.dataset.labels
        clusters_to_images = self.get_clusters(labels)
        n_non_empty_clusters = len(clusters_to_images)
        size_per_pseudolabel = int(self.n_samples / n_non_empty_clusters) + 1
        res = np.array([])
        for name, cluster_indexes in clusters_to_images.items():
            indexes = np.random.choice(
                cluster_indexes, size_per_pseudolabel,
                replace=(len(cluster_indexes) <= size_per_pseudolabel))
            res = np.concatenate((res, indexes))
        np.random.shuffle(res)
        res = list(res.astype("int"))
        if len(res) >= self.n_samples:
            return res[:self.n_samples]
        res += res[: (self.n_samples - len(res))]
        return res

    def get_clusters(self, labels):
        """ Get indexes associated to each cluster.
        """
        tally = collections.defaultdict(list)
        for idx, item in enumerate(labels):
            tally[item].append(idx)
        return tally

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


def my_loss(x, y):
    criterion = nn.CrossEntropyLoss()
    print("  x: {0} - {1}".format(x.shape, x.dtype))
    print(torch.argmax(x, dim=1))
    print("  y: {0} - {1}".format(y.shape, y.dtype))
    print(y)
    return criterion(x, y)


# Create model
train_loader = manager.get_dataloader(train=True, fold_index=0).train
if UNIFORM_SAMPLING:
    sampler = UniformLabelSampler(train_loader)
    raise ValueError("Uniform data sampling option not yet supported.")
if AVOID_EMPTY_CLUSTERS:
    kmeans = FKmeans(n_clusters=N_CLUSTERS)
else:
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=None,
        # verbose=100,
        max_iter=20)
net = BrainNetCNN(
    input_shape=(90, 90),
    in_channels=1,
    num_classes=N_CLUSTERS,
    nb_e2e=32,
    nb_e2n=64,
    nb_n2g=30,
    dropout=0,
    leaky_alpha=0.1,
    twice_e2e=False,
    dense_sml=False)
net_params = pynet.NetParameters(
    network=net,
    clustering=kmeans,
    data_loader=train_loader,
    n_batchs=10,
    pca_dim=6,
    assignment_logfile=None,
    use_cuda=False)
model = DeepClusterClassifier(
    net_params,
    optimizer_name="SGD",
    learning_rate=0.001,
    momentum=0.9,
    weight_decay=10**-5,
    # loss=my_loss)
    loss_name="CrossEntropyLoss")
model.board = Board(port=8097, host="http://localhost", env="deepcluster")
model.add_observer("before_epoch", update_pseudo_labels)
model.add_observer("after_epoch", update_board)


# Train model
test_history, train_history = model.training(
    manager=manager,
    nb_epochs=N_EPOCHS,
    checkpointdir=None,
    fold_index=0,
    scheduler=None,
    with_validation=False)


# Test model
manager = DataManager.from_numpy(
    test_inputs=data, test_labels=labels, batch_size=BATCH_SIZE)
test_model = DeepLearningInterface(
    model=model.model.network,
    optimizer_name="SGD",
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=10**-5,
    loss_name="CrossEntropyLoss")
y_pred, X, y_true, loss, values = test_model.testing(
    manager=manager,
    with_logit=True,
    # logit_function="sigmoid",
    predict=False)
print(y_pred.shape, X.shape, y_true.shape)


# Inspect results
result = pd.DataFrame.from_dict(collections.OrderedDict([
    ("pred", (np.argmax(y_pred, axis=1)).astype(int)),
    ("truth", y_true.squeeze()),
    ("prob_0", y_pred[:, 0].squeeze()),
    ("prob_1", y_pred[:, 1].squeeze())]))
print(result)
y_pred = np.argmax(y_pred, axis=1)
fig, ax = plt.subplots()
cmap = plt.get_cmap("Blues")
cm = SKMetrics("confusion_matrix", with_logit=False)(y_pred, y_true)
sns.heatmap(cm, cmap=cmap, annot=True, fmt="g", ax=ax)
ax.set_xlabel("predicted values")
ax.set_ylabel("actual values")
print(classification_report(y_true, y_pred >= 0.4))
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
