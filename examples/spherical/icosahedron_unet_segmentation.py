"""
pynet: icosahedron UNet segmentation
====================================

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
from pynet.datasets import DataManager
from pynet.interfaces import SphericalUNetEncoder
from pynet.utils import setup_logging
from pynet.plotting import Board, update_board
from pynet.models.spherical.sampling import icosahedron
from pynet.plotting.surface import plot_trisurf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# Global Parameters
OUTDIR = "/tmp/ico_unet"
BATCH_SIZE = 5
N_EPOCHS = 5
N_CLASSES = 2
N_SAMPLES = 40
ICO_ORDER = 4
SAMPLES = {
    0: [(0, 1), (4, 2)],
    1: [(2, 2), (2, 1)]}
setup_logging(level="debug")


def gaussian_sdist(vertices, triangles, n_maps, scales):
    """ Generate gaussian distance features maps.
    """
    assert len(scales) == n_maps
    locs = vertices[np.random.randint(0, len(vertices), n_maps)]
    features = []
    for loc, scale in zip(locs, scales):
        dist = np.linalg.norm(vertices - loc, axis=1)
        features.append(norm.pdf(dist, loc=0, scale=scale))
    return np.asarray(features)


# Load the data
ico_vertices, ico_triangles = icosahedron(order=ICO_ORDER)
print(ico_vertices.shape, ico_triangles.shape)
prob = gaussian_sdist(ico_vertices, ico_triangles, n_maps=1, scales=[1])
labels = (prob[0] > 0.25).astype(int)
fig, ax = plt.subplots(1, 1, subplot_kw={
    "projection": "3d", "aspect": "auto"}, figsize=(10,10))
tri_texture = np.asarray(
    [np.round(np.mean(labels[tri])) for tri in ico_triangles])
plot_trisurf(fig, ax, ico_vertices, ico_triangles, tri_texture)
data = np.zeros((N_SAMPLES, N_CLASSES, len(labels)), dtype=float)
for klass in (0, 1):
    k_indices = np.argwhere(labels == 0).squeeze()
    for loc, scale in SAMPLES[klass]:
        data[:, klass, k_indices] = np.random.normal(
            loc=loc, scale=scale, size=len(k_indices))
labels = np.ones((N_SAMPLES, 1)) * labels
print("dataset: x {0} - y {1}".format(data.shape, labels.shape))


# Create data manager
manager = DataManager.from_numpy(
    train_inputs=data, train_labels=labels, test_inputs=data,
    test_labels=labels, batch_size=BATCH_SIZE)


# Create model
net_params = pynet.NetParameters(
    in_order=ICO_ORDER,
    in_channels=2,
    out_channels=N_CLASSES,
    depth=3,
    start_filts=32,
    conv_mode="1ring",
    up_mode="transpose",
    cachedir=os.path.join(OUTDIR, "cache"))
model = SphericalUNetEncoder(
    net_params,
    optimizer_name="SGD",
    learning_rate=0.1,
    momentum=0.99,
    weight_decay=10**-4,
    loss_name="CrossEntropyLoss",
    use_cuda=True)
model.board = Board(port=8097, host="http://localhost", env="spherical_unet")
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
y_pred, X, y_true, loss, values = model.testing(
    manager=manager,
    with_logit=True,
    predict=True)
print(y_pred.shape, X.shape, y_true.shape)


# Inspect results
fig, ax = plt.subplots(1, 1, subplot_kw={
    "projection": "3d", "aspect": "auto"}, figsize=(10,10))
tri_texture = np.asarray(
    [np.round(np.mean(y_pred[:, tri])) for tri in ico_triangles])
plot_trisurf(fig, ax, ico_vertices, ico_triangles, tri_texture)

plt.show()
