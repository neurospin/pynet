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
from scipy.stats import norm
import matplotlib.pyplot as plt


# Global Parameters
OUTDIR = "/tmp/ico_unet"
BATCH_SIZE = 5
N_EPOCHS = 20
N_CLASSES = 2
N_SAMPLES = 40
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
ico6_vertices, ico6_triangles = icosahedron(order=6)
print(ico6_vertices.shape, ico6_triangles.shape)
prob = gaussian_sdist(ico6_vertices, ico6_triangles, n_maps=1, scales=[1])
labels = (prob[0] > 0.25).astype(int)
fig, ax = plt.subplots(1, 1, subplot_kw={
    "projection": "3d", "aspect": "equal"}, figsize=(10,10))
tri_texture = np.asarray(
    [np.round(np.mean(labels[tri])) for tri in ico6_triangles])
plot_trisurf(fig, ax, ico6_vertices, ico6_triangles, tri_texture)
#plt.show()
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
    train_inputs=data, train_labels=labels, batch_size=BATCH_SIZE)


def my_loss(x, y):
    criterion = nn.CrossEntropyLoss()
    print("  x: {0} - {1}".format(x.shape, x.dtype))
    print(torch.argmax(x, dim=1))
    print("  y: {0} - {1}".format(y.shape, y.dtype))
    print(y)
    return criterion(x, y)


# Create model
net_params = pynet.NetParameters(
    in_order=6,
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
    # loss=my_loss)
    loss_name="CrossEntropyLoss")
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


stop

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
