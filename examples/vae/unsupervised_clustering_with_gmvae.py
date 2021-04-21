"""
Unsupervised clustering with GMVAE
==================================

Credit: A Grigis

Unsupervised Gaussian Mixture Variational Autoencoder (GMVAE) on a synthetic
dataset.

GMVAE is an attempt to replicate the work described in this
[blog](http://ruishu.io/2016/12/25/gmvae/) and inspired from this
[paper](https://arxiv.org/abs/1611.02648).

Let's begin with importing stuffs:
"""

# Imports
import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pynet
from pynet import NetParameters
from pynet.datasets import DataManager
from pynet.interfaces import GMVAENetClassifier
from pynet.utils import setup_logging


#############################################################################
# Parameters
# ----------
#
# Define some global parameters that will be used to create and train the
# model:

n_samples = 100
n_classes = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5
snr = 10
batch_size = 10
adam_lr = 2e-3
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = pynet.get_tools(tool_name="losses")
metrics = pynet.get_tools(tool_name="metrics")
setup_logging(level="info")


#############################################################################
# Synthetic dataset
# -----------------
#
# A Gaussian Linear Multi-Klass synthetic dataset is generated as
# follows. The number of the latent dimensions used to generate the data can be
# controlled.

class GeneratorUniform(nn.Module):
    """ Generate multiple sources (channels) of data through a linear
    generative model:

    z ~ N(mu,sigma)
    for c_idx in n_channels:
        x_ch = W_ch(c_idx)
    where 'W_ch' is an arbitrary linear mapping z -> x_ch
    """
    def __init__(self, lat_dim=2, n_channels=2, n_feats=5, seed=100):
        super(GeneratorUniform, self).__init__()
        self.lat_dim = lat_dim
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.seed = seed
        np.random.seed(self.seed)
        W = []
        for c_idx in range(n_channels):
            w_ = np.random.uniform(-1, 1, (self.n_feats, lat_dim))
            u, s, vt = np.linalg.svd(w_, full_matrices=False)
            w = (u if self.n_feats >= lat_dim else vt)
            W.append(torch.nn.Linear(lat_dim, self.n_feats, bias=False))
            W[c_idx].weight.data = torch.FloatTensor(w)
        self.W = torch.nn.ModuleList(W)

    def forward(self, z):
        if isinstance(z, list):
            return [self.forward(_) for _ in z]
        if type(z) == np.ndarray:
            z = torch.FloatTensor(z)
        assert z.size(dim=1) == self.lat_dim
        obs = []
        for c_idx in range(self.n_channels):
            x = self.W[c_idx](z)
            obs.append(x.detach())
        return obs


class SyntheticDataset(object):
    def __init__(self, n_samples=500, lat_dim=2, n_feats=5, n_classes=2,
                 generatorclass=GeneratorUniform, snr=1, train=True):
        super(SyntheticDataset, self).__init__()
        self.n_samples = n_samples
        self.lat_dim = lat_dim
        self.n_feats = n_feats
        self.n_classes = n_classes
        self.snr = snr
        self.train = train
        self.labels = []
        self.z = []
        self.x = []
        seed = 7 if self.train else 14
        np.random.seed(seed)
        locs = np.random.uniform(-5, 5, (self.n_classes, ))
        np.random.seed(seed)
        scales = np.random.uniform(0, 2, (self.n_classes, ))
        np.random.seed(seed)
        for k_idx in range(self.n_classes):
            self.z.append(
                np.random.normal(loc=locs[k_idx], scale=scales[k_idx],
                                 size=(self.n_samples, self.lat_dim)))
            self.generator = generatorclass(
                lat_dim=self.lat_dim, n_channels=1, n_feats=self.n_feats)
            self.x.append(self.generator(self.z[-1])[0])
            self.labels += [k_idx] * self.n_samples
        self.data = np.concatenate(self.x, axis=0).astype(np.float32)
        self.labels = np.asarray(self.labels)
        _, self.data = preprocess_and_add_noise(self.data, snr=snr)


def preprocess_and_add_noise(x, snr, seed=0):
    scalers = StandardScaler().fit(x)
    x_std = scalers.transform(x)
    np.random.seed(seed)
    sigma_noise = np.sqrt(1. / snr)
    x_std_noisy = x_std + sigma_noise * np.random.randn(*x_std.shape)
    return x_std, x_std_noisy


# Create dataset
ds_train = SyntheticDataset(
    n_samples=n_samples,
    lat_dim=true_lat_dims,
    n_feats=n_feats,
    n_classes=n_classes,
    train=True,
    snr=snr)
ds_val = SyntheticDataset(
    n_samples=n_samples,
    lat_dim=true_lat_dims,
    n_feats=n_feats,
    n_classes=n_classes,
    train=False,
    snr=snr)
image_datasets = {
    "train": ds_train,
    "val": ds_val}
manager = DataManager.from_numpy(
    train_inputs=ds_train.data, train_outputs=None, train_labels=ds_train.labels,
    validation_inputs=ds_val.data, validation_outputs=None,
    validation_labels=ds_val.labels, batch_size=batch_size, sampler="random",
    add_input=True)
print("- datasets:", image_datasets)
print("- shapes:", ds_train.data.shape, ds_val.data.shape)


# Display generated data
method = manifold.TSNE(n_components=2, init="pca", random_state=0)
y_train = method.fit_transform(ds_train.data)
y_val = method.fit_transform(ds_val.data)
fig, axs = plt.subplots(nrows=3, ncols=2)
for cnt, (name, y, labels) in enumerate((
        ("train", y_train, ds_train.labels),
        ("val", y_val, ds_val.labels))):
    colors = labels.astype(float)
    colors /= colors.max()
    axs[0, cnt].scatter(y[:, 0], y[:, 1], c=colors, cmap=plt.cm.Spectral)
    axs[0, cnt].xaxis.set_major_formatter(NullFormatter())
    axs[0, cnt].yaxis.set_major_formatter(NullFormatter())
    axs[0, cnt].set_title("GT clustering ({0})".format(name))
    axs[0, cnt].axis("tight")



#############################################################################
# ML clustering
# -------------
#
# As a ground truth we performed a K-means clustering of the data.

kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(ds_train.data)
train_labels = kmeans.labels_
train_acc = losses["GMVAELoss"].cluster_acc(train_labels, ds_train.labels)
print("-- K-Means ACC train", train_acc)
val_labels = kmeans.predict(ds_val.data)
val_acc = losses["GMVAELoss"].cluster_acc(val_labels, ds_val.labels)
print("-- K-Means ACC val",val_acc)

for cnt, (name, y, labels, acc) in enumerate((
        ("train", y_train, train_labels, train_acc),
        ("val", y_val, val_labels, val_acc))):
    colors = labels.astype(float)
    colors /= colors.max()
    axs[1, cnt].scatter(y[:, 0], y[:, 1], c=colors, cmap=plt.cm.Spectral)
    axs[1, cnt].xaxis.set_major_formatter(NullFormatter())
    axs[1, cnt].yaxis.set_major_formatter(NullFormatter())
    axs[1, cnt].set_title(
        "K-means clustering ({0}-ACC:{1:.3f})".format(name, acc))
    axs[1, cnt].axis("tight")


#############################################################################
# Training
# --------
#
# We'll create and train the model to optimize the losses using Adam
# optimizer.

torch.manual_seed(42)
params = NetParameters(
    input_dim=n_feats,
    latent_dim=fit_lat_dims,
    n_mix_components=n_classes,
    sigma_min=0.001,
    raw_sigma_bias=0.25,
    dropout=0,
    temperature=1,
    gen_bias_init=0.)
model = GMVAENetClassifier(
    params,
    optimizer_name="Adam",
    learning_rate=adam_lr,
    loss=losses["GMVAELoss"](),
    use_cuda=(device.type != "cpu"))
print("- model:", model)

print("- training...")
train_history, valid_history = model.training(
    manager=manager,
    nb_epochs=epochs,
    checkpointdir=None,
    fold_index=0,
    with_validation=True)


#############################################################################
# Results
# -------
#
# Lets now display the clustering results.

net = model.model
net.eval()
with torch.no_grad():
    p_x_given_z, dists = net(
        torch.from_numpy(ds_train.data.astype(np.float32)).to(device))
q_y_given_x = dists["q_y_given_x"]
train_labels = np.argmax(q_y_given_x.logits.detach().cpu().numpy(), axis=1)
train_acc = losses["GMVAELoss"].cluster_acc(
    q_y_given_x.logits, ds_train.labels, is_logits=True)
print("-- GMVAE ACC train", train_acc)
with torch.no_grad():
    p_x_given_z, dists = net(
            torch.from_numpy(ds_val.data.astype(np.float32)).to(device))
q_y_given_x = dists["q_y_given_x"]
val_labels = np.argmax(q_y_given_x.logits.detach().cpu().numpy(), axis=1)
val_acc = losses["GMVAELoss"].cluster_acc(
    q_y_given_x.logits, ds_val.labels, is_logits=True)
print("-- GMVAE ACC val", val_acc)

for cnt, (name, y, labels, acc) in enumerate((
        ("train", y_train, train_labels, train_acc),
        ("val", y_val, val_labels, val_acc))):
    colors = labels.astype(float)
    colors /= colors.max()
    axs[2, cnt].scatter(y[:, 0], y[:, 1], c=colors, cmap=plt.cm.Spectral)
    axs[2, cnt].xaxis.set_major_formatter(NullFormatter())
    axs[2, cnt].yaxis.set_major_formatter(NullFormatter())
    axs[2, cnt].set_title(
        "GMVAE clustering ({0}-ACC:{1:.3f})".format(name, acc))
    axs[2, cnt].axis("tight")

plt.show()
