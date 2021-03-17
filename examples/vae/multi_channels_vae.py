"""
Multi Channels VAE (MCVAE)
==========================

Credit: A Grigis & C. Ambroise
"""

# Imports
import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pynet
from pynet import NetParameters
from pynet.datasets import DataManager
from pynet.datasets.core import DataItem
from pynet.interfaces import MCVAEEncoder
from pynet.utils import setup_logging


# Global parameters
n_samples = 500
n_channels = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5
snr = 10
adam_lr = 2e-3
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = pynet.get_tools(tool_name="losses")
setup_logging(level="info")


# Create synthetic data


class GeneratorUniform(nn.Module):
    """ Generate multiple sources (channels) of data through a linear
    generative model:
    z ~ N(0,I)
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
        assert z.size(1) == self.lat_dim
        obs = []
        for ch in range(self.n_channels):
            x = self.W[ch](z)
            obs.append(x.detach())
        return obs


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=500, lat_dim=2, n_feats=5, n_channels=2,
                 generatorclass=GeneratorUniform, snr=1, train=True):
        super(SyntheticDataset, self).__init__()
        self.n_samples = n_samples
        self.lat_dim = lat_dim
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.snr = snr
        self.train = train
        seed = (7 if self.train is True else 14)
        np.random.seed(seed)
        self.z = np.random.normal(size=(self.n_samples, self.lat_dim))
        self.generator = generatorclass(
            lat_dim=self.lat_dim, n_channels=self.n_channels,
            n_feats=self.n_feats)
        self.x = self.generator(self.z)
        self.X, self.X_noisy = preprocess_and_add_noise(self.x, snr=snr)
        self.X = [np.expand_dims(x.astype(np.float32), axis=1) for x in self.X]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return DataItem(inputs=[x[item] for x in self.X], outputs=None,
                        labels=None)

    @property
    def shape(self):
        return (len(self), len(self.X))

def preprocess_and_add_noise(x, snr, seed=0):
    if not isinstance(snr, list):
        snr = [snr] * len(x)
    scalers = [StandardScaler().fit(c_arr) for c_arr in x]
    x_std = [scalers[c_idx].transform(x[c_idx]) for c_idx in range(len(x))]
    # seed for reproducibility in training/testing based on prime number basis
    seed = (seed + 3 * int(snr[0] + 1) + 5 * len(x) + 7 * x[0].shape[0] +
            11 * x[0].shape[1])
    np.random.seed(seed)
    x_std_noisy = []
    for c_idx, arr in enumerate(x_std):
        sigma_noise = np.sqrt(1. / snr[c_idx])
        x_std_noisy.append(arr + sigma_noise * np.random.randn(*arr.shape))
    return x_std, x_std_noisy


# Create dataset
ds_train = SyntheticDataset(
    n_samples=n_samples,
    lat_dim=true_lat_dims,
    n_feats=n_feats,
    n_channels=n_channels,
    train=True,
    snr=snr)
ds_val = SyntheticDataset(
    n_samples=n_samples,
    lat_dim=true_lat_dims,
    n_feats=n_feats,
    n_channels=n_channels,
    train=False,
    snr=snr)
image_datasets = {
    "train": ds_train,
    "val": ds_val}
manager = DataManager.from_dataset(
    train_dataset=image_datasets["train"],
    validation_dataset=image_datasets["val"],
    batch_size=n_samples, sampler="random", multi_bloc=True)
print("- datasets:", image_datasets)


# Create models
models = {}
torch.manual_seed(42)
params = NetParameters(
    latent_dim=fit_lat_dims,
    n_channels=n_channels,
    n_feats=[n_feats] * n_channels,
    vae_model="dense",
    vae_kwargs={},
    sparse=False)
models["mcvae"] = MCVAEEncoder(params,
    optimizer_name="Adam",
    learning_rate=adam_lr,
    loss=losses["MCVAELoss"](n_channels, beta=1., sparse=False),
    use_cuda=False)
torch.manual_seed(42)
params = NetParameters(
    latent_dim=fit_lat_dims,
    n_channels=n_channels,
    n_feats=[n_feats] * n_channels,
    vae_model="dense",
    vae_kwargs={},
    sparse=True)
models["smcvae"] = MCVAEEncoder(params,
    optimizer_name="Adam",
    learning_rate=adam_lr,
    loss=losses["MCVAELoss"](n_channels, beta=1., sparse=True),
    use_cuda=False)
print("- models:", models)


# Fit models
for model_name, interface in models.items():
    print("- training:", model_name)
    train_history, valid_history = interface.training(
        manager=manager,
        nb_epochs=epochs,
        checkpointdir=None,
        fold_index=0,
        with_validation=True)


# Display results
pred = {}  # Prediction
z = {}     # Latent Space
g = {}     # Generative Parameters
x_hat = {}  # Reconstructed channels
loaders = manager.get_dataloader(validation=True, fold_index=0)
dataitem = next(iter(loaders.validation))

for model_name, interface in models.items():
    model = interface.model
    model.eval()
    X = [x.to(interface.device) for x in dataitem.inputs]
    print("--", model_name)
    print("-- X", [x.size() for x in X])

    with torch.no_grad():
        q = model.encode(X)  # encoded distribution q(z|x)
    print("-- encoded distribution q(z|x)", [n for n in q])

    z[model_name] = model.p_to_prediction(q)
    print("-- z", [e.shape for e in z[model_name]])

    if model.sparse:
        z[model_name] = model.apply_threshold(z[model_name], 0.2)
    z[model_name] = np.array(z[model_name]).reshape(-1) # flatten
    print("-- z", z[model_name].shape)

    g[model_name] = [
        model.vae[c_idx].encode.w_mu.weight.detach().numpy()
        for c_idx in range(n_channels)]
    g[model_name] = np.array(g[model_name]).reshape(-1)  #flatten


# With such a simple dataset, mcvae and sparse-mcvae gives the same results in
# terms of latent space and generative parameters.
# However, only with the sparse model is possible to easily identify the
# important latent dimensions.

plt.figure()
plt.subplot(1,2,1)
plt.hist([z["smcvae"], z["mcvae"]], bins=20, color=["k", "gray"])
plt.legend(["Sparse", "Non sparse"])
plt.title("Latent dimensions distribution")
plt.ylabel("Count")
plt.xlabel("Value")
plt.subplot(1,2,2)
plt.hist([g["smcvae"], g["mcvae"]], bins=20, color=["k", "gray"])
plt.legend(["Sparse", "Non sparse"])
plt.title(r"Generative parameters $\mathbf{\theta} = \{\mathbf{\theta}_1 "
          r"\ldots \mathbf{\theta}_C\}$")
plt.xlabel("Value")

do = np.sort(models["smcvae"].model.dropout.detach().numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.suptitle("Dropout probability of {0} fitted latent dimensions in Sparse "
             "Model".format(fit_lat_dims))
plt.title("{0} true latent dimensions".format(true_lat_dims))

plt.show()
print("See you!")
