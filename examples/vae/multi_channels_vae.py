"""
Multi Channels VAE (MCVAE)
==========================

Credit: A Grigis
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
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from pynet.models.vae.mcvae import MCVAE, MCVAELoss
from pynet.utils import setup_logging


# Global parameters
setup_logging(level="info")
n_samples = 500
n_channels = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5
snr = 10
adam_lr = 2e-3
epochs = 1500


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
        self.X = [x.astype(np.float32) for x in self.X]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return [x[item] for x in self.X]


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
    # "val": ds_val
}
print("- datasets:", image_datasets)


# Create models
models = {}
torch.manual_seed(42)
vae_kwargs = {}
    # "hidden_dims": [10]}
models["mcvae"] = MCVAE(
    latent_dim=fit_lat_dims, n_channels=n_channels,
    n_feats=[n_feats] * n_channels, vae_model="dense", vae_kwargs=vae_kwargs)
torch.manual_seed(42)
models["smcvae"] = MCVAE(
    latent_dim=fit_lat_dims, n_channels=n_channels,
    n_feats=[n_feats] * n_channels, vae_model="dense", vae_kwargs=vae_kwargs,
    sparse=True)
print("- models:", models)


# Fit models

def train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs=25):
    # Parameters
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    X = {}
    for phase in image_datasets.keys():
        big_X = [ [] for _ in range(n_channels) ]
        X_temp = dataloaders[phase]
        for x in X_temp:
            for idx, c in enumerate(x):
                big_X[idx].append(c)
        X[phase] = [torch.cat(x) for x in big_X]   

    # Loop over epochs
    t_epoch = time.time()
    loss_computing_time = 0
    forwarding_time = 0
    backwarding_time = 0
    optimizer_time = 0
    convert_inputs_time = 0
    epoch_loss_time = 0
    epoch_time = 0
    optimization_time = 0
    iteration_time = 0
    set_gradients_time = 0
    time_to_iterate = 0
    for epoch in range(num_epochs):
        t_epoch = time.time()
        # Each epoch has a training and validation phase
        for phase in image_datasets.keys():
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            t_iterate = time.time()
            t_to_iterate = time.time()

            # for inputs in dataloaders[phase]:
            time_to_iterate += time.time() - t_to_iterate
            t = time.time()
            # inputs = [ch_inputs.to(device) for ch_inputs in inputs]
            inputs = [x.to(device) for x in X[phase]]
            convert_inputs_time += time.time() - t

            # zero the parameter gradients
            t = time.time()
            optimizer.zero_grad()
            set_gradients_time += time.time() - t
            # forward
            # track history if only in train
            t_opt = time.time()
            with torch.set_grad_enabled(phase == "train"):
                t = time.time()
                outputs, kwargs = model(inputs)
                forwarding_time += time.time() - t
                t = time.time()
                loss = criterion(inputs, outputs, **kwargs)
                loss_computing_time += time.time() - t

                # backward + optimize only if in training phase
                if phase == "train":
                    t = time.time()
                    loss.backward()
                    backwarding_time += time.time() - t
                    t = time.time()
                    optimizer.step()
                    optimizer_time += time.time() - t

            # statistics
            running_loss += loss.item() * inputs[0].size(0)
            # running_corrects += torch.sum(preds == labels.data)
            optimization_time += time.time() - t_opt
            t_to_iterate = time.time()

            iteration_time += time.time() - t_iterate
            # Update scheduler
            t = time.time()
            if phase == "train":
                scheduler.step()

            # Epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_loss_time = time.time() - t
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_time += time.time() - t_epoch
            if epoch % 10 == 0:
                print("===> {} : epoch {}/{}, Loss: {:.4f}, times : compute loss: {:.4f}, forward: {:.4f}, backward: {:.4f}, optimizer: {:.4f}, inputs: {:.4f}, epoch_loss: {:.4f}, optimization: {:.4f}, iteration: {:.4f}, gadrients: {:.4f}, time_to_iterate: {:.4f}, epochs: {:.4f}".format(
                    phase, epoch, num_epochs - 1, epoch_loss, loss_computing_time, forwarding_time, backwarding_time, optimizer_time, convert_inputs_time, epoch_loss_time, optimization_time, iteration_time, set_gradients_time, time_to_iterate, epoch_time))#, epoch_acc))
                epoch_time  = 0
                loss_computing_time = 0
                forwarding_time = 0
                backwarding_time = 0
                optimizer_time = 0
                convert_inputs_time = 0
                epoch_loss_time = 0
                optimization_time = 0
                iteration_time = 0
                set_gradients_time = 0
                time_to_iterate = 0

            # Save weights of the best model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    # print("Best val Acc: {:4f}".format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model


dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=100, shuffle=True, num_workers=4)
              for x in image_datasets.keys()}

for model_name, model in models.items():
    
    print("- training:", model_name)
    criterion = MCVAELoss(model.n_channels, beta=1., sparse=model.sparse)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    print(model)
    train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs=epochs)


# Display results
pred = {}  # Prediction
z = {}     # Latent Space
g = {}     # Generative Parameters
x_hat = {}  # reconstructed channels

print("Latent space ...")
for model_name, model in models.items():
    X = dataloaders['train']
    # print(type(X))
    # print(len(X))
    big_X = [[],[],[]]
    for x in X:
        for idx, c in enumerate(x):
            big_X[idx].append(c)
    X = [torch.cat(x) for x in big_X]    
    
        
    # X = [x.to(device) for x in X]
    print("--", model_name)
    print("-- X", [e.size() for e in X])
    m = model_name
    # plot_loss(model)
    q = model.encode(X)  # encoded distribution q(z|x)
    print("-- encoded distribution q(z|x)", [n for n in q])
    z[m] = [q[i].loc.squeeze().detach().numpy() for i in range(n_channels)]
    print("-- z", [e.shape for e in z[m]])
    # if model.sparse:
    #     z[m] = model.apply_threshold(z[m], 0.2)
    z[m] = np.array(z[m]).reshape(-1)  # flatten
    print("-- z", z[m].shape)
    # x_hat[m] = model.reconstruct(X, dropout_threshold=0.2)  # it will raise a warning in non-sparse mcvae
    g[m] = [model.vae[i].fc_mu.weight.detach().numpy() for i in range(n_channels)]
    g[m] = np.array(g[m]).reshape(-1)  #flatten


# lsplom(ltonumpy(x), title=f'Ground truth')
# lsplom(ltonumpy(x_noisy), title=f'ENoisy data fitted by the models (snr={snr})')
# for m in models.keys():
#     lsplom(ltonumpy(x_hat[m]), title=f'Reconstructed with {m} model')

"""
With such a simple dataset, mcvae and sparse-mcvae gives the same results in terms of
latent space and generative parameters.
However, only with the sparse model is possible to easily identify the important latent dimensions.
"""
plt.figure()
plt.subplot(1,2,1)
plt.hist([z['smcvae'], z['mcvae']], bins=20, color=['k', 'gray'])
plt.legend(['Sarse', 'Non sparse'])
plt.title(r'Latent dimensions distribution')
plt.ylabel('Count')
plt.xlabel('Value')
plt.subplot(1,2,2)
plt.hist([g['smcvae'], g['mcvae']], bins=20, color=['k', 'gray'])
plt.legend(['Sparse', 'Non sparse'])
plt.title(r'Generative parameters $\mathbf{\theta} = \{\mathbf{\theta}_1 \ldots \mathbf{\theta}_C\}$')
plt.xlabel('Value')

print(models['smcvae'].dropout)
do = np.sort(models['smcvae'].dropout.detach().numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.suptitle(f'Dropout probability of {fit_lat_dims} fitted latent dimensions in Sparse Model')
plt.title(f'({true_lat_dims} true latent dimensions)')

plt.show()
print("See you!")
