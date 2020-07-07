"""
Vanilla Variational AutoEncoder (VAE)
=====================================

Credit: A Grigis

Based on:

- https://ravirajag.dev

This tutorial is for the intuition of simple Variational Autoencoder (VAE)
implementation in pynet.
After reading this tutorial, you'll understand the technical details needed to
implement VAE.

Let’s begin with importing stuffs:
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as func
from pynet.datasets import DataManager, fetch_minst
from pynet.interfaces import DeepLearningInterface
from pynet.plotting import Board, update_board


#############################################################################
# The model will be trained on MNIST - handwritten digits dataset. The input
# is an image in R(28×28).

def flatten(arr):
    return arr.flatten()

data = fetch_minst(datasetdir="/neurospin/nsap/datasets/minst")
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    stratify_label="label",
    number_of_folds=10,
    batch_size=64,
    test_size=0,
    input_transforms=[flatten],
    add_input=True,
    sample_size=0.05)


#############################################################################
# The Model
# ---------
#
# The model is composed of two sub-networks:
# 1. Given x (image), encode it into a distribution over the latent space -
#    referred to as Q(z|x).
# 2. Given z in latent space (code representation of an image), decode it into
#    the image it represents - referred to as f(z).

class Encoder(nn.Module):
    """ This the encoder part of VAE.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the size of input (in case of MNIST 28 * 28).
        hidden_dim: int
            the size of hidden dimension.
        latent_dim: int
            the latent dimension.
        """
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden = func.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var

class Decoder(nn.Module):
    """ This the decoder part of VAE.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """ Init class.

        Parameters
        ----------
        latent_dim: int
            the latent size.
        hidden_dim: int
            the size of hidden dimension.
        output_dim: int
            the output dimension (in case of MNIST it is 28 * 28).
        """
        super(Decoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]
        hidden = func.relu(self.latent_to_hidden(x))
        # hidden is of shape [batch_size, hidden_dim]
        predicted = torch.sigmoid(self.hidden_to_out(hidden))
        # predicted is of shape [batch_size, output_dim]

        return predicted

class VAE(nn.Module):
    """ This the VAE, which takes an encoder and a decoder.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the size of input (in case of MNIST 28 * 28).
        hidden_dim: int
            the size of hidden dimension.
        latent_dim: int
            the latent dimension.
        """
        super(VAE, self).__init__()
        self.encorder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decorder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        # encode an image into a distribution over the latent space
        z_mu, z_var = self.encorder(x)
        # sample a latent vector from the latent space - using the
        # reparameterization trick
        # sample from the distribution having latent parameters z_mu, z_var
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        # decode the latent vector 
        predicted = self.decorder(x_sample)

        return predicted, {"z_mu": z_mu, "z_var": z_var}


#############################################################################
# Loss
# ----
#
# VAE consists of two loss functions:
# 1. Reconstruction loss: how well we can reconstruct the image
# 2. KL divergence loss: how off the distribution over the latent space is 
#    from the prior. Given the prior is a standard Gaussian and the inferred
#    distribution is a Gaussian with a diagonal covariance matrix,
#    the KL-divergence becomes analytically solvable.

class DecodeLoss(object):
    def __init__(self):
        super(DecodeLoss, self).__init__()
        self.layer_outputs = None

    def __call__(self, x_sample, x):
        if self.layer_outputs is None:
            raise ValueError("The model needs to return the latent space "
                             "distribution parameters z_mu, z_var.")
        z_mu = self.layer_outputs["z_mu"]
        z_var = self.layer_outputs["z_var"]
        # reconstruction loss
        recon_loss = func.binary_cross_entropy(x_sample, x, reduction="sum")
        # KL divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

        return recon_loss + kl_loss


#############################################################################
# Training
# --------
#
# We'll train the model to optimize the losses using Adam optimizer.

def prepare_pred(y_pred):
    y_pred = y_pred[:3]
    y_pred = y_pred.reshape(-1, 28, 28)
    y_pred = np.asarray([ndimage.zoom(arr, 5, order=0) for arr in y_pred])
    y_pred = np.expand_dims(y_pred, axis=1)
    y_pred = (y_pred / y_pred.max()) * 255
    return y_pred

def sampling(signal):
    """ Sample from the distribution and generate a image.
    """
    device = signal.object.device
    model = signal.object.model
    board = signal.object.board
    # sample and generate a image
    z = torch.randn(1, 20).to(device)
    # run only the decoder
    reconstructed_img = model.decorder(z)
    img = reconstructed_img.view(28, 28).detach().numpy()
    # display result
    img = ndimage.zoom(img, 5, order=0)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = (img / img.max()) * 255
    board.viewer.images(
        img,
        opts={
            "title": "sampling",
            "caption": "sampling"},
        win="sampling")    

model = VAE(input_dim=(28 * 28), hidden_dim=128, latent_dim=20)
interface = DeepLearningInterface(
    model=model,
    optimizer_name="Adam",
    learning_rate=0.001,
    loss=DecodeLoss())
interface.board = Board(
    port=8097, host="http://localhost", env="vae", display_pred=True,
    prepare_pred=prepare_pred)
interface.add_observer("after_epoch", update_board)
interface.add_observer("after_epoch", sampling)
test_history, train_history = interface.training(
    manager=manager,
    nb_epochs=10,
    checkpointdir=None,
    fold_index=0,
    with_validation=True)


#############################################################################
# Conclusion
# ----------
#
# Using a simple feed forward network (no fancy convolutions) we're able to
# generate nice looking images after 10 epochs. We generate
# image randomly. We will see in a next tutorial how to add a condition on
# the number we want to generate the image.

