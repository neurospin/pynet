"""
Conditional Variational AutoEncoder (VAE)
=========================================

Credit: A Grigis

Based on:

- https://ravirajag.dev

This tutorial is for the intuition of simple Variational Autoencoder(VAE)
implementation in pynet.
After reading this tutorial, you'll understand the technical details needed to
implement conditional VAE.
The main difference from the vanilla VAE is, in the vanilla case we generate
image randomly, and here we can condition for which number we want to generate
the image.

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
    labels="label",
    stratify_label="label",
    number_of_folds=10,
    batch_size=64,
    test_size=0,
    input_transforms=[flatten],
    add_input=True,
    sample_size=1)


#############################################################################
# The Model
# ---------
#
# The model is composed of two sub-networks:
#
# 1. Given x (image), encode it into a distribution over the latent space -
#    referred to as Q(z|x).
# 2. Given z in latent space (code representation of an image), decode it into
#    the image it represents - referred to as f(z).
#
# We want to enforce some of the latent dimensions to encode the digit found
# in an image. In the vanilla VAE case we don't care what
# information each dimension of the latent space holds. The model can learn
# to encode whatever information it finds valuable for its task. Since we're
# familiar with the dataset, we know the digit type should be important.
# We want to help the model by providing it with this information. Moreover,
# we'll use this information to generate images conditioned on the digit type,
# as we will see later.
# 
# Given the digit type, we'll encode it using one hot encoding, that is, a
# vector of size 10. These 10 numbers will be concatenated into the latent 
# vector, so when decoding that vector into an image, the model will make use
# of the digit information.
#
# There are two ways to provide the model with a one hot encoding vector:
#
# 1. Add it as an input to the model.
# 2. Add it as a label so the model will have to predict it by itself: add
#    another sub-network that predicts a vector of size 10 where the loss is
#    the cross entropy with the expected one hot vector.
#
# We'll go with the first option first.

def idx2onehot(idx, n):
    """ Given a class label, we will convert it into one-hot encoding.
    """
    assert idx.ndim == 1
    assert torch.max(idx).item() < n
    idx = idx.view(-1, 1)
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)

    return onehot

class Encoder(nn.Module):
    """ This the encoder part of VAE.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the size of input (in case of MNIST 28 * 28).
        hidden_dim: int
            the size of hidden dimension.
        latent_dim: int
            the latent dimension.
        n_classes: int
            the number of classes (dimension of one-hot representation of   
            labels).
        """
        super().__init__()
        self.linear = nn.Linear(input_dim + n_classes, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden = func.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]: this is log(var)

        return z_mu, z_var

class Decoder(nn.Module):
    """ This the decoder part of VAE
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        """ Init class.

        Parameters
        ----------
        latent_dim: int
            the latent size.
        hidden_dim: int
            the size of hidden dimension.
        output_dim: int
            the output dimension (in case of MNIST it is 28 * 28).
        n_classes: int
            the number of classes (dimension of one-hot representation of   
            labels).
        """
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]
        hidden = func.relu(self.latent_to_hidden(x))
        # hidden is of shape [batch_size, hidden_dim]
        predicted = torch.sigmoid(self.hidden_to_out(hidden))
        # predicted is of shape [batch_size, output_dim]

        return predicted


class CVAE(nn.Module):
    """ This is the conditional VAE.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the size of input (in case of MNIST 28 * 28).
        hidden_dim: int
            the size of hidden dimension.
        latent_dim: int
            the latent dimension.
        n_classes: int
            the number of classes (dimension of one-hot representation of   
            labels).
        """
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)
        self.n_classes = n_classes

    def forward(self, x, y):
        # convert y into one-hot encoding
        y = idx2onehot(y, n=self.n_classes)
        y = y.to(x.device)

        # encode an image into a distribution over the latent space
        z_mu, z_var = self.encoder(torch.cat((x, y), dim=1))

        # sample a latent vector from the latent space - using the
        # reparameterization trick
        # sample from the distribution having latent parameters z_mu, z_var
        # the reason we exponentiate is because we need the variance to be
        # positive. Any activation function whose range is the positive numbers
        # could be used here.
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode the latent vector - concatenated to the digits
        # classification into an image
        predicted = self.decoder(torch.cat((x_sample, y), dim=1))

        return predicted, {"z_mu": z_mu, "z_var": z_var, "y": y}


#############################################################################
# Loss
# ----
#
# VAE consists of two loss functions:
#
# 1. Reconstruction loss: how well we can reconstruct the image
# 2. KL divergence: how off the distribution over the latent space is 
#    from the prior. Given the prior is a standard Gaussian and the inferred
#    distribution is a Gaussian with a diagonal covariance matrix,
#    the KL-divergence becomes analytically solvable.

class DecodeLoss(object):
    def __init__(self):
        super(DecodeLoss, self).__init__()
        self.layer_outputs = None

    def __call__(self, x_sample, x, y):
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
#
# At the end of every epoch we'll sample latent vectors and decode them into
# images, so we can visualize how the generative power of the model improves
# over the epochs. The sampling method is as follows:
#
# 1. Deterministically set the dimensions which are used for digit
#    classification according to the digit we want to generate an image for.
#    If for example we want to generate an image of the digit 2, these
#    dimensions will be set to [0010000000].
# 2. Randomly sample the other dimensions according to the prior - a
#    multivariate Gaussian. We'll use these sampled values for all the
#    different digits we generate in a given epoch. This way we can have a
#    feeling of what is encoded in the other dimensions, for example stroke
#    style.
#
# The intuition behind step 1 is that after convergence the model should be
# able to classify the digit in an input image using these dimensions. On the
# other hand, these dimensions are also used in the decoding step to generate
# the image. It means the decoder sub-network learns that when these
# dimensions have the values corresponding to the digit 2, it should generate
# an image of that digit. Therefore, if we manually set these dimensions to
# contain the information of the digit 2, we'll get a generated image of that
# digit.

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
    z = torch.randn(1, 20).to(device).repeat(10, 1)
    y = torch.eye(10).to(device, dtype=z.dtype)
    z = torch.cat((z, y), dim=1)
    # run only the decoder
    reconstructed_img = model.decoder(z)
    img = reconstructed_img.view(-1, 28, 28).detach().numpy()
    # display result
    img = np.asarray([ndimage.zoom(arr, 5, order=0) for arr in img])
    img = np.expand_dims(img, axis=1)
    img = (img / img.max()) * 255
    board.viewer.images(
        img,
        opts={
            "title": "sampling",
            "caption": "sampling"},
        win="sampling")    

model = CVAE(input_dim=(28 * 28), hidden_dim=128, latent_dim=20, n_classes=10)
interface = DeepLearningInterface(
    model=model,
    optimizer_name="Adam",
    learning_rate=0.001,
    loss=DecodeLoss(),
    add_labels=True)
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
# We choose to add the one hot encoded vector directly as an input of the
# model.
# Well, this might be not the best option especially when we want to perform
# representation learning. Indeed, in test time we want to provide an image as
# input, and infer a latent vector. We can't provide the model
# with the digit as input, since we won't know it in test time. We will
# learn how to overcome this issue in the second CVAE tutorial.
