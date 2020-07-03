"""
Unsupervised Conditional Variational AutoEncoder (VAE)
======================================================

Credit: A Grigis

Based on:

- https://ravirajag.dev

The Variational Autoencoder (VAE) is a  neural networks that try to learn the
shape of the input space. Once trained, the model can be used to generate
new samples from the input space.

If we have labels for our input data, it’s also possible to condition the
generation process on the label. The idea here is to achieve the same results
using an unsupervised approach.

Mixture of Experts
------------------

MoE is a supervised learning framework. MoE relies on the possibility that the
input might be segmented according to the x->y mapping. How can we train a
model that learns the split points while at the same time learns the mapping
that defines the split points.

MoE does so using an architecture of multiple subnetworks - one manager and
multiple experts. The manager maps the input into a soft decision over the
experts, which is used in two contexts:
1. The output of the network is a weighted average of the experts' outputs,
   where the weights are the manager's output.
2. The loss function is $\sum_i p_i(y - \bar{y_i})^2$. y is the label,
   $\bar{y_i}$ is the output of the i'th expert, $p_i$ is the i'th entry of
   the manager's output. When you differentiate the loss, you get these
   results: a) the manager decides for each expert how much it contributes to
   the loss. In other words, the manager chooses which experts should tune
   their weights according to their error, and b) the manager tunes the
   probabilities it outputs in such a way that the experts that got it right
   will get higher probabilities than those that didn’t. This loss function
   encourages the experts to specialize in different kinds of inputs.

MoE is a framework for supervised learning. Surely we can change y to be x for
the unsupervised case, right? MoE's power stems from the fact that each expert
specializes in a different segment of the input space with a unique mapping
x ->y. If we use the mapping x->x, each expert will specialize in a different
segment of the input space with unique patterns in the input itself.

We'll use VAEs as the experts. Part of the VAE’s loss is the reconstruction
loss, where the VAE tries to reconstruct the original input image x.

A cool byproduct of this architecture is that the manager can classify the
digit found in an image using its output vector!

One thing we need to be careful about when training this model is that the
manager could easily degenerate into outputting a constant vector -
regardless of the input in hand. This results in one VAE specialized in all
digits, and nine VAEs specialized in nothing. One way to mitigate it, which
is described in the MoE paper, is to add a balancing term to the loss.
It encourages the outputs of the manager over a batch of inputs to
be balanced: $\sum_\text{examples in batch} \vec{p} \approx Uniform$.

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
        super().__init__()
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
    """ This the decoder part of VAE
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
        super().__init__()
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
    """ This is the VAE.
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
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        # encode an image into a distribution over the latent space
        z_mu, z_var = self.encoder(x)

        # sample a latent vector from the latent space - using the
        # reparameterization trick
        # sample from the distribution having latent parameters z_mu, z_var
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode the latent vector - concatenated to the digits
        # classification into an image
        predicted = self.decoder(x_sample)

        return predicted, {"z_mu": z_mu, "z_var": z_var}

class VAELoss(object):
    def __init__(self):
        super(VAELoss, self).__init__()
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

class Manager(nn.Module):
    def __init__(self, input_dim, hidden_dim, experts):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the size of input (in case of MNIST 28 * 28).
        hidden_dim: int
            the size of hidden dimension.
        experts: list of VAE
            the manager experts.
        """
        super(Manager, self).__init__()
        self._experts = experts
        self._experts_results = []
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, len(experts))
        
    def forward(self, x):
        hidden = func.relu(self.linear1(x))
        logits = self.linear2(hidden)
        probs = func.softmax(logits, dim=1)
        self._experts_results = []
        for net in self._experts:
            self._experts_results.append(net(x))
        return probs, {"experts_results": self._experts_results}

class ManagerLoss(object):
    def __init__(self, balancing_weight=0.1):
        """ Init class.

        Parameters
        ----------
        balancing_weight: float, default 0.1    
            how much the balancing term will contribute to the loss.
        """
        super(ManagerLoss, self).__init__()
        self.layer_outputs = None
        self.balancing_weight = balancing_weight

    def __call__(self, probs, x):
        if self.layer_outputs is None:
            raise ValueError("The model needs to return the latent space "
                             "distribution parameters z_mu, z_var.")
        losses = []
        criterion = VAELoss()
        for result in self.layer_outputs["experts_results"]:
            criterion.layer_outputs = result[1]
            losses.append(criterion(result[0], x).view(-1, 1))
        losses = torch.cat(losses, dim=1)
        expected_expert_loss = torch.sum(losses * probs, dim=1).mean()
        experts_importance = torch.sum(probs, dim=0)
        balancing_loss = experts_importance.std(dim=0)
        loss = expected_expert_loss + self.balancing_weight * balancing_loss

        return loss


#############################################################################
# Training
# --------
#
# We'll train the model to optimize the losses using Adam optimizer.

def sampling(signal):
    """ Sample from the distribution and generate a image.
    """
    device = signal.object.device
    experts = signal.object.model._experts
    board = signal.object.board
    # sample and generate a image
    z = torch.randn(1, 20).to(device)
    # run only the decoder
    images = []
    for model in experts:
        reconstructed_img = model.decoder(z)
        img = reconstructed_img.view(-1, 28, 28).detach().numpy()
        img = np.asarray([ndimage.zoom(arr, 5, order=0) for arr in img])        
        images.append(img)
    # display result
    images = np.asarray(images)
    images = (images / images.max()) * 255
    board.viewer.images(
        images,
        opts={
            "title": "sampling",
            "caption": "sampling"},
        win="sampling")    

experts = [
    VAE(input_dim=(28 * 28), hidden_dim=128, latent_dim=20) for idx in range(10)]
model = Manager(input_dim=(28 * 28), hidden_dim=128, experts=experts)
interface = DeepLearningInterface(
    model=model,
    optimizer_name="Adam",
    learning_rate=0.001,
    loss=ManagerLoss(balancing_weight=0.1))
interface.board = Board(
    port=8097, host="http://localhost", env="vae")
interface.add_observer("after_epoch", update_board)
interface.add_observer("after_epoch", sampling)
test_history, train_history = interface.training(
    manager=manager,
    nb_epochs=10,
    checkpointdir=None,
    fold_index=0,
    with_validation=True)



