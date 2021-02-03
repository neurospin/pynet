# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
The Variational U-Net auto-encoder.
"""


# Imports
import logging
import collections
import torch
import torch.nn as nn
import torch.nn.functional as func
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import numpy as np
from .base import BaseVAE
from ..unet import Down, Up, Conv1x1x1


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae"))
class VUNet(BaseVAE):
    """ VUNet.

    The Variational U-Net is a convolutional encoder-decoder neural network.
    The convolutional encoding/decoding parts are the same as the UNet.

    The model is composed of two sub-networks:

    1. Given x (image), encode it into a distribution over the latent space -
       referred to as Q(z|x).
    2. Given z in latent space (code representation of an image), decode it
       into the image it represents - referred to as f(z).
    """

    def __init__(self, latent_dim, in_channels=1, depth=5,
                 start_filts=64, up_mode="transpose",
                 batchnorm=True, dim="3d", input_shape=None,
                 num_classes=None):
        """ Init class.

        Parameters
        ----------
        latent_dim: int
            the latent dimension.
        in_channels: int, default 1
            number of channels in the input tensor.
        depth: int, default 5
            number of layers in the U-Net.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        up_mode: string, default 'transpose'
            type of upconvolution. Choices: 'transpose' for transpose
            convolution, 'upsample' for nearest neighbour upsampling.
        batchnorm: bool, default False
            normalize the inputs of the activation function.
        dim: str, default '3d'
            '3d' or '2d' input data.
        input_shape: uplet
            the tensor data shape (X, Y, Z) used during upsample (by default
            use a scale factor of 2).
        num_classes: int, default None
            the number of classes for the conditioning.
        """
        # Inheritance
        nn.Module.__init__(self)

        # Check inputs
        if dim in ("2d", "3d"):
            self.dim = dim
        else:
            raise ValueError(
                "'{}' is not a valid mode for merging up and down paths. Only "
                "'3d' and '2d' are allowed.".format(dim))
        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                "'{}' is not a valid mode for upsampling. Only 'transpose' "
                "and 'upsample' are allowed.".format(up_mode))

        # Declare class parameters
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down = []
        self.up = []
        self.shapes = None
        if input_shape is not None:
            self.shapes = self.downsample_shape(
                input_shape, nb_iterations=(depth - 1))
            self.shapes = self.shapes[::-1]

        # Create the encoder pathway
        self.hidden_dims = []
        for cnt in range(depth):
            in_channels = self.in_channels if cnt == 0 else out_channels
            out_channels = self.start_filts * (2**cnt)
            self.hidden_dims.append(out_channels)
            pooling = False if cnt == 0 else True
            self.down.append(
                Down(in_channels, out_channels, self.dim, pooling=pooling,
                     batchnorm=batchnorm))

        # Create the decoder pathway
        # - careful! decoding only requires depth-1 blocks
        for cnt in range(depth - 1):
            in_channels = out_channels
            out_channels = in_channels // 2
            shape = None
            if self.shapes is not None:
                shape = self.shapes[cnt + 1]
            self.up.append(
                Up(in_channels, out_channels, up_mode=up_mode, dim=self.dim,
                   merge_mode="none", batchnorm=batchnorm, shape=shape))

        # Add the list of modules to current module
        self.down = nn.Sequential(*self.down)
        hidden_dim = self.hidden_dims[-1] * np.prod(self.shapes[0])
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.up = nn.Sequential(*self.up)
        self.conv_final = Conv1x1x1(out_channels, self.in_channels, self.dim)
        self.logit = nn.Tanh()

        # Kernel initializer
        self.kernel_initializer()

    def encode(self, x):
        """ Encodes the input by passing through the encoder network
        and returns the latent codes.

        Parameters
        ----------
        x: Tensor, (N, C, X, Y, Z)
            input tensor to encode.

        Returns
        -------
        mu: Tensor (N, D)
            mean of the latent Gaussian.
        logvar: Tensor (N, D)
            standard deviation of the latent Gaussian.
        """
        logger.debug("Encode...")
        self.debug("input", x)
        x = self.down(x)
        self.debug("down", x)
        x = torch.flatten(x, start_dim=1)
        self.debug("flatten", x)
        # Split x into mu and var components of the latent Gaussian
        # distribution
        z_mu = self.mu(x)
        z_logvar = self.var(x)
        self.debug("z_mu", z_mu)
        self.debug("z_logvar", z_logvar)
        return z_mu, z_logvar

    def decode(self, x_sample):
        """ Maps the given latent codes onto the image space.

        Parameters
        ----------
        x_sample: Tensor (N, D)
            sample from the distribution having latent parameters mu, var.

        Returns
        -------
        x: Tensor, (N, C, X, Y, Z)
            the prediction.
        """
        logger.debug("Decode...")
        self.debug("x sample", x_sample)
        x = self.latent_to_hidden(x_sample)
        self.debug("hidden", x)
        x = x.view(-1, self.hidden_dims[-1], *self.shapes[0])
        self.debug("view", x)
        x = self.up(x)
        self.debug("up", x)
        x = self.conv_final(x)
        self.debug("final", x)
        return self.logit(x)

    def reparameterize(self, z_mu, z_logvar):
        """ Reparameterization trick to sample from N(mu, var) from
        N(0,1).

        Parameters
        ----------
        mu: Tensor (N, D)
            mean of the latent Gaussian.
        logvar: Tensor (N, D)
            standard deviation of the latent Gaussian.

        Returns
        -------
        x_sample: Tensor (N, D)
            sample from the distribution having latent parameters mu, var.
        """
        logger.debug("Reparameterize...")
        self.debug("z_mu", z_mu)
        self.debug("z_logvar", z_logvar)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        self.debug("x sample", x_sample)
        return x_sample

    def forward(self, x):
        logger.debug("VUnet...")
        z_mu, z_logvar = self.encode(x)
        x_sample = self.reparameterize(z_mu, z_logvar)
        predicted = self.decode(x_sample)
        return predicted, {"z_mu": z_mu, "z_logvar": z_logvar}


class DecodeLoss(object):
    """ VAE consists of two loss functions:

    1. Reconstruction loss: how well we can reconstruct the image
    2. KL divergence loss: how off the distribution over the latent space is
       from the prior. Given the prior is a standard Gaussian and the inferred
       distribution is a Gaussian with a diagonal covariance matrix,
       the KL-divergence becomes analytically solvable.

    loss = REC_loss + k1 * KL_loss.
    """
    def __init__(self, k1=1, rec_loss="mse", nodecoding=False):
        super(DecodeLoss, self).__init__()
        if rec_loss not in ("mse", "bce"):
            raise ValueError("Requested loss not yet supported.")
        self.layer_outputs = None
        self.k1 = k1
        self.rec_loss = rec_loss
        self.nodecoding = nodecoding

    def __call__(self, x_sample, x):
        if self.nodecoding:
            return -1
        if self.layer_outputs is None:
            raise ValueError("The model needs to return the latent space "
                             "distribution parameters z_mu, z_logvar.")
        z_mu = self.layer_outputs["z_mu"]
        z_logvar = self.layer_outputs["z_logvar"]
        if self.rec_loss == "bce":
            recon_loss = func.binary_cross_entropy(
                x_sample, x, reduction="sum")
        else:
            recon_loss = func.mse_loss(
                x_sample, x, reduction="mean")
        # kld_loss = 0.5 * torch.sum(
        #     torch.exp(z_logvar) + z_mu**2 - 1.0 - z_logvar)
        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim=-1), dim=0)

        return recon_loss + self.k1 * kld_loss
