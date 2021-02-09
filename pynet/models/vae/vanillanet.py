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
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, Bernoulli
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import numpy as np
from .base import BaseVAE


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae"))
class VanillaNet(BaseVAE):
    """
    The model is composed of two sub-networks:

    1. Given x (image), encode it into a distribution over the latent space -
       referred to as Q(z|x).
    2. Given z in latent space (code representation of an image), decode it
       into the image it represents - referred to as f(z).
    """
    def __init__(self, latent_dim, in_channels, hidden_dims, input_shape,
                 num_classes=None, **kwargs):
        """ Init class.

        Parameters
        ----------
        latent_dim: int
            the latent dimension.
        in_channels: int
            number of channels in the input tensor.
        hidden_dims: list of int
            the model hidden dimensions.
        input_shape: uplet
            the tensor data shape (X, Y, Z) used during upsample (by default
            use a scale factor of 2).
        num_classes: int, default None
            the number of classes for the conditioning.
        """
        super(VanillaNet, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.shapes = self.downsample_shape(
            input_shape, nb_iterations=len(hidden_dims) - 1)
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Build Encoder
        modules = []
        for cnt, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1 if cnt == 0 else 2,
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        hidden_dim = hidden_dims[-1] * np.prod(self.shapes[-1])
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Build Decoder
        modules = []
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        hidden_dims.reverse()
        for cnt in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[cnt],
                                       hidden_dims[cnt + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[cnt + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1], out_channels=self.in_channels,
                kernel_size=3, padding=1),
            nn.Sigmoid()
            # nn.Tanh())
        )

        # Digit classifier
        if self.num_classes is not None:
            self.embed_class = nn.Linear(num_classes, np.prod(input_shape))
            self.embed_data = nn.Conv2d(in_channels, in_channels,
                                        kernel_size=1)

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
        x = self.encoder(x)
        self.debug("down", x)
        x = torch.flatten(x, start_dim=1)
        self.debug("flatten", x)
        z_mu = self.fc_mu(x)
        z_logvar = self.fc_var(x)
        self.debug("z_mu", z_mu)
        self.debug("z_logvar", z_logvar)
        if self.use_distributions:
            return Normal(z_mu, z_logvar.exp().pow(0.5))
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
        x = x.view(-1, self.hidden_dims[0], *self.shapes[-1])
        self.debug("view", x)
        x = self.decoder(x)
        self.debug("up", x)
        x = self.final_layer(x)
        self.debug("final", x)
        return x

    def reparameterize(self, z_mu, z_logvar):
        """ Reparameterization trick to sample from N(mu, var) from N(0,1).

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
        x_sample = eps * std + z_mu
        self.debug("x sample", x_sample)
        return x_sample

    def forward(self, x, **kwargs):
        logger.debug("Vanilla Net...")
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        return self.decode(z), {"z_mu": z_mu, "z_logvar": z_logvar}


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae"))
class DVAENet(BaseVAE):
    """
    Dense Variational AutoEncoder (DVAE).

    DEEP VARIATIONAL AUTOENCODER FOR MODELING FUNCTIONAL BRAIN NETWORKS AND
    ADHD IDENTIFICATION, ISBI 2020.

    The model is composed of two sub-networks:

    1. Given x (image), encode it into a distribution over the latent space -
       referred to as Q(z|x).
    2. Given z in latent space (code representation of an image), decode it
       into the image it represents - referred to as f(z).
    """
    def __init__(self,
        latent_dim,
        input_dim,
        hidden_dims=None,
        sparse=False,
        nodecoding=False,
        noise_fixed=False,
        noise_init_logvar=-3,
        log_alpha=None,
        **kwargs,
    ):
        """ Init class.

        Parameters
        ----------
        latent_dim: int
            the latent dimension.
        input_dim: int
            the input dimension.
        hidden_dims: list of int
            the model hidden dimensions.
        nodecoding: bool, default False
            if set do not apply the decoding.
        """
        super(DVAENet, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.nodecoding = nodecoding
        self.sparse = sparse
        self.noise_fixed = noise_fixed
        self.noise_init_logvar = noise_init_logvar
        self.log_alpha = log_alpha

        # Build Encoder
        if hidden_dims is not None:
            modules = []
            e_input_dim = self.input_dim
            for cnt, h_dim in enumerate(hidden_dims):
                logger.debug("Encoding level {0} with dim {1} - {2}...".format(
                    cnt, e_input_dim, h_dim))
                modules.append(
                    nn.Sequential(
                        nn.Linear(e_input_dim, h_dim),
                        nn.Tanh())
                )
                e_input_dim = h_dim
            self.encoder = nn.Sequential(*modules)
            input_dim = hidden_dims[-1]

        self.fc_mu = nn.Linear(input_dim, latent_dim)
        if self.sparse and self.log_alpha is None:
            self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.latent_dim).normal_(0, 0.01))
        elif not self.sparse:
            self.fc_logvar = nn.Linear(input_dim, latent_dim)

        # Build Decoder
        if self.use_distributions:
            tmp_noise_par = torch.FloatTensor(1, self.input_dim).fill_(self.noise_init_logvar)
            if self.noise_fixed:
                self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=False)
            else:
                self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=True)
            del tmp_noise_par
       
        if hidden_dims is not None:
            modules = []
            hidden_dims.reverse()
            d_input_dim = latent_dim
            for cnt, h_dim in enumerate(hidden_dims):
                logger.debug("Decoding level {0} with dim {1} - {2}...".format(
                    cnt, d_input_dim, h_dim))
                modules.append(
                    nn.Sequential(
                        nn.Linear(d_input_dim, h_dim),
                        nn.Tanh())
                )
                d_input_dim = h_dim
            self.decoder = nn.Sequential(*modules)
            latent_dim = hidden_dims[-1]
            
        self.final_layer = nn.Sequential(
            nn.Linear(latent_dim, self.input_dim),
            # nn.Sigmoid()
        )

    def encode(self, x):
        """ Encodes the input by passing through the encoder network
        and returns the latent codes.

        Parameters
        ----------
        x: Tensor, (N, C, F)
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
        if self.hidden_dims is not None:
            x = self.encoder(x)
            self.debug("encoded", x)
        z_mu = self.fc_mu(x)
        if not self.sparse:
            z_logvar = self.fc_logvar(x)
        else:
            z_logvar = compute_logvar(z_mu, self.log_alpha)
        self.debug("z_mu", z_mu)
        self.debug("z_logvar", z_logvar)
        if self.use_distributions:
            return Normal(z_mu, z_logvar.exp().pow(0.5))
        return z_mu, z_logvar

    def decode(self, x_sample):
        """ Maps the given latent codes onto the image space.

        Parameters
        ----------
        x_sample: Tensor (N, D)
            sample from the distribution having latent parameters mu, var.

        Returns
        -------
        x: Tensor, (N, C, F)
            the prediction.
        """
        logger.debug("Decode...")
        self.debug("x sample", x_sample)
        x = x_sample
        if self.hidden_dims is not None:
            x = self.decoder(x_sample)
            self.debug("decoded", x)
        x = self.final_layer(x)
        self.debug("final", x)
        if self.use_distributions:
            # return Bernoulli(probs=x)
            return Normal(x, self.W_out_logvar.exp().pow(0.5))
        return x

    def reparameterize(self, z_mu, z_logvar):
        """ Reparameterization trick to sample from N(mu, var) from N(0,1).

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
        x_sample = eps * std + z_mu
        self.debug("x sample", x_sample)
        return x_sample

    def forward(self, x, **kwargs):
        logger.debug("DVAE Net...")
        if self.use_distributions:
            q = self.encode(x)
            z = q.rsample()
            if self.nodecoding:
                return q
            return self.decode(z), q
        
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        if self.nodecoding:
            return z, {"z_mu": z_mu, "z_logvar": z_logvar}
        return self.decode(z), {"z_mu": z_mu, "z_logvar": z_logvar}
    
    @property
    def dropout(self):
        if self.sparse:
            alpha = torch.exp(self.log_alpha.detach())
            return alpha / (alpha + 1)
        else:
            raise NotImplementedError

def compute_logvar(mu, log_alpha):
	return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)
