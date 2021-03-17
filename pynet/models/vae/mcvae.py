# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Sparse Multi-Channel Variational Autoencoderfor the Joint Analysis of
Heterogeneous Data.
"""

# Imports
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, kl_divergence
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks, Losses
from .vae import VAENet


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae"))
class MCVAE(nn.Module):
    """ Sparse Multi-Channel Variational Autoencoder (sMCVAE).

    Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Luigi Antelmi, Nicholas Ayache, Philippe Robert,
    Marco Lorenzi Proceedings of the 36th International Conference on Machine
    Learning, PMLR 97:302-311, 2019.
    """
    def __init__(self, latent_dim, n_channels, n_feats,
                 noise_init_logvar=-3, noise_fixed=False, sparse=False,
                 vae_model="dense", vae_kwargs=None, nodecoding=False):
        """ Init class.

        Parameters
        ----------
        latent_dim: int
            the number of latent dimensions.
        n_channels: int
            the number of channels.
        n_feats: list of int
            each channel input dimensions.
        noise_init_logvar: float, default -3
            default noise parameters values.
        noise_fixed: bool, default False
            if set not set do not required gradients on noise parameters.
        sparse: bool, default False
            use sparsity contraint.
        vae_model: str, default "dense"
            the VAE network used to encode each channel.
        vae_kwargs: dict, default None
            extra parameters passed initialization of the VAE model.
        nodecoding: bool, default False
            if set do not apply the decoding.
        """
        super(MCVAE, self).__init__()
        assert(n_channels == len(n_feats))
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.sparse = sparse
        self.noise_init_logvar = noise_init_logvar
        self.noise_fixed = noise_fixed
        if vae_model == "dense":
            self.vae_class = VAENet
        else:
            raise ValueError("Unknown VAE model.")
        self.vae_kwargs = vae_kwargs or {}
        self.nodecoding = nodecoding
        self.init_vae()

    def init_vae(self):
        """ Create one VAE model per channel.
        """
        if self.sparse:
            self.log_alpha = nn.Parameter(
                torch.FloatTensor(1, self.latent_dim).normal_(0, 0.01))
        else:
            self.log_alpha = None
        vae = []
        for c_idx in range(self.n_channels):
            if "conv_flts" not in self.vae_kwargs:
                self.vae_kwargs["conv_flts"] = None
            if "dense_hidden_dims" not in self.vae_kwargs:
                self.vae_kwargs["dense_hidden_dims"] = None
            vae.append(
                self.vae_class(
                    input_channels=1,
                    input_dim=self.n_feats[c_idx],
                    latent_dim=self.latent_dim,
                    noise_out_logvar=self.noise_init_logvar,
                    noise_fixed=self.noise_fixed,
                    sparse=self.sparse,
                    act_func=torch.nn.Tanh,
                    final_activation=False,
                    log_alpha=self.log_alpha,
                    **self.vae_kwargs))
        self.vae = torch.nn.ModuleList(vae)

    def encode(self, x):
        """ Encodes the input by passing through the encoder network
        and returns the latent distribution for each channel.

        Parameters
        ----------
        x: list of Tensor, (C,) -> (N, Fc)
            input tensors to encode.

        Returns
        -------
        out: list of 2-uplet (C,) -> (N, D)
            each channel distribution parameters mu (mean of the latent
            Gaussian) and logvar (standard deviation of the latent Gaussian).
        """
        return [self.vae[c_idx].encode(x[c_idx])
                for c_idx in range(self.n_channels)]

    def decode(self, z):
        """ Maps the given latent codes onto the image space.

        Parameters
        ----------
        z: list of Tensor (N, D)
            sample from the distribution having latent parameters mu, var.

        Returns
        -------
        p: list of Tensor, (N, C, F)
            the prediction p(x|z).
        """
        p = []
        for c_idx1 in range(self.n_channels):
            pi = [self.vae[c_idx1].decode(z[c_idx2])
                  for c_idx2 in range(self.n_channels)]
            p.append(pi)
            del pi
        return p

    def reconstruct(self, p):
        x_hat = []
        for c_idx1 in range(self.n_channels):
            x_tmp = torch.stack([
                p[c_idx1][c_idx2].loc.detach()
                for c_idx2 in range(self.n_channels)]).mean(dim=0)
            x_hat.append(x_tmp.cpu().numpy())
            del x_tmp
        return x_hat

    def forward(self, x):
        qs = self.encode(x)
        z = [q.rsample() for q in qs]
        if self.nodecoding:
            return z, {"q": qs, "x": x}
        else:
            p = self.decode(z)
            return p, {"q": qs, "x": x}

    def p_to_prediction(self, p):
        """ Get the prediction from various types of distributions.
        """
        if isinstance(p, list):
            return [self.p_to_prediction(_p) for _p in p]
        elif isinstance(p, Normal):
            pred = p.loc.cpu().detach().numpy()
        elif isinstance(p, Bernoulli):
            pred = p.probs.cpu().detach().numpy()
        else:
            raise NotImplementedError
        return pred

    def apply_threshold(self, z, threshold, keep_dims=True, reorder=False):
        """ Apply dropout threshold.

        Parameters
        ----------
        z: Tensor
            distribution samples.
        threshold: float
            dropout threshold.
        keep_dims: bool default True
            dropout lower than threshold is set to 0.
        reorder: bool default False
            reorder dropout rates.

        Returns
        -------
        z_keep: list
            dropout rates.
        """
        assert(threshold <= 1.0)
        order = torch.argsort(self.dropout).squeeze()
        keep = (self.dropout < threshold).squeeze()
        z_keep = []
        for drop in z:
            if keep_dims:
                drop[:, ~keep] = 0
            else:
                drop = drop[:, keep]
                order = torch.argsort(
                    self.dropout[self.dropout < threshold]).squeeze()
            if reorder:
                drop = drop[:, order]
            z_keep.append(drop)
            del drop
        return z_keep

    @property
    def dropout(self):
        if self.sparse:
            alpha = torch.exp(self.log_alpha.detach())
            return alpha / (alpha + 1)
        else:
            raise NotImplementedError
