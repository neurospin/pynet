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
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, kl_divergence
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import numpy as np
from .vanillanet import DVAENet


# Global parameters
logger = logging.getLogger("pynet")


# @Networks.register
# @DeepLearningDecorator(family=("encoder", "vae"))
class MCVAE(nn.Module):
    """ Sparse Multi-Channel Variational Autoencoderfor (sMCVAE).

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
        noise_fixed: bool, default False
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
            self.vae_class = DVAENet
        else:
            raise ValueError("unknown VAE model.")
        self.vae_kwargs = vae_kwargs or {}
        self.nodecoding = nodecoding
        self.init_vae()

    def init_vae(self):
        """ Create one VAE model per channel.
        """
        if self.sparse:
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.latent_dim).normal_(0, 0.01))
        else:
            self.log_alpha = None
        vae = []
        for c_idx in range(self.n_channels):
            # ToDo: adapt VAE class - init + sparse
            vae.append(
                self.vae_class(
                    latent_dim=self.latent_dim,
                    input_dim=self.n_feats[c_idx],
                    # noise_init_logvar=self.noise_init_logvar,
                    # noise_fixed=self.noise_fixed,
                    # sparse=self.sparse,
                    # log_alpha=self.log_alpha,
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
        return [self.vae[c_idx].encode(x[c_idx].unsqueeze(1))
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

    def forward(self, x):
        """ Forward pass.
        """
        q = self.encode(x)
        z = [self.vae[c_idx].reparameterize(*q[c_idx])
             for c_idx in range(self.n_channels)]
        if self.nodecoding:
            return z, {"q": q}
        else:
            p = self.decode(z)
            return p, {"q": q}


class MCVAELoss(object):
    """ MCVAE consists of two loss functions:

    1. KL divergence loss: how off the distribution over the latent space is
       from the prior. Given the prior is a standard Gaussian and the inferred
       distribution is a Gaussian with a diagonal covariance matrix,
       the KL-divergence becomes analytically solvable.
    2. log-likelihood LL

    loss = beta * KL_loss + LL_loss.
    """
    def __init__(self, n_channels, beta=1., enc_channels=None,
                 dec_channels=None, sparse=False, nodecoding=False):
        """ Init class.

        Parameters
        ----------
        n_channels: int
            the number of channels.
        beta, float, default 1.
            for beta-VAE.
        enc_channels: list of int, default None
            encode only these channels (for kl computation).
        dec_channels: list of int, default None
            decode only these channels (for ll computation).
        sparse: bool, default False
            use sparsity contraint.
        nodecoding: bool, default False
            if set do not apply the decoding.
        """
        super(MCVAELoss, self).__init__()
        self.n_channels = n_channels
        self.beta = beta
        self.sparse = sparse
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        if enc_channels is None:
            self.enc_channels = list(range(n_channels))
        else:
            assert(len(enc_channels) <= n_channels)
        if dec_channels is None:
            self.dec_channels = list(range(n_channels))
        else:
            assert(len(dec_channels) <= n_channels)
        self.n_enc_channels = len(self.enc_channels)
        self.n_dec_channels = len(self.dec_channels)
        self.nodecoding = nodecoding

        # ToDo: fix
        tmp_noise_par = torch.FloatTensor(1, 4).fill_(-3)
        self.pi_logvar = torch.nn.Parameter(
            data=tmp_noise_par, requires_grad=False)
        del tmp_noise_par

    def __call__(self, x_pred, x_true, q):
        """ Compute loss.

        Parameters
        ----------
        x_pred: list of Tensor, (C,) -> (N, F)
            reconstructed channels data.
        x_true: list of Tensor, (C,) -> (N, F)
            inputs channels data.
        q: list of 2-uplet
            the distribution parameters z_mu, z_logvar for each channel.
        """
        if self.nodecoding:
            return -1
        q = [Normal(loc=mu, scale=logvar.exp().pow(0.5)) for mu, logvar in q]
        kl = self.compute_kl(q, self.beta)
        ll = self.compute_ll(p=x_pred, x=x_true)
        print(kl.shape)
        print(ll.shape)
        total = kl - ll
        return total

    def compute_kl(self, q, beta):
        kl = 0
        if not self.sparse:
            for c_idx, qi in enumerate(q):
                if c_idx in self.enc_channels:
                    kl += kl_divergence(qi, Normal(0, 1))
        else:
            for c_idx, qi in enumerate(q):
                if c_idx in self.enc_channels:
                    kl += _kl_log_uniform(
                        mu=qi.loc, logvar=qi.scale.pow(2).log()).sum(
                            1, keepdims=True).mean(0)
        return beta * kl / self.n_enc_channels

    def compute_ll(self, p, x):
        # p[x][z]: p(x|z)
        ll = 0
        self.pi_logvar = self.pi_logvar.to(x[0].get_device())
        for c_idx1 in range(self.n_channels):
            for c_idx2 in range(self.n_channels):
                if c_idx1 in self.dec_channels and c_idx2 in self.enc_channels:
                    pi = Normal(loc=p[c_idx1][c_idx2],
                                scale=self.pi_logvar.exp().pow(0.5))
                    # average ll per observation
                    ll += _compute_ll(p=pi, x=x[c_idx1]).mean(0)
        return ll / self.n_enc_channels / self.n_dec_channels


def _compute_ll(p, x):
    return p.log_prob(x).sum(1, keepdims=True)


def _kl_log_uniform(mu, logvar):
    """
    Paragraph 4.2 from:
    Variational Dropout Sparsifies Deep Neural Networks
    Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
    https://arxiv.org/abs/1701.05369
    https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/
    blob/master/KL%20approximation.ipynb
    """
    log_alpha = compute_log_alpha(mu, logvar)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    neg_kl = (k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 *
              torch.log1p(torch.exp(-log_alpha)) - k1)
    return - neg_kl
