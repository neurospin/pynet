# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Mixture of Experts VAE with similarity constraint: MoE-Sim-VAE.

Reference: Mixture-of-Experts Variational Autoencoder for Clustering and
Generating from Similarity-Based Representations on Single Cell Data,
Andreas Kopf, arXiv 2020.
"""

# Imports
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
from pynet.models.vae.vae import Encoder
from pynet.models.vae.gmvae import VAEGMPNet
from pynet.models.vae.distributions import ConditionalBernoulli


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae"))
class MOESimVAENet(VAEGMPNet):
    """ Implementation of a Mixture of Experts VAE with similarity constraint.
    """
    def __init__(self, input_dim, latent_dim, n_mix_components=1,
                 dense_hidden_dims=None, classifier_hidden_dims=None,
                 sigma_min=0.001, raw_sigma_bias=0.25, gen_bias_init=0,
                 dropout=0.5, random_seed=None):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the input size.
        latent_dim: int,
            the size of the stochastic latent state of the GMVAE.
        n_mix_components: int, default 1
            the number of components in the mixture prior. If 1, a classical
            VAE is generated with prior z ~ N(0, 1).
        dense_hidden_dims: list of int, default None
            the sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs. If None,
            then the default is a single-layered dense network.
        classifier_hidden_dims: list of int, default None
            the sizes of the hidden layers of the classifier.
        sigma_min: float, default 0.001
            the minimum value that the standard deviation of the
            distribution over the latent state can take.
        raw_sigma_bias: float, default 0.25
            a scalar that is added to the raw standard deviation
            output from the neural networks that parameterize the prior and
            approximate posterior. Useful for preventing standard deviations
            close to zero.
        gen_bias_init: float, default 0
            a bias to added to the raw output of the fully connected network
            that parameterizes the generative distribution. Useful for
            initalising the mean to a sensible starting point e.g. mean of
            training set.
        dropout: float, default 0.5
            define the dropout rate.
        random_seed: int, default None
            the seed for the random operations.
        """
        decoder = nn.ModuleList([
            ConditionalBernoulli(
                input_dim=latent_dim, final_dim=input_dim,
                dense_hidden_dims=dense_hidden_dims, bias_init=gen_bias_init,
                dropout=dropout)
            for k_idx in range(n_mix_components)])
        super(MOESimVAENet, self).__init__(
            input_dim=input_dim, latent_dim=latent_dim,
            n_mix_components=n_mix_components,
            dense_hidden_dims=dense_hidden_dims, sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias, gen_bias_init=gen_bias_init,
            dropout=dropout, decoder=decoder, random_seed=random_seed)
        classifier_layers = Encoder.init_dense_layers(
            input_dim=latent_dim,
            hidden_dims=classifier_hidden_dims + [n_mix_components],
            act_func=nn.Sigmoid, dropout=dropout, final_activation=False)
        self._classifier = nn.Sequential(*classifier_layers)

    def decoder(self, z):
        """ Computes the generative distribution p(x | z).

        Parameters
        ----------
        z: torch.Tensor (num_samples, mix_components, latent_size)
            the stochastic latent state z.

        Returns
        -------
        p(x | z): list of Bernoulli (n_mix_components, )
            a Bernouilli distribution for each decoder.
        """
        return [_dec([z]) for _dec in self._decoder]

    def forward(self, x):
        """ The forward method.
        """
        p_x_given_z, dists = super(MOESimVAENet, self).forward(x)
        z = dists["z"]
        logits = self._classifier(z)
        probs = func.softmax(logits, dim=1)
        dists["logits"] = logits
        dists["probs"] = probs
        dists["model"] = self
        return p_x_given_z, dists
