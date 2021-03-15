# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Gaussian Mixture Variational Auto-Encoder (GMVAE).

Code: https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Networks.py
Code: https://github.com/mazrk7/gmvae
"""

# Imports
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import (
    MultivariateNormal, Bernoulli, Independent, RelaxedOneHotCategorical,
    LowRankMultivariateNormal, kl_divergence)
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks, init_weight
from pynet.models.vae.vae import Encoder


class ConditionalNormal(nn.Module):
    """ A multivariate Normal distribution conditioned on inputs via a dense
    network.
    """
    def __init__(self, input_dim, final_dim, dense_hidden_dims=None,
                 sigma_min=0.0, raw_sigma_bias=0.25,
                 hidden_activation_fn=nn.ReLU, dropout=0):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the input size.
        final_dim: int
            the dimension of the random variable.
        dense_hidden_dims: list of int, default None
            the sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs. If None,
            then the default is a single-layered dense network.
        sigma_min: float, default 0
            the minimum standard deviation allowed.
        raw_sigma_bias: float, default 0.25
            a scalar that is added to the raw standard deviation
            output from the fully connected network. Set to 0.25 by default to
            prevent standard deviations close to 0.
        hidden_activation_fn: @callable, default relu
            the activation function to use on the hidden layers of the fully
            connected network.
        dropout: float, default 0
            define the dropout rate.
        """
        super(ConditionalNormal, self).__init__()
        self._size = final_dim
        if dense_hidden_dims is None:
            self.w_dense = None
            final_hidden_dim = input_dim
        else:
            w_dense_layers = Encoder.init_dense_layers(
                input_dim, dense_hidden_dims, hidden_activation_fn, dropout)
            self.w_dense = nn.Sequential(*w_dense_layers)
            final_hidden_dim = dense_hidden_dims[-1]
        self.w_gaussian = Gaussian(final_hidden_dim, final_dim, sigma_min,
                                   raw_sigma_bias)

    def forward(self, tensor_list):
        """ Creates a Diag Multivariate Normal distribution conditioned on
        the inputs.

        Parameters
        ----------
        tensor_list: list of torch.Tensor
            a list of tensors that will be first concatenatedd on the last
            dimension.
        """
        x = torch.cat(tensor_list, dim=-1)
        if self.w_dense is not None:
            out = self.w_dense(x)
        else:
            out = x
        z_mu, z_var = self.w_gaussian(out)
        return MultivariateNormal(
            loc=z_mu, scale_tril=torch.diag_embed(z_var.pow(0.5)))


class Gaussian(nn.Module):
    """ Sample from a Gaussian distribution.
    """
    def __init__(self, input_dim, z_dim, sigma_min=0.0, raw_sigma_bias=0.25):
        super(Gaussian, self).__init__()
        self._sigma_min = sigma_min
        self._raw_sigma_bias = raw_sigma_bias
        self.w_mu = nn.Linear(input_dim, z_dim)
        self.w_var = nn.Linear(input_dim, z_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        z_mu = self.w_mu(x)
        z_var = func.softplus(self.w_var(x) + self._raw_sigma_bias)
        z_var = torch.clamp(z_var, min=self._sigma_min)
        return z_mu, z_var


class ConditionalBernoulli(nn.Module):
    """ A Bernoulli distribution conditioned on inputs via a dense network.
    """
    def __init__(self, input_dim, final_dim, dense_hidden_dims=None,
                 bias_init=0., hidden_activation_fn=nn.ReLU, dropout=0):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the input size.
        final_dim: int
            the dimension of the random variable.
        dense_hidden_dims: list of int, default None
            the sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs. If None,
            then the default is a single-layered dense network.
        bias_init: float, default 0
            a scalar or tensor that is added to the output of the
            fully connected network and parameterizes the distribution mean.
        hidden_activation_fn: @callable, default relu
            the activation function to use on the hidden layers of the fully
            connected network.
        dropout: float, default 0
            define the dropout rate.
        """
        super(ConditionalBernoulli, self).__init__()
        self._size = final_dim
        self._bias_init = bias_init
        if dense_hidden_dims is None:
            dense_hidden_dims = []
        w_dense_layers = Encoder.init_dense_layers(
            input_dim, dense_hidden_dims + [final_dim], hidden_activation_fn,
            dropout)
        self.w_dense = nn.Sequential(*w_dense_layers[:-1])

    def forward(self, tensor_list):
        """ Creates a Bernoulli distribution conditioned on the inputs.

        Parameters
        ----------
        tensor_list: list of torch.Tensor
            a list of tensors that will be first concatenatedd on the last
            dimension.
        """
        x = torch.cat(tensor_list, dim=-1)
        logits = self.w_dense(x) + self._bias_init
        # Assuming 1-D vector inputs (bs discluded)
        return Independent(Bernoulli(logits=logits),
                           reinterpreted_batch_ndims=1)


class ConditionalCategorical(nn.Module):
    """ A relaxed one hot Categorical distribution conditioned on inputs via a
    dense network.
    """
    def __init__(self, input_dim, final_dim, dense_hidden_dims=None,
                 temperature=1.0, hidden_activation_fn=nn.ReLU, dropout=0):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the input size.
        final_dim: int
            the dimension of the random variable.
        dense_hidden_dims: list of int, default None
            the sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs. If None,
            then the default is a single-layered dense network.
        temperature: float, default 1
            degree of how approximately discrete the distribution is. The
            closer to 0, the more discrete and the closer to infinity, the
            more uniform.
        hidden_activation_fn: @callable, default relu
            the activation function to use on the hidden layers of the fully
            connected network.
        dropout: float, default 0
            define the dropout rate.
        """
        super(ConditionalCategorical, self).__init__()
        self._size = final_dim
        self._temperature = temperature
        if dense_hidden_dims is None:
            dense_hidden_dims = []
        w_dense_layers = Encoder.init_dense_layers(
            input_dim, dense_hidden_dims + [final_dim], hidden_activation_fn,
            dropout)
        self.w_dense = nn.Sequential(*w_dense_layers[:-1])

    def forward(self, tensor_list):
        """ Creates a RelaxedOneHotCategorical distribution conditioned
        on the inputs.

        Parameters
        ----------
        tensor_list: list of torch.Tensor
            a list of tensors that will be first concatenatedd on the last
            dimension.
        """
        x = torch.cat(tensor_list, dim=-1)
        logits = self.w_dense(x)
        return RelaxedOneHotCategorical(self._temperature, logits=logits)


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae", "classifier"))
class GMVAENet(nn.Module):
    """ The Gaussian Mixture VAE architecture.

    Meta-GMVAE: Mixture of Gaussian VAE for Unsupervised Meta-Learning
    Dong Bok Lee, ICLR 2021.

    Gaussian Mixture VAE: Lessons in Variational Inference, Generative Models,
    and Deep Nets: http://ruishu.io/2016/12/25/gmvae

    Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders
    Nat Dilokthanakul, arXiv 2017.
    """
    def __init__(self, input_dim, latent_dim, n_mix_components,
                 dense_hidden_dims=None, sigma_min=0.001, raw_sigma_bias=0.25,
                 dropout=0, temperature=1, gen_bias_init=0.,
                 prior_gmm=None, decoder=None, encoder_y=None,
                 encoder_gmm=None, random_seed=None):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the input size.
        latent_dim: int,
            the size of the stochastic latent state of the GMVAE.
        n_mix_components: int
            the number of mixture components.
        dense_hidden_dims: list of int, default None
            the sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs. If None,
            then the default is a single-layered dense network.
        sigma_min: float, default 0.001
            the minimum value that the standard deviation of the
            distribution over the latent state can take.
        raw_sigma_bias: float, default 0.25
            a scalar that is added to the raw standard deviation
            output from the neural networks that parameterize the prior and
            approximate posterior. Useful for preventing standard deviations
            close to zero.
        dropout: float, default 0
            define the dropout rate.
        temperature: float, default 1
            degree of how approximately discrete the distribution is. The
            closer to 0, the more discrete and the closer to infinity, the
            more uniform.
        gen_bias_init: float, default 0
            a bias to added to the raw output of the fully connected network
            that parameterizes the generative distribution. Useful for
            initalising the mean to a sensible starting point e.g. mean of
            training set.
        prior_gmm: @callalbe, default None
            a callable that implements the prior distribution p(z | y)
            Must accept as argument the y discrete variable and return
            a tf.distributions.MultivariateNormalDiag distribution.
        decoder: : @callalbe, default None
            a callable that implements the generative distribution
            p(x | z). Must accept as arguments the encoded latent state z
            and return a subclass of tf.distributions.Distribution that
            can be used to evaluate the log_prob of the targets.
        encoder_y: : @callalbe, default None
            a callable that implements the inference q(y | x) over
            the discrete latent variable y.
        encoder_gmm: : @callalbe, default None
            a callable that implements the inference q(z | x, y) over
            the continuous latent variable z.
        random_seed: int, default None
            the seed for the random operations.
        """
        super(GMVAENet, self).__init__()
        self.n_mix_components = n_mix_components
        self.random_seed = random_seed

        # Prior p(z | y) is a learned mixture of Gaussians, where mu and
        # sigma are output from a fully connected network conditioned on y
        if prior_gmm is not None:
            self._prior_gmm = prior_gmm
        else:
            self._prior_gmm = ConditionalNormal(
                input_dim=n_mix_components,
                final_dim=latent_dim,
                dense_hidden_dims=None,
                sigma_min=sigma_min,
                raw_sigma_bias=raw_sigma_bias,
                dropout=dropout)

        # The generative distribution p(x | z) is conditioned on the latent
        # state variable z via a fully connected network
        if decoder is not None:
            self._decoder = decoder
        else:
            self._decoder = ConditionalBernoulli(
                input_dim=latent_dim,
                final_dim=input_dim,
                dense_hidden_dims=dense_hidden_dims,
                bias_init=gen_bias_init)

        # A callable that implements the inference distribution q(y | x)
        # Use the Gumbel-Softmax distribution to model the categorical latent
        # variable
        if encoder_y is not None:
            self._encoder_y = encoder_y
        else:
            self._encoder_y = ConditionalCategorical(
                input_dim=input_dim,
                final_dim=n_mix_components,
                temperature=temperature,
                dense_hidden_dims=dense_hidden_dims)

        # A callable that implements the inference distribution q(z | x, y)
        if encoder_gmm is not None:
            self._encoder_gmm = encoder_gmm
        else:
            self._encoder_gmm = ConditionalNormal(
                input_dim=input_dim + n_mix_components,
                final_dim=latent_dim,
                dense_hidden_dims=dense_hidden_dims,
                sigma_min=sigma_min,
                raw_sigma_bias=raw_sigma_bias)

    def prior_gmm(self, y):
        """ Computes the GMM prior distribution p(z | y).

        Parameters
        ----------
        y: torch.Tensor (batch_size, mix_components)
            the discrete intermediate variable y.

        Returns
        -------
        p(z | y): MultivariateNormal (batch_size, latent_size)
            a GMM distribution.
        """
        return self._prior_gmm([y])

    def decoder(self, z):
        """ Computes the generative distribution p(x | z).

        Parameters
        ----------
        z: torch.Tensor (num_samples, mix_components, latent_size)
            the stochastic latent state z.

        Returns
        -------
        p(x | z): Bernoulli (batch_size, data_size)
            a Bernouilli distribution.
        """
        return self._decoder([z])

    def encoder_y(self, x):
        """ Computes the inference distribution q(y | x).

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            the input images to the inference network.

        Returns
        -------
        q(y | x): RelaxedOneHotCategorical (batch_size, mix_components)
           a relaxed one hot Categorical distribution.
        """
        return self._encoder_y([x])

    def encoder_gmm(self, x, y):
        """ Computes the inference distribution q(z | x, y).

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            input images.
        y: torch.Tensor (batch_size, mix_components)
            discrete variable.

        Returns
        -------
        q(z | x, y): MultivariateNormal (batch_size, latent_size)
            a Multivariate Normal Diag distribution.
        """
        return self._encoder_gmm([x, y])

    def reconstruct(self, x):
        """ Reconstruct the data from the model.

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            input images.

        Returns
        -------
        recon: torch.Tensor (batch_size, data_size)
            the reconstruucted data.
        """
        z = self.transform(x)
        recon = self.generate_sample_images(z=z)
        return recon

    def generate_sample_images(self, z=None, num_samples=1):
        """ Generates mean sample images from the model.

        Can provide latent variable 'z' to generate data for
        this point in the latent space, else draw from prior.

        Parameters
        ----------
        z: torch.Tensor (num_samples, mix_components, latent_size)
            the stochastic latent state z.

        Returns
        -------
        recon: torch.Tensor (batch_size, data_size)
            the reconstruucted mean samples data.
        """
        if z is None:
            z = self.generate_samples(num_samples)
        p_x_given_z = self.decoder(z)
        recon = p_x_given_z.mean()
        return recon

    def transform(self, x):
        """ Transform inputs 'x' to yield mean latent code.

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            input images.

        Returns
        -------
        z: torch.Tensor (num_samples, mix_components, latent_size)
            the stochastic latent state z.
        """
        q_y_given_x = self.encoder_y(x)
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        y = q_y_given_x.sample()

        q_z_givenxy = self.encoder_gmm(x, y)
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        z = q_z_givenxy.sample()

        return z

    def generate_samples(self, num_samples, clusters=None):
        """ Samples components from the static latent GMM prior.

        Parameters
        ----------
        num_samples: int
            number of samples to draw from the static GMM prior.
        clusters: list of int, default None
            if desired, can sample from a specific batch of clusters.

        Returns
        -------
        z: Tensor (num_samples, mix_components, latent_size)
            representing samples drawn from each component of the GMM if
            clusters is None else if clusters the Tensor is of shape
            (num_samples, batch_size, latent_size) where batch_size is the
            first dimension of clusters, dependening on how many were supplied.
        """
        # If no specific clusters supplied, sample from each component in GMM
        # Generate outputs over each component in GMM
        if clusters is None:
            clusters = torch.range(0, self.n_mix_components)
        y = func.one_hot(clusters, self.n_mix_components)
        p_z_given_y = self.prior_gmm(y)

        # Draw 'num_samples' samples from each cluster
        # Return shape: [num_samples, mix_components, latent_size]
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        z = p_z_given_y.sample(num_samples)
        # z = torch.reshape(z, [num_samples * self.n_mix_components, -1])
        # z = torch.reshape(z, [num_samples * clusters.size(dim=0), -1])

        return z

    def forward(self, x):
        """ The forward method.
        """
        # Encoder accepts images x and implements q(y | x)
        q_y_given_x = self.encoder_y(x)
        # Sample categorical variable y from the Gumbel-Softmax distribution
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        y = q_y_given_x.sample()

        # Prior accepts y as input and implements p(z | y)
        p_z_given_y = self.prior_gmm(y)

        # Encoder accept images x and y as inputs to implement q(z | x, y)
        q_z_given_xy = self.encoder_gmm(x, y)
        # Sample latent Gaussian variable z
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        z = q_z_given_xy.rsample()

        # Generative distribution p(x | z)
        p_x_given_z = self.decoder(z)

        return p_x_given_z, {"q_y_given_x": q_y_given_x, "y": y,
                             "p_z_given_y": p_z_given_y,
                             "q_z_given_xy": q_z_given_xy, "z": z}
