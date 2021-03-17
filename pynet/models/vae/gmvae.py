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

Two implementations are proposed:

* VAEGMP is an adaptation of VAE to make use of a Gaussian Mixture prior,
  instead of a standard Normal distribution.
* GMVAE is an attempt to replicate the work described in [1] and [2]

[1] Gaussian Mixture VAE: Lessons in Variational Inference, Generative Models,
and Deep Nets: http://ruishu.io/2016/12/25/gmvae
[2] Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders
Nat Dilokthanakul, arXiv 2017.
Code: https://github.com/jariasf/GMVAE
Code: https://github.com/mazrk7/gmvae
"""

# Imports
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, Categorical, Independent
try:
    from torch.distributions import MixtureSameFamily
except:
    pass
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks, init_weight
from pynet.models.vae.vae import Encoder
from pynet.models.vae.distributions import (
    ConditionalNormal, Gaussian, ConditionalBernoulli, ConditionalCategorical)


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
        prior_gmm: @callable, default None
            a callable that implements the prior distribution p(z | y)
            Must accept as argument the y discrete variable and return
            a tf.distributions.MultivariateNormalDiag distribution.
        decoder: : @callable, default None
            a callable that implements the generative distribution
            p(x | z). Must accept as arguments the encoded latent state z
            and return a subclass of tf.distributions.Distribution that
            can be used to evaluate the log_prob of the targets.
        encoder_y: : @callable, default None
            a callable that implements the inference q(y | x) over
            the discrete latent variable y.
        encoder_gmm: : @callable, default None
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
            the input data to the inference network.

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
            the input data.
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
            the input data.

        Returns
        -------
        recon: torch.Tensor (batch_size, data_size)
            the reconstruucted data.
        """
        z = self.transform(x)
        recon = self.generate_sample_data(z=z)
        return recon

    def generate_sample_data(self, z=None, num_samples=1):
        """ Generates mean sample data from the model.

        Can provide latent variable 'z' to generate data for
        this point in the latent space, else draw from prior.

        Parameters
        ----------
        z: torch.Tensor (num_samples, mix_components, latent_size)
            the stochastic latent state z.

        Returns
        -------
        recon: torch.Tensor (batch_size, data_size)
            the reconstructed mean samples data.
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
            the input data.

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


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae"))
class VAEGMPNet(nn.Module):
    """ Implementation of a Variational Autoencoder (VAE) with Gaussian
    Mixture Prior (GMP).
    """
    def __init__(self, input_dim, latent_dim, n_mix_components=1,
                 dense_hidden_dims=None, sigma_min=0.001, raw_sigma_bias=0.25,
                 gen_bias_init=0, dropout=0, prior=None, encoder=None,
                 decoder=None, random_seed=None):
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
        dropout: float, default 0
            define the dropout rate.
        prior: @callable, default None
            a distribution that implements p(z).
        encoder: @callable, default None
            a distribution that implements inference q(z | x).
        decoder: @callable, default None
            a distribution that implements p(x | z). Must accept as arguments
            the latent state z and return a distribution that
            can be used to evaluate the log_prob of the targets.
        random_seed: int, default None
            the seed for the random operations.
        """
        super(VAEGMPNet, self).__init__()
        self.n_mix_components = n_mix_components
        self.random_seed = random_seed

        # Prior p(z) is a learned mixture of Gaussians, where mu and
        # sigma are output from a fully connected network
        if prior is not None:
            self._prior = prior
        else:
            if self.n_mix_components > 1:
                loc = nn.Parameter(torch.zeros(
                    self.n_mix_components, latent_dim), requires_grad=True)
                raw_scale = nn.Parameter(torch.ones(
                    self.n_mix_components, latent_dim), requires_grad=True)
                mixture_probs = nn.Parameter(torch.ones(
                    self.n_mix_components) / self.n_mix_components,
                    requires_grad=True)
                mix = Categorical(probs=mixture_probs)
                comp = Independent(
                    Normal(loc=loc, scale=func.softplus(raw_scale)),
                    reinterpreted_batch_ndims=1)
                self._prior = MixtureSameFamily(
                    mixture_distribution=mix, component_distribution=comp)
            else:
                self._prior = Normal(loc=torch.zeros(latent_dim), scale=1)

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

        # A callable that implements the inference distribution q(z | x)
        if encoder is not None:
            self._encoder = encoder
        else:
            self._encoder = ConditionalNormal(
                input_dim=input_dim,
                final_dim=latent_dim,
                dense_hidden_dims=dense_hidden_dims,
                sigma_min=sigma_min,
                raw_sigma_bias=raw_sigma_bias)

    def prior(self):
        """ Get the prior distribution p(z).

        Returns
        -------
        p(z): @callable
            the distribution p(z) with shape (batch_size, latent_size).
        """
        return self._prior

    def decoder(self, z):
        """ Computes the generative distribution p(x | z).

        Parameters
        ----------
        z: torch.Tensor (batch_size, latent_size)
            the stochastic latent state z.

        Returns
        -------
        p(x | z): @callable
            the distribution p(x | z) with shape (batch_size, data_size).
        """
        return self._decoder([z])

    def encoder(self, x):
        """ Computes the inference distribution q(z | x).

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            the input data.

        Returns
        -------
        q(z | x): @callable
            the distribution q(z | x) with shape (batch_size, latent_size).
        """
        return self._encoder([x])

    def reconstruct(self, x):
        """ Reconstruct the data from the model.

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            the input data.

        Returns
        -------
        recon: torch.Tensor (batch_size, data_size)
            the reconstructed data.
        """
        q_z = self.encoder(x)
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        if self.training:
            z = q_z.sample()
        else:
            z = q_z.mean()
        p_x_given_z = self.decoder(z)
        recon = p_x_given_z.mean()
        return recon

    def generate_sample_data(self, z=None, num_samples=1):
        """ Generates mean sample data from the model.

        Can provide latent variable 'z' to generate data for
        this point in the latent space, else draw from prior.

        Parameters
        ----------
        z: torch.Tensor (num_samples, latent_size)
            the stochastic latent state z.

        Returns
        -------
        recon: torch.Tensor (batch_size, data_size)
            the reconstructed mean samples data.
        """
        if z is None:
            z = self.generate_samples(num_samples)
        p_x_given_z = self.decoder(z)
        sample_images = p_x_given_z.mean()
        return sample_images

    def transform(self, x):
        """ Transform inputs 'x' to yield mean latent code.

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            the input data.

        Returns
        -------
        z: torch.Tensor (num_samples, latent_size)
            the stochastic latent state z.
        """
        q_z = self.encoder(inputs)
        z = q_z.mean()
        return z

    def generate_samples(self, num_samples):
        """ Generate 'num_samples' samples from the model prior.

        Parameters
        ----------
        num_samples: int
            number of samples to draw from the prior distribution.

        Returns
        -------
        z: Tensor (num_samples, latent_size)
            representing samples drawn from the prior distribution.
        """
        p_z = self.prior()
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        z = p_z.sample(num_samples)
        return z

    def forward(self, x):
        """ The forward method.
        """
        # Prior with Gaussian distribution p(z)
        p_z = self.prior()

        # Encoder accept images x implement q(z | x)
        q_z_given_x = self.encoder(x)
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        z = q_z_given_x.rsample()

        # Generative distribution p(x | z)
        p_x_given_z = self.decoder(z)

        return p_x_given_z, {"p_z": p_z, "q_z_given_x": q_z_given_x, "z": z}
