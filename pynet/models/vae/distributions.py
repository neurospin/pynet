# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common distributions.
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import (
    MultivariateNormal, Bernoulli, Independent, RelaxedOneHotCategorical,
    LowRankMultivariateNormal, kl_divergence)
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
        return Independent(Bernoulli(logits=logits, validate_args=False),
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
