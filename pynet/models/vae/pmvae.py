# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Pathway Modules Variational Auto-Encoder (pmVAE).

[1] pmVAE: Learning Interpretable Single-Cell Representations with Pathway
Modules, Gilles Gut, biorxiv 2021.

Code: https://github.com/ratschlab/pmvae
"""

# Imports
import logging
import math
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import torch
import torch.nn as nn
import torch.nn.functional as func
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks, init_weight


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae", "genetic"))
class PMVAE(nn.Module):
    def __init__(self, membership_mask, latent_dim, hidden_layers,
                 bias_last_layer=False, add_auxiliary_module=True,
                 terms=None, activation=None):
        """ pmVAE constructs a pathway-factorized latent space.

        Parameters
        ----------
        membership_mask: bool array (pathways, genes)
            a binary mask encoding which genes belong to wich pathways.
        latent_dim: int
            the dimension of each module latent space.
        hidden_layers: list of int
            the dimension of each module encoder/decoder hidden layer.
        bias_last_layer: bool, default False
            use a bias term on the final decoder output.
        add_auxiliary_module: bool, default True
            include a fully connected pathway module.
        terms: list of str (pathways, ), default None
            the pathway names.
        activation: klass, default None
            the activation function.
        """
        super(PMVAE, self).__init__()
        self.n_annotated_modules, self.num_feats = membership_mask.shape
        if isinstance(membership_mask, pd.DataFrame):
            terms = membership_mask.index
            membership_mask = membership_mask.values
        self.add_auxiliary_module = add_auxiliary_module
        if add_auxiliary_module:
            membership_mask = np.vstack(
                    (membership_mask, np.ones_like(membership_mask[0])))
            if terms is not None:
                terms = list(terms) + ["AUXILIARY"]
        self.activation = activation or nn.ELU
        # Then encoder maps the input data to the latent space.
        self.encoder = PMVAE.build_encoder(
            membership_mask, hidden_layers, latent_dim, self.activation,
            batch_norm=True)
        # The decoder maps a code to the output of each module.
        # The merger connects each module output to its genes.
        self.decoder, self.merger = PMVAE.build_decoder(
            membership_mask, hidden_layers, latent_dim, self.activation,
            batch_norm=True, bias_last_layer=bias_last_layer)
        self.membership_mask = membership_mask
        self.module_isolation_mask = PMVAE.build_module_isolation_mask(
            self.membership_mask.shape[0], hidden_layers[-1])
        self._latent_dim = latent_dim
        self._hidden_layers = hidden_layers
        assert len(terms) == len(self.membership_mask)
        self.terms = list(terms)
        self.kernel_initializer()

    def kernel_initializer(self):
        """ Init network weights.
        """
        for module in self.modules():
            if isinstance(module, MaskedLinear):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                    module.weight)
                limit = math.sqrt(6 / fan_in)
                nn.init.uniform_(module.weight, a=-limit, b=limit)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def build_base_masks(membership_mask, hidden_layers, latent_dim):
        """ Builds the masks used by the encoders/decoders.

        Parameters
        ----------
        membership_mask: bool array (pathways, genes)
            a binary mask encoding which genes belong to wich pathways.
        latent_dim: int
            the dimension of each module latent space.
        hidden_layers: list of int
            the dimension of each module encoder/decoder hidden layer.

        Returns
        -------
        base: list of array
            pathway mask assigns genes to pathway modules, and separation
            masks keep modules separated. Encoder modifies the last
            separation mask to give mu/logvar, and the decoder reverses and
            transposes the masks.
        """
        n_modules, n_feats = membership_mask.shape
        base = []
        base.append(PMVAE.build_pathway_mask(
            n_feats, membership_mask, hidden_layers[0]))
        dims = hidden_layers + [latent_dim]
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            base.append(PMVAE.build_separation_mask(
                input_dim, output_dim, n_modules))
        base = [mask.astype(np.float32) for mask in base]
        return base

    @staticmethod
    def build_pathway_mask(nfeats, membership_mask, hidden_layers):
        """ Connects genes to pathway modules.
        Repeats the membership mask for each module input node.
        See M in Methods 2.2.
        """
        return np.repeat(membership_mask, hidden_layers, axis=0).T

    @staticmethod
    def build_separation_mask(input_dim, out_put_dim, nmodules):
        """ Removes connections betweens pathway modules.
        Block diagonal matrix, see Sigma in Methods 2.2.
        """
        blocks = [np.ones((input_dim, out_put_dim))] * nmodules
        return block_diag(*blocks)

    @staticmethod
    def build_module_isolation_mask(nmodules, module_output_dim):
        """ Isolates a single module for gradient steps.
        Used for the local reconstruciton terms, drops all modules except one.
        """
        blocks = [np.ones((1, module_output_dim))] * nmodules
        return block_diag(*blocks)

    @staticmethod
    def build_encoder(membership_mask, hidden_layers, latent_dim,
                      activation, batch_norm=True):
        """ Build the encoder module.
        """
        masks = PMVAE.build_base_masks(
            membership_mask, hidden_layers, latent_dim)
        masks[-1] = np.hstack((masks[-1], masks[-1]))
        masks = [torch.from_numpy(mask.T) for mask in masks]

        modules = []
        in_features = membership_mask.shape[1]
        for cnt, mask in enumerate(masks):
            out_features = mask.shape[0]
            modules.append(MaskedLinear(in_features, out_features, mask))
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_features, eps=0.001,
                                              momentum=0.99))
            if cnt != (len(masks) - 1):
                modules.append(activation())
            in_features = out_features
        encoder = nn.Sequential(*modules)

        return encoder

    @staticmethod
    def build_decoder(membership_mask, hidden_layers, latent_dim,
                      activation, batch_norm=True, bias_last_layer=False):
        """ Build the decoder/merger modules.
        """
        masks = PMVAE.build_base_masks(
            membership_mask, hidden_layers, latent_dim)
        in_features = masks[-1].shape[1]
        masks = [torch.from_numpy(mask) for mask in masks[::-1]]

        modules = []
        for mask in masks[:-1]:
            out_features = mask.shape[0]
            modules.append(MaskedLinear(in_features, out_features, mask))
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_features, eps=0.001,
                                              momentum=0.99))
            modules.append(activation())
            in_features = out_features
        decoder = nn.Sequential(*modules)

        merger = MaskedLinear(in_features, masks[-1].shape[0], masks[-1],
                              bias=bias_last_layer)

        return decoder, merger

    def encode(self, x):
        """ Computes the inference distribution q(z | x).

        Parameters
        ----------
        x: torch.Tensor (batch_size, data_size)
            the input data.

        Returns
        -------
        q(z | x): @callable
            the distribution q(z | x) with shape (batch_size, latent_dim.
        """
        params = self.encoder(x)
        mu, logvar = torch.split(
            params, split_size_or_sections=(params.size(dim=1) // 2), dim=1)
        return mu, logvar

    def decode(self, z):
        """ Computes the generative distribution p(x | z).

        Parameters
        ----------
        z: torch.Tensor (batch_size, latent_dim)
            the stochastic latent state z.

        Returns
        -------
        p(x | z): @callable
            the distribution p(x | z) with shape (batch_size, data_size).
        """
        module_outputs = self.decoder(z)
        global_recon = self.merger(module_outputs, **kwargs)
        return global_recon

    def reparametrize(self, mu, logvar):
        """ Implement the reparametrization trick.
        """
        eps = torch.randn_like(logvar)
        return mu + torch.exp(logvar / 2.) * eps

    def forward(self, x):
        """ The forward method.
        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        module_outputs = self.decoder(z)
        global_recon = self.merger(module_outputs)
        return global_recon, {"z": z, "module_outputs": module_outputs,
                              "mu": mu, "logvar": logvar, "model": self}

    def get_masks_for_local_losses(self):
        """ Get module/pathway associated masks.
        """
        if self.add_auxiliary_module:
            return zip(self.membership_mask[:-1],
                       self.module_isolation_mask[:-1])
        return zip(self.membership_mask, self.module_isolation_mask)

    def latent_space_names(self, terms=None):
        """ Get latent space associated names.
        """
        terms = self.terms or terms
        assert terms is not None, "Need to specify gene set terms."

        if (self.add_auxiliary_module and
                (len(terms) == self.n_annotated_modules)):
            terms = list(terms) + ["AUXILIARY"]

        z = self._latent_dim
        repeated_terms = np.repeat(terms, z)
        index = np.tile(range(z), len(terms)).astype(str)
        latent_dim_names = map("-".join, zip(repeated_terms, index))

        return list(latent_dim_names)


class MaskedLinear(nn.Linear):
    """ Masked Linear module.
    """
    def __init__(self, in_features, out_features, mask, *args, **kwargs):
        """ Init class.

        Parameters
        ----------
        in_features: int
            size of each input sample.
        out_features: int
            size of each output sample.
        mask: torch.Tensor
            mask weights with this boolean tensor.
        """
        super(MaskedLinear, self).__init__(
            in_features, out_features, *args, **kwargs)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, inputs):
        """ Forward method.
        """
        assert self.mask.shape == self.weight.shape
        return func.linear(inputs, self.weight * self.mask, self.bias)
