# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Base class for VAE
"""

# Imports
import logging
import torch
from torch import nn
from abc import abstractmethod
import numpy as np


# Global parameters
logger = logging.getLogger("pynet")


class BaseVAE(nn.Module):

    def __init__(self, use_distributions=False, *args, **kwargs):
        super(BaseVAE, self).__init__(*args, **kwargs)
        self.use_distributions = use_distributions

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x_sample):
        raise NotImplementedError

    def sample(self, num_samples, current_device):
        """ Samples from the latent space and return the corresponding image
        space map.
        """
        logger.debug("Sample...")
        x_sample = torch.randn(num_samples, self.latent_dim)
        x_sample = x_sample.to(current_device)
        self.debug("x sample", x_sample)
        samples = self.decode(x_sample)
        self.debug("samples", samples)
        return samples

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        pass

    @staticmethod
    def downsample_shape(shape, nb_iterations=1, scale_factor=2):
        shape = np.asarray(shape)
        all_shapes = [shape.astype(int).tolist()]
        for idx in range(nb_iterations):
            shape = np.floor(shape / scale_factor)
            all_shapes.append(shape.astype(int).tolist())
        return all_shapes

    def kernel_initializer(self):
        for module in self.modules():
            self.init_weight(module, self.dim)

    @staticmethod
    def init_weight(module, dim):
        conv_fn = getattr(nn, "Conv{0}".format(dim))
        if isinstance(module, conv_fn):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def debug(name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))
