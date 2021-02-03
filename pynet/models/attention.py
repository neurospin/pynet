# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
The Spatiotemporal Attention Autoencoder network (STAAENet).
"""

# Imports
import logging
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import torch
import torch.nn as nn
import torch.nn.functional as func

# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("encoder", ))
class STAAENet(nn.Module):
    """ SpatioTemporal Attention AutoEncoder (STAAE).
    """
    def __init__(self, input_dim, nodecoding=False):
        """ Init class.

        Parameters
        ----------
        input_dim: int
            the input dimension.
        nodecoding: bool, default False
            if set do not apply the decoding.
        """
        super(STAAENet, self).__init__()
        self.input_dim = input_dim
        self.nodecoding = nodecoding

        # Build Encoder
        self.enc_dense1 = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh())
        self.enc_dense2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh())
        self.enc_attention = SelfAttention(128, 64)
        self.encoder = nn.Sequential(
            self.enc_dense1, self.enc_dense2, self.enc_attention)

        # Build Decoder
        self.dec_attention = SelfAttention(64, 128)
        self.dec_dense1 = nn.Sequential(
            nn.Linear(128, 512),
            nn.Tanh())
        self.dec_dense2 = nn.Sequential(
            nn.Linear(512, self.input_dim),
            nn.Tanh())
        self.decoder = nn.Sequential(
            self.dec_attention, self.dec_dense1, self.dec_dense2)

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
        x = self.encoder(x)
        return x

    def decode(self, x):
        """ Maps the given latent codes onto the image space.

        Parameters
        ----------
        x: Tensor (N, D)
            sample from the distribution having latent parameters mu, var.

        Returns
        -------
        x: Tensor, (N, C, F)
            the prediction.
        """
        logger.debug("Decode...")
        self.debug("x", x)
        x = self.decoder(x)
        self.debug("decoded", x)
        return x

    def forward(self, x, **kwargs):
        logger.debug("STAAE Net...")
        code = self.encode(x)
        if self.nodecoding:
            return code
        else:
            return self.decode(code)

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


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.output_dim = output_dim
        self.kernel = nn.Parameter(
            torch.zeros(3, input_dim, output_dim), requires_grad=True)
        nn.init.uniform_(self.kernel)

    def forward(self, x, **kwargs):
        logger.debug("Self Attention...")
        self.debug("x", x)
        self.debug("kernel", self.kernel)
        WQ = torch.matmul(x, self.kernel[0])
        self.debug("WQ", WQ)
        WK = torch.matmul(x, self.kernel[1])
        self.debug("WK", WK)
        WV = torch.matmul(x, self.kernel[2])
        self.debug("WV", WV)

        QK = torch.matmul(WQ, WK.permute(0, 2, 1))
        QK = QK / (self.output_dim ** 0.5)
        self.debug("QK", QK)
        QK = torch.softmax(QK, dim=0)
        self.debug("QK", QK)
        V = torch.matmul(QK, WV)
        self.debug("V", V)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

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
