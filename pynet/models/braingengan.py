# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
3D MRI Brain Generation with Generative Adversarial Networks (BGGAN) with
Variational Auto Encoder (VAE).
"""

# Imports
import logging
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as func
from pynet.utils import Networks


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
class BGDiscriminator(nn.Module):
    """ This is the discriminator part of the BGGAN.
    """
    def __init__(self, in_shape, in_channels=1, out_channels=1,
                 start_filts=64, with_logit=True):
        """ Init class.

        Parameters
        ----------
        in_shape: uplet
            the input tensor data shape (X, Y, Z).
        in_channels: int, default 1
            number of channels in the input tensor.
        out_channels: int, default 1
            number of channels in the output tensor.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        with_logit: bool, default True
            apply the logit function to the result.
        """
        super(BGDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.with_logit = with_logit
        self.in_shape = in_shape
        self.shapes = _downsample_shape(
            self.in_shape, nb_iterations=4, scale_factor=2)
        self.conv1 = nn.Conv3d(
            self.in_channels, self.start_filts, kernel_size=4, stride=2,
            padding=1)
        self.conv2 = nn.Conv3d(
            self.start_filts, self.start_filts * 2, kernel_size=4, stride=2,
            padding=1)
        self.bn2 = nn.BatchNorm3d(self.start_filts * 2)
        self.conv3 = nn.Conv3d(
            self.start_filts * 2, self.start_filts * 4, kernel_size=4,
            stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(self.start_filts * 4)
        self.conv4 = nn.Conv3d(
            self.start_filts * 4, self.start_filts * 8, kernel_size=4,
            stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(self.start_filts * 8)
        self.conv5 = nn.Conv3d(
            self.start_filts * 8, self.out_channels,
            kernel_size=self.shapes[-1], stride=1, padding=0)

    def forward(self, x):
        logger.debug("BGGAN Discriminator...")
        self.debug("input", x)
        h1 = func.leaky_relu(self.conv1(x), negative_slope=0.2)
        self.debug("conv1", h1)
        h2 = func.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        self.debug("conv2", h2)
        h3 = func.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        self.debug("conv3", h3)
        h4 = func.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        self.debug("conv4", h4)
        h5 = self.conv5(h4)
        self.debug("conv5", h5)
        if self.with_logit:
            output = torch.sigmoid(h5.view(h5.size(0), -1))
            self.debug("output", output)
        else:
            output = h5
        logger.debug("Done.")
        return output

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Networks.register
class BGEncoder(nn.Module):
    """ This is the encoder part of the BGGAN.
    """
    def __init__(self, in_shape, in_channels=1, start_filts=64,
                 latent_dim=1000):
        """ Init class.

        Parameters
        ----------
        in_shape: uplet
            the input tensor data shape (X, Y, Z).
        in_channels: int, default 1
            number of channels in the input tensor.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        latent_dim: int, default 1000
            the latent variable sizes.
        """
        super(BGEncoder, self).__init__()
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.latent_dim = latent_dim
        self.in_shape = in_shape
        self.shapes = _downsample_shape(
            self.in_shape, nb_iterations=4, scale_factor=2)
        self.dense_features = np.prod(self.shapes[-1])
        logger.debug("BGGAN Encoder shapes: {0}".format(self.shapes))
        self.conv1 = nn.Conv3d(
            self.in_channels, self.start_filts, kernel_size=4, stride=2,
            padding=1)
        self.conv2 = nn.Conv3d(
            self.start_filts, self.start_filts * 2, kernel_size=4, stride=2,
            padding=1)
        self.bn2 = nn.BatchNorm3d(self.start_filts * 2)
        self.conv3 = nn.Conv3d(
            self.start_filts * 2, self.start_filts * 4, kernel_size=4,
            stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(self.start_filts * 4)
        self.conv4 = nn.Conv3d(
            self.start_filts * 4, self.start_filts * 8, kernel_size=4,
            stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(self.start_filts * 8)
        self.mean = nn.Sequential(
            nn.Linear(self.start_filts * 8 * self.dense_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_dim))
        self.logvar = nn.Sequential(
            nn.Linear(self.start_filts * 8 * self.dense_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_dim))

    def forward(self, x):
        logger.debug("BGGAN Encoder...")
        batch_size = x.size(0)
        logger.debug("  batch_size: {0}".format(batch_size))
        self.debug("input", x)
        h1 = func.leaky_relu(self.conv1(x), negative_slope=0.2)
        self.debug("conv1", h1)
        h2 = func.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        self.debug("conv2", h2)
        h3 = func.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        self.debug("conv3", h3)
        h4 = func.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        self.debug("conv4", h4)
        mean = self.mean(h4.view(batch_size, -1))
        self.debug("mean", mean)
        logvar = self.logvar(h4.view(batch_size, -1))
        self.debug("logvar", logvar)
        std = logvar.mul(0.5).exp_()
        reparametrized_noise = Variable(
            torch.randn((batch_size, self.latent_dim))).to(x.device)
        reparametrized_noise = mean + std * reparametrized_noise
        self.debug("reparametrization", reparametrized_noise)
        logger.debug("Done.")
        return mean, logvar, reparametrized_noise

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Networks.register
class BGCodeDiscriminator(nn.Module):
    """ This is the code discriminator part of the BGGAN.
    """
    def __init__(self, out_channels=1, code_size=1000, n_units=4096):
        """ Init class.

        Parameters
        ----------
        out_channels: int, default 1
            number of channels in the output tensor.
        code_size: int, default 1000
            the code sier.
        n_units: int, default 4096
            the number of hidden units.
        """
        super(BGCodeDiscriminator, self).__init__()
        self.out_channels = out_channels
        self.code_size = code_size
        self.n_units = n_units
        self.layer1 = nn.Sequential(
            nn.Linear(self.code_size, self.n_units),
            nn.BatchNorm1d(self.n_units),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Linear(self.n_units, self.n_units),
            nn.BatchNorm1d(self.n_units),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Linear(self.n_units, self.out_channels)

    def forward(self, x):
        logger.debug("BGGAN Code Discriminator...")
        self.debug("input", x)
        h1 = self.layer1(x)
        self.debug("layer1", h1)
        h2 = self.layer2(h1)
        self.debug("layer2", h2)
        output = self.layer3(h2)
        self.debug("layer3", output)
        logger.debug("Done.")
        return output

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Networks.register
class BGGenerator(nn.Module):
    """ This is the generator part of the BGGAN.
    """
    def __init__(self, in_shape, out_channels=1, start_filts=64,
                 latent_dim=1000, mode="trilinear", with_code=False):
        """ Init class.

        Parameters
        ----------
        in_shape: uplet
            the input tensor data shape (X, Y, Z).
        out_channels: int, default 1
            number of channels in the output tensor.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        latent_dim: int, default 1000
            the latent variable sizes.
        mode: str, default 'trilinear'
            the interpolation mode.
        with_code: bool, default False
            change the architecture if code discriminator is used.
        """
        super(BGGenerator, self).__init__()
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.latent_dim = latent_dim
        self.in_shape = in_shape
        self.mode = mode
        self.with_code = with_code
        self.shapes = _downsample_shape(
            self.in_shape, nb_iterations=4, scale_factor=2)
        self.dense_features = np.prod(self.shapes[-1])
        logger.debug("BGGAN Generator shapes: {0}".format(self.shapes))
        if self.with_code:
            self.tp_conv1 = nn.ConvTranspose3d(
                self.latent_dim, self.start_filts * 8, kernel_size=4,
                stride=1, padding=0, bias=False)
        else:
            self.fc = nn.Linear(
                self.latent_dim, self.start_filts * 8 * self.dense_features)
        self.bn1 = nn.BatchNorm3d(self.start_filts * 8)

        self.tp_conv2 = nn.Conv3d(
            self.start_filts * 8, self.start_filts * 4, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.start_filts * 4)

        self.tp_conv3 = nn.Conv3d(
            self.start_filts * 4, self.start_filts * 2, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.start_filts * 2)

        self.tp_conv4 = nn.Conv3d(
            self.start_filts * 2, self.start_filts, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(self.start_filts)

        self.tp_conv5 = nn.Conv3d(
            self.start_filts, self.out_channels, kernel_size=3, stride=1,
            padding=1, bias=False)

    def forward(self, noise):
        logger.debug("BGGAN Generator...")
        self.debug("input", noise)
        if self.with_code:
            noise = noise.view(-1, self.latent_dim, 1, 1, 1)
            self.debug("view", noise)
            h = self.tp_conv1(noise)
            self.debug("tp_conv1", h)
        else:
            noise = noise.view(-1, self.latent_dim)
            self.debug("view", noise)
            h = self.fc(noise)
            self.debug("dense", h)
            h = h.view(-1, self.start_filts * 8, *self.shapes[-1])
            self.debug("view", h)
        h = func.relu(self.bn1(h))

        h = nn.functional.interpolate(
            h, size=self.shapes[-2], mode=self.mode, align_corners=False)
        h = self.tp_conv2(h)
        h = func.relu(self.bn2(h))
        self.debug("tp_conv2", h)

        h = nn.functional.interpolate(
            h, size=self.shapes[-3], mode=self.mode, align_corners=False)
        h = self.tp_conv3(h)
        h = func.relu(self.bn3(h))
        self.debug("tp_conv3", h)

        h = nn.functional.interpolate(
            h, size=self.shapes[-4], mode=self.mode, align_corners=False)
        h = self.tp_conv4(h)
        h = func.relu(self.bn4(h))
        self.debug("tp_conv4", h)

        h = nn.functional.interpolate(
            h, size=self.shapes[-5], mode=self.mode, align_corners=False)
        h = self.tp_conv5(h)
        self.debug("tp_conv5", h)

        h = torch.tanh(h)
        self.debug("output", h)
        logger.debug("Done.")
        return h

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


def _downsample_shape(shape, nb_iterations=1, scale_factor=2):
    shape = np.asarray(shape)
    all_shapes = [shape.astype(int).tolist()]
    for idx in range(nb_iterations):
        shape = np.floor(shape / scale_factor)
        all_shapes.append(shape.astype(int).tolist())
    return all_shapes
