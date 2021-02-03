# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
The Residual Auto-Endocer network (ResAENet).
"""

# Imports
import logging
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import torch
import torch.nn as nn
import torch.nn.functional as func
from functools import partialmethod
import numpy as np

# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("encoder", ))
class ResAENet(nn.Module):
    """ Restidual Auto-Encoder Network.

    Reference: Discovering Functional Brain Networks with 3D Residual
    Autoencoder (ResAE), MICCAI 2020.
    """
    def __init__(self,
                 input_shape,
                 cardinality=1,
                 layers=[3, 4, 6, 3],
                 n_channels_in=1,
                 decode=True):
        """ Initilaize class.

        Parameters
        ----------
        input_shape: uplet
            the input tensor data shape (X, Y, Z).
        cardinality: int, default 1
            control the numbber of paths (ResNeXt architecture).
        layers: 4-uplet, default [3, 4, 6, 3]
            the layers blocks definition.
        n_channels_in: int, default 1
            the number of input channels.
        decode: bool, default True
            if set apply decoding.
        """
        # Parameters
        logger.debug("ResAENet init...")
        super().__init__()
        if len(layers) != 4:
            raise ValueError("The model was designed for 4 layers only!")
        if cardinality == 1:
            logger.debug("- using resnet block.")
            block = ResNetBottleneck
        else:
            logger.debug("- using resnext block with {0} paths.".format(
                cardinality))
            block = partialclass(
                ResNeXtBottleneck, cardinality=cardinality)
        self.shapes = self._downsample_shape(
            input_shape, nb_iterations=len(layers))
        logger.debug("- shapes: {0}.".format(self.shapes))
        self.in_planes = 32
        self.decode = decode

        # First conv
        self.conv1 = nn.Conv3d(
            n_channels_in, self.in_planes, kernel_size=(7, 7, 7),
            stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Encoder
        self.enc_layer1 = self._make_layer(block, 32, layers[0])
        self.enc_layer2 = self._make_layer(block, 32, layers[1])
        self.enc_layer3 = self._make_layer(block, 32, layers[2])
        self.enc_layer4 = self._make_layer(block, 32, layers[3])
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Decoder
        self.dec_layer1 = self._make_layer(block, 32, layers[3])
        self.dec_layer2 = self._make_layer(block, 32, layers[2])
        self.dec_layer3 = self._make_layer(block, 32, layers[1])
        self.dec_layer4 = self._make_layer(block, 32, layers[0])
        self.upsample1 = nn.Upsample(mode="nearest", size=self.shapes[-2])
        self.upsample2 = nn.Upsample(mode="nearest", size=self.shapes[-3])
        self.upsample3 = nn.Upsample(mode="nearest", size=self.shapes[-4])

        # Final conv
        self.conv_final = nn.Conv3d(
            32, n_channels_in, kernel_size=(7, 7, 7), stride=1, bias=False)
        self.bn_final = nn.BatchNorm3d(n_channels_in)
        self.upsample_final = nn.Upsample(mode="nearest", size=self.shapes[-5])

        # Kernel initializer
        self.kernel_initializer()

    def _downsample_shape(self, shape, nb_iterations=1, scale_factor=2):
        shape = np.asarray(shape)
        all_shapes = [shape.astype(int).tolist()]
        for idx in range(nb_iterations):
            shape = np.floor((shape + 1) / scale_factor)
            all_shapes.append(shape.astype(int).tolist())
        return all_shapes

    def kernel_initializer(self):
        for module in self.modules():
            self.init_weight(module)

    @staticmethod
    def init_weight(module):
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, n_blocks, stride=1):
        downsample = nn.Sequential(
            conv1x1x1(self.in_planes, planes, stride),
            nn.BatchNorm3d(planes))
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  out_planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes
        for i in range(1, n_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        logger.debug("ResAENet...")
        debug("input", x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        debug("{0} + bn + relu".format(self.conv1), x)
        logger.debug("Encorder block 1:")
        x = self.enc_layer1(x)
        x = self.maxpool1(x)
        debug(repr(self.maxpool1), x)
        logger.debug("Encorder block 2:")
        x = self.enc_layer2(x)
        x = self.maxpool2(x)
        debug(repr(self.maxpool2), x)
        logger.debug("Encorder block 3:")
        x = self.enc_layer3(x)
        x = self.maxpool3(x)
        debug(repr(self.maxpool3), x)
        logger.debug("Encorder block 4:")
        x = self.enc_layer4(x)

        if self.decode:
            logger.debug("Decoder block 1:")
            x = self.dec_layer1(x)
            x = self.upsample1(x)
            debug("upsample1", x)
            logger.debug("Decoder block 2:")
            x = self.dec_layer2(x)
            x = self.upsample2(x)
            debug("upsample2", x)
            logger.debug("Decoder block 3:")
            x = self.dec_layer3(x)
            x = self.upsample3(x)
            debug("upsample3", x)
            logger.debug("Decoder block 4:")
            x = self.dec_layer4(x)

            x = self.conv_final(x)
            x = self.bn_final(x)
            x = self.relu(x)
            debug("{0} + bn + relu".format(self.conv_final), x)
            x = self.upsample_final(x)
            debug("last upsample", x)
        else:
            x = self.avgpool(x)
            debug("avgpool", x)

        return x


def partialclass(cls, *args, **kwargs):
    """ Return a new partial class object which when initialized will behave
    like cls.__init__ called with the positional arguments args and kwargs.
    In other words it 'freezes' some portion of the init arguments and/or
    kwargs resulting in a new class with a simplified init signature.
    """
    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return PartialClass


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


def conv3x3x3(in_planes, out_planes, stride=1):
    """ 3d convolution with a fix 3x3x3 kernel.
    """
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    """ 3d convolution with a fix 1x1x1 kernel.
    """
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class ResNetBottleneck(nn.Module):
    """ Residual block definition as defined in "Deep Residual Learning for
    Image Recognition" (https://arxiv.org/pdf/1512.03385.pdf).
    """
    __name__ = "ResNetBottleneck"

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        """ Initilaize class.

        Parameters
        ----------
        in_planes: int
            the umber of input channels.
        out_planes: int
            the number of output channels.
        stride: int, default 1
            the convolution stride.
        downsample: @callable, default None
            if set downsample the input for the 'identity shortcut connection'.
        """
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, out_planes)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = conv3x3x3(out_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.conv3 = conv3x3x3(out_planes, out_planes, stride)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.3)
        self.downsample = downsample

    def forward(self, x):
        logger.debug("{0}...".format(self.__name__))
        debug("x", x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        debug("{0} + bn + relu".format(self.conv1), out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        debug("{0} + bn + relu".format(self.conv2), out)
        out = self.conv3(out)
        out = self.bn3(out)
        debug("{0} + bn".format(self.conv3), out)
        if self.downsample is not None:
            residual = self.downsample(x)
            debug("downsample: {0} + bn".format(self.downsample[0]), residual)
        out += residual
        out = self.relu(out)
        debug("add + relu", out)
        return out


class ResNeXtBottleneck(ResNetBottleneck):
    """ Residual block definition as defined in "Aggregated Residual
    Transformations for Deep Neural Networks"
    (https://arxiv.org/pdf/1611.05431.pdf).
    """
    __name__ = "ResNeXtBottleneck"

    def __init__(self, in_planes, out_planes, cardinality, stride=1,
                 downsample=None):
        """ Initilaize class.

        Parameters
        ----------
        in_planes: int
            the umber of input channels.
        out_planes: int
            the number of output channels.
        cardinality: int
            the number of independent paths (adjust the model capacity).
        stride: int, default 1
            the convolution stride.
        downsample: @callable, default None
            if set downsample the input for the 'identity shortcut connection'.
        """
        super().__init__(in_planes, out_planes, stride, downsample)
        # groups controls the connections between inputs and outputs.
        # in_channels and out_channels must both be divisible by groups.
        mid_planes = cardinality * out_planes
        self.conv1 = conv1x1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes, mid_planes, kernel_size=3, stride=stride,
            padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, out_planes)
