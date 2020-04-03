# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
BrainNetCNNs are convolutional neural networks for connectomes.
"""

# Imports
import collections
import torch
import torch.nn.functional as F
import torch.nn as nn
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks


@Networks.register
@DeepLearningDecorator(family="graph")
class BrainNetCNN(nn.Module):
    """ BrainNetCNN.

    BrainNetCNN is composed of novel edge-to-edge, edge-to-node and
    node-to-graph convolutional filters that leverage thetopological
    locality of structural brain networks.

    Reference: https://www2.cs.sfu.ca/~hamarneh/ecopy/neuroimage2017.pdf.
    Code: https://github.com/nicofarr/brainnetcnnVis_pytorch.
    """
    def __init__(self, input_shape, in_channels, num_classes):
        """ Init class.

        Parameters
        ----------
        input_shape: tuple
            the data shape.
        in_channels: int
            number of channels in the input tensor.
        num_classes: int
            the number of classes to be predicted.
        """
        # Inheritance
        nn.Module.__init__(self)

        # Class parameters
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.kernel_height, self.kernel_width = input_shape

        # Define layers
        self.cnn_layers = nn.Sequential(collections.OrderedDict([
            ("e2e1", E2EBlock(input_shape, in_channels, 32, bias=True)),
            ("relu1", nn.LeakyReLU(negative_slope=0.33)),
            ("e2e2", E2EBlock(input_shape, 32, 64, bias=True)),
            ("relu2", nn.LeakyReLU(negative_slope=0.33)),
            ("e2n", torch.nn.Conv2d(64, 1, (1, self.kernel_width))),
            ("relu3", nn.LeakyReLU(negative_slope=0.33)),
            ("n2g", torch.nn.Conv2d(1, 256, (self.kernel_height, 1))),
            ("relu4", nn.LeakyReLU(negative_slope=0.33)),
            ("dropout1", nn.Dropout(0.5))
        ]))
        self.dense_layers = nn.Sequential(collections.OrderedDict([
            ("dense1", torch.nn.Linear(256, 128)),
            ("relu1", nn.LeakyReLU(negative_slope=0.33)),
            ("dropout1", nn.Dropout(0.5)),
            ("dense2", torch.nn.Linear(128, 30)),
            ("relu2", nn.LeakyReLU(negative_slope=0.33)),
            ("dropout2", nn.Dropout(0.5)),
            ("dense3", torch.nn.Linear(30, self.num_classes)),
            ("relu3", nn.LeakyReLU(negative_slope=0.33)),
        ]))

    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


class E2EBlock(nn.Module):
    """ Implementation of the Edge to Edge filter.
    """
    def __init__(self, input_shape, in_planes, out_planes, bias=False):
        super(E2EBlock, self).__init__()
        self.kernel_height, self.kernel_width = input_shape
        self.cnn1 = torch.nn.Conv2d(
            in_planes, out_planes, (1, self.kernel_width), bias=bias)
        self.cnn2 = torch.nn.Conv2d(
            in_planes, out_planes, (self.kernel_height, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return (torch.cat([a] * self.kernel_width, 3) +
                torch.cat([b] * self.kernel_height, 2))
