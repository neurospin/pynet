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
import logging
import collections
import torch
import torch.nn.functional as F
import torch.nn as nn
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family="graph")
class BrainNetCNN(nn.Module):
    """ BrainNetCNN.

    BrainNetCNN is composed of novel edge-to-edge, edge-to-node and
    node-to-graph convolutional filters (layers) that leverage the topological
    locality of structural brain networks.

    An E2E filter computes a weighted sum of edge weights over all edges
    connected either to node i or j, like a convolution.
    An E2N filter summarizes the responses of neighbouring edges into a set
    of node responses.
    A N2G filter can be interpreted as getting a single response from all the
    nodes in the graph.

    BrainNetCNN is able to predict an infant's age with an average error of
    about 2 weeks, demonstrating that it can learn relevant topological
    features from the  connectome  data. BrainNetCNN can also be applied to
    the much more challenging task of predicting neurodevelopmental scores.

    Reference: https://www2.cs.sfu.ca/~hamarneh/ecopy/neuroimage2017.pdf.
    Code: https://github.com/nicofarr/brainnetcnnVis_pytorch.
    """
    def __init__(self, input_shape, in_channels, num_classes, twice_e2e=False,
                 dense_sml=True):
        """ Init class.

        Parameters
        ----------
        input_shape: tuple
            the data shape.
        in_channels: int
            number of channels in the input tensor.
        num_classes: int
            the number of classes to be predicted.
        twice_e2e: bool, default False
            if set use two E2E filter twice.
        dense_sml: bool, default True
            if set reduce the number of hidden dense layers.
        """
        # Inheritance
        nn.Module.__init__(self)

        # Class parameters
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.kernel_height, self.kernel_width = input_shape
        self.twice_e2e = twice_e2e
        self.dense_sml = dense_sml

        # The brain network adjacency matrix is first convolved with one or
        # more (two  in  this  case) E2E filters which  weight edges of
        # adjacent brain regions. The response is convolved with an E2N filter
        # which assigns each brain region a weighted sum of its edges. The N2G
        # assigns a single response based on all the weighted nodes. Finally,
        # fully connected (FC) layers reduce the number of features down to
        # N output score predictions.
        # The dense2 layer output can be interpreted as a set of high-level
        # features learned by the previous layers.
        if self.twice_e2e:
            self.e2e = nn.Sequential(collections.OrderedDict([
                ("e2e1", E2EBlock(input_shape, in_channels, 32, bias=True)),
                ("e2e2", E2EBlock(input_shape, 32, 32, bias=True))
            ]))
        else:
            self.e2e = nn.Sequential(collections.OrderedDict([
                ("e2e", E2EBlock(input_shape, in_channels, 32, bias=True)),
            ]))
        self.e2n = nn.Sequential(collections.OrderedDict([
            ("e2n", torch.nn.Conv2d(32, 64, (1, self.kernel_width), stride=1)),
            ("dropout", nn.Dropout(0.5)),
            ("relu", nn.LeakyReLU(negative_slope=0.33))
        ]))
        if self.dense_sml:
            self.n2g = nn.Sequential(collections.OrderedDict([
                ("n2g", torch.nn.Conv2d(64, 30, (self.kernel_height, 1))),
                ("relu", nn.LeakyReLU(negative_slope=0.33)),
            ]))
            self.dense_layers = nn.Sequential(collections.OrderedDict([
                ("dense", torch.nn.Linear(30, self.num_classes))
            ]))
        else:
            self.n2g = nn.Sequential(collections.OrderedDict([
                ("n2g", torch.nn.Conv2d(64, 256, (self.kernel_height, 1))),
                ("relu", nn.LeakyReLU(negative_slope=0.33)),
            ]))
            self.dense_layers = nn.Sequential(collections.OrderedDict([
                ("dense1", torch.nn.Linear(256, 128)),
                ("dropout1", nn.Dropout(0.5)),
                ("relu1", nn.LeakyReLU(negative_slope=0.33)),
                ("dense2", torch.nn.Linear(128, 30)),
                ("dropout2", nn.Dropout(0.5)),
                ("relu2", nn.LeakyReLU(negative_slope=0.33)),
                ("dense3", torch.nn.Linear(30, self.num_classes))
            ]))

        # Init weights
        @torch.no_grad()
        def weights_init(module):
            if isinstance(module, nn.Conv2d):
                logger.debug("Init weights of {0}...".format(module))
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant(module.bias, 0)
        self.apply(weights_init)

    def forward(self, x):
        logger.debug("BrainNetCNN layer...")
        logger.debug("  input: {0} - {1} - {2}".format(
            x.shape, x.get_device(), x.dtype))
        out = self.e2e(x)
        logger.debug("  e2e: {0} - {1} - {2}".format(
            out.shape, out.get_device(), out.dtype))
        out = self.e2n(out)
        logger.debug("  e2n: {0} - {1} - {2}".format(
            out.shape, out.get_device(), out.dtype))
        out = self.n2g(out)
        logger.debug("  n2g: {0} - {1} - {2}".format(
            out.shape, out.get_device(), out.dtype))
        out = out.view(out.size(0), -1)
        logger.debug("  view: {0} - {1} - {2}".format(
            out.shape, out.get_device(), out.dtype))
        out = self.dense_layers(out)
        logger.debug("  dense: {0} - {1} - {2}".format(
            out.shape, out.get_device(), out.dtype))
        return out


class E2EBlock(nn.Module):
    """ Implementation of the Edge-to-Edge filter.

    The E2E filter is defined in terms of topological locality, by combining
    the weights of edges that share nodes together.
    """
    def __init__(self, input_shape, in_planes, out_planes, bias=False):
        super(E2EBlock, self).__init__()
        self.kernel_height, self.kernel_width = input_shape
        self.conv_1xd = torch.nn.Conv2d(
            in_planes, out_planes, (1, self.kernel_width), bias=bias,
            stride=1)
        self.conv_dx1 = torch.nn.Conv2d(
            in_planes, out_planes, (self.kernel_height, 1), bias=bias,
            stride=1)

    def forward(self, x):
        logger.debug("E2E layer...")
        logger.debug("  input: {0} - {1} - {2}".format(
            x.shape, x.get_device(), x.dtype))
        conv_1xd = self.conv_1xd(x)
        logger.debug("  1xd: {0} - {1} - {2}".format(
            conv_1xd.shape, conv_1xd.get_device(), conv_1xd.dtype))
        conv_dx1 = self.conv_dx1(x)
        logger.debug("  dx1: {0} - {1} - {2}".format(
            conv_dx1.shape, conv_dx1.get_device(), conv_dx1.dtype))
        return (torch.cat([conv_dx1] * self.kernel_width, dim=2) +
                torch.cat([conv_1xd] * self.kernel_height, dim=3))
