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
    def __init__(self, input_shape, in_channels, num_classes, nb_e2e=32,
                 nb_e2n=64, nb_n2g=30, dropout=0.5, leaky_alpha=0.33,
                 twice_e2e=False, dense_sml=True):
        """ Init class.

        Parameters
        ----------
        input_shape: tuple
            the size of the functional connectivity matrix.
        in_channels: int
            number of channels in the input tensor.
        num_classes: int
            the number of classes to be predicted.
        twice_e2e: bool, default False
            if set use two E2E filter twice.
        nb_e2e: int, default 32
            number of e2e filters.
        nb_e2n: int, default 64
            number of e2n filters.
        nb_n2g: int, default 30
            number of n2g filters.
        dropout: float, default 0.5
            the dropout rate.
        leaky_alpha: float, default 0.33
            leaky ReLU alpha rate.
        twice_e2e: bool, default False
            if set apply two times the Edge-to-Edge layer.
        dense_sml: bool, default True
            if set reduce the number of hidden dense layers otherwise set
            nb_n2g to 256.
        """
        # Inheritance
        nn.Module.__init__(self)

        # Class parameters
        self.num_classes = num_classes
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
                ("e2e1", Edge2Edge(input_shape, in_channels, nb_e2e)),
                ("relu1", nn.LeakyReLU(negative_slope=leaky_alpha)),
                ("e2e2", Edge2Edge(input_shape, nb_e2e, nb_e2e)),
                ("relu2", nn.LeakyReLU(negative_slope=leaky_alpha))
            ]))
        else:
            self.e2e = nn.Sequential(collections.OrderedDict([
                ("e2e", Edge2Edge(input_shape, in_channels, nb_e2e)),
                ("relu", nn.LeakyReLU(negative_slope=leaky_alpha)),
            ]))
        self.e2n = nn.Sequential(collections.OrderedDict([
            ("e2n", Edge2Node(input_shape, nb_e2e, nb_e2n)),
            ("relu", nn.LeakyReLU(negative_slope=leaky_alpha)),
            ("dropout", nn.Dropout(dropout))
        ]))
        self.n2g = nn.Sequential(collections.OrderedDict([
            ("n2g", Node2Graph(input_shape, nb_e2n, nb_n2g)),
            ("relu", nn.LeakyReLU(negative_slope=leaky_alpha)),
        ]))
        if self.dense_sml:
            self.dense_layers = nn.Sequential(collections.OrderedDict([
                ("dense", torch.nn.Linear(nb_n2g, self.num_classes))
            ]))
        else:
            self.dense_layers = nn.Sequential(collections.OrderedDict([
                ("dense1", torch.nn.Linear(nb_n2g, 128)),
                ("dropout1", nn.Dropout(dropout)),
                ("relu1", nn.LeakyReLU(negative_slope=leaky_alpha)),
                ("dense2", torch.nn.Linear(128, 30)),
                ("dropout2", nn.Dropout(dropout)),
                ("relu2", nn.LeakyReLU(negative_slope=leaky_alpha)),
                ("dense3", torch.nn.Linear(30, self.num_classes))
            ]))

        # Init weights
        @torch.no_grad()
        def weights_init(module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
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


class Edge2Edge(nn.Module):
    """ Implementation of the Edge-to-Edge (e2e) layer.

    The E2E filter is defined in terms of topological locality, by combining
    the weights of edges that share nodes together.
    """
    def __init__(self, input_shape, channels, filters):
        """ Init class.

        Parameters
        ----------
        input_shape: int
            the size of the functional connectivity matrix.
        channels: int
            number of input channel.
        filters: int
            number of output channel
        """
        super(Edge2Edge, self).__init__()
        self.kernel_height, self.kernel_width = input_shape
        self.row_conv = nn.Conv2d(channels, filters, (1, self.kernel_width))
        self.col_conv = nn.Conv2d(channels, filters, (self.kernel_height, 1))

    def forward(self, x):
        """ e2e by two conv2d with line filter.
        """
        logger.debug("E2E layer...")
        logger.debug("  input: {0} - {1} - {2}".format(
            x.shape, x.get_device(), x.dtype))
        row = self.row_conv(x)
        logger.debug("  row: {0} - {1} - {2}".format(
            row.shape, row.get_device(), row.dtype))
        col = self.col_conv(x)
        logger.debug("  col: {0} - {1} - {2}".format(
            col.shape, col.get_device(), col.dtype))
        return (torch.cat([col] * self.kernel_width, dim=2) +
                torch.cat([row] * self.kernel_height, dim=3))


class Edge2Node(nn.Module):
    """ Implementation of the Edge-to-Node (e2n) layer.
    """
    def __init__(self, input_shape, channels, filters):
        """ Init class.

        Parameters
        ----------
        input_shape: int
            the size of the functional connectivity matrix.
        channels: int
            number of input channel.
        filters: int
            number of output channel
        """
        super(Edge2Node, self).__init__()
        self.kernel_height, self.kernel_width = input_shape
        self.row_conv = nn.Conv2d(channels, filters, (1, self.kernel_width))
        self.col_conv = nn.Conv2d(channels, filters, (self.kernel_height, 1))

    def forward(self, x):
        """ e2n by add two conv2d.
        """
        row = self.row_conv(x)
        col = self.col_conv(x)
        return row + col.permute(0, 1, 3, 2)


class Node2Graph(nn.Module):
    """ Implementation of the Node-to-Graph (n2g) layer.
    """
    def __init__(self, input_shape, channels, filters):
        """ Init class.

        Parameters
        ----------
        input_shape: int
            the size of the functional connectivity matrix.
        channels: int
            number of input channel.
        filters: int
            number of output channel
        """
        super(Node2Graph, self).__init__()
        self.kernel_height, self.kernel_width = input_shape
        self.conv = nn.Conv2d(channels, filters, (self.kernel_height, 1))

    def forward(self, x):
        """ n2g by convolution.
        """
        return self.conv(x)
