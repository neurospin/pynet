# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides spherical layers.
"""

# Imports
import logging
import collections
import torch
import torch.nn as nn
import numpy as np
from .utils import debug


# Global parameters
logger = logging.getLogger("pynet")


class RePaIcoConvLayer(nn.Module):
    """ Define the convolutional layer on icosahedron discretized sphere using
    rectagular filter in tangent plane.
    """
    def __init__(self, in_feats, out_feats, neighs):
        """ Init.

        Parameters
        ----------
        in_feats: int
            input features/channels.
        out_feats: int
            output features/channels.
        neighs: 2-uplet
            neigh_indices: array (N, k, 3) - the neighbors indices.
            neigh_weights: array (N, k, 3) - the neighbors distances.
        """
        super(RePaIcoConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_indices, self.neigh_weights = neighs
        self.n_vertices, self.neigh_size, _ = self.neigh_indices.shape
        self.neigh_indices = self.neigh_indices.reshape(self.n_vertices, -1)
        self.neigh_weights = torch.from_numpy(
            self.neigh_weights.reshape(self.n_vertices, -1).astype(np.float32))
        self.weight = nn.Linear(self.neigh_size * in_feats, out_feats)

    def forward(self, x):
        logger.debug("RePaIcoConvLayer...")
        device = x.get_device()
        if self.neigh_weights.get_device() != device:
            self.neigh_weights = self.neigh_weights.to(device)
        debug("input", x)
        logger.debug(" weight: {0}".format(self.weight))
        logger.debug(" neighbors indices: {0}".format(
            self.neigh_indices.shape))
        logger.debug(" neighbors weights: {0}".format(
            self.neigh_weights.shape))
        n_samples = len(x)
        mat = x[:, :, self.neigh_indices.reshape(-1)].view(
            n_samples, self.in_feats, self.n_vertices, self.neigh_size * 3)
        debug("neighors", mat)
        x = torch.mul(mat, self.neigh_weights).view(
            n_samples, self.in_feats, self.n_vertices, self.neigh_size, 3)
        debug("weighted neighors", x)
        x = torch.sum(x, dim=4)
        debug("sum", x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(n_samples * self.n_vertices,
                      self.in_feats * self.neigh_size)
        out = self.weight(x)
        out = out.view(n_samples, self.n_vertices, self.out_feats)
        out = out.permute(0, 2, 1)
        debug("output", out)
        return out


class DiNeIcoConvLayer(nn.Module):
    """ The convolutional layer on icosahedron discretized sphere using
    n-ring filter (based on the Direct Neighbor (DiNe) formulation).
    """
    def __init__(self, in_feats, out_feats, neigh_indices, n_ring=1):
        """ Init.

        Parameters
        ----------
        in_feats: int
            input features/channels.
        out_feats: int
            output features/channels.
        neigh_indices: array (N, k)
            conv layer's filters' neighborhood indices, where N is the ico
            number of vertices and k the considered nodes neighbors.
        """
        super(DiNeIcoConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_indices = neigh_indices
        self.n_vertices, self.neigh_size = neigh_indices.shape
        self.weight = nn.Linear(self.neigh_size * in_feats, out_feats)

    def forward(self, x):
        logger.debug("DiNeIcoConvLayer...")
        debug("input", x)
        logger.debug(" weight: {0}".format(self.weight))
        logger.debug(" neighbors indices: {0}".format(
            self.neigh_indices.shape))
        mat = x[:, :, self.neigh_indices.reshape(-1)].view(
            len(x), self.in_feats, self.n_vertices, self.neigh_size)
        mat = mat.permute(0, 2, 1, 3)
        mat = mat.reshape(len(x) * self.n_vertices,
                          self.in_feats * self.neigh_size)
        debug("neighors", mat)
        out_features = self.weight(mat)
        out_features = out_features.view(len(x), self.n_vertices,
                                         self.out_feats)
        out_features = out_features.permute(0, 2, 1)
        debug("output", out_features)
        return out_features


class IcoPoolLayer(nn.Module):
    """ The pooling layer on icosahedron discretized sphere using
    1-ring filter.
    """
    def __init__(self, down_neigh_indices, down_indices, pooling_type="mean"):
        """ Init.

        Parameters
        ----------
        down_neigh_indices: array
            downsampling neighborhood indices at sampling i + 1.
        down_indices: array
            downsampling indices at sampling i.
        pooling_type: str, default 'mean'
            the pooling type: 'mean' or 'max'.
        """
        super(IcoPoolLayer, self).__init__()
        self.down_indices = down_indices
        self.down_neigh_indices = down_neigh_indices[down_indices]
        self.n_vertices, self.neigh_size = self.down_neigh_indices.shape
        self.pooling_type = pooling_type

    def forward(self, x):
        logger.debug("PoolLayer...")
        debug("input", x)
        n_vertices = int((x.size(2) + 6) / 4)
        assert self.n_vertices == n_vertices
        n_features = x.size(1)
        logger.debug(" down neighbors indices: {0}".format(
            self.down_neigh_indices.shape))
        x = x[:, :, self.down_neigh_indices.reshape(-1)].view(
            len(x), n_features, n_vertices, self.neigh_size)
        debug("neighors", x)
        if self.pooling_type == "mean":
            x = torch.mean(x, dim=-1)
            max_pool_indices = None
        elif self.pooling_type == "max":
            x, max_pool_indices = torch.max(x, dim=-1)
            debug("max pool indices", max_pool_indices)
        else:
            raise RuntimeError("Invalid pooling.")
        debug("pool", x)
        return x, max_pool_indices


class IcoUpConvLayer(nn.Module):
    """ The transposed convolution layer on icosahedron discretized sphere
    using 1-ring filter.
    """
    def __init__(self, in_feats, out_feats, up_neigh_indices, down_indices):
        """ Init.

        Parameters
        ----------
        in_feats: int
            input features/channels.
        out_feats: int
            output features/channels.
        up_neigh_indices: array
            upsampling neighborhood indices at sampling i + 1.
        down_indices: array
            downsampling indices at sampling i
        """
        super(IcoUpConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.up_neigh_indices = up_neigh_indices
        self.neigh_indices = up_neigh_indices[down_indices]
        self.down_indices = down_indices
        self.n_vertices, self.neigh_size = self.up_neigh_indices.shape

        self.flat_neigh_indices = self.neigh_indices.reshape(-1)
        self.argsort_neigh_indices = np.argsort(self.flat_neigh_indices)
        self.sorted_neigh_indices = self.flat_neigh_indices[
            self.argsort_neigh_indices]
        assert(np.unique(self.sorted_neigh_indices).tolist() ==
               list(range(self.n_vertices)))

        self.sorted_2occ_12neigh_indices = self.sorted_neigh_indices[:24]
        self._check_occurence(self.sorted_2occ_12neigh_indices, occ=2)
        self.sorted_1occ_neigh_indices = self.sorted_neigh_indices[
            24: len(down_indices) + 12]
        self._check_occurence(self.sorted_1occ_neigh_indices, occ=1)
        self.sorted_2occ_neigh_indices = self.sorted_neigh_indices[
            len(down_indices) + 12:]
        self._check_occurence(self.sorted_2occ_neigh_indices, occ=2)
        self.argsort_2occ_12neigh_indices = self.argsort_neigh_indices[:24]
        self.argsort_1occ_neigh_indices = self.argsort_neigh_indices[
            24: len(down_indices) + 12]
        self.argsort_2occ_neigh_indices = self.argsort_neigh_indices[
            len(down_indices) + 12:]

        self.weight = nn.Linear(in_feats, self.neigh_size * out_feats)

    def _check_occurence(self, data, occ):
        count = collections.Counter(data)
        unique_count = np.unique(list(count.values()))
        assert len(unique_count) == 1
        assert unique_count[0] == occ

    def forward(self, x):
        logger.debug("UpSampleLayer: transpose conv...")
        debug("input", x)
        n_samples, n_feats, n_vertices = x.size()
        logger.debug(" weight: {0}".format(self.weight))
        logger.debug(" neighbors indices: {0}".format(
            self.neigh_indices.shape))
        x = x.permute(0, 2, 1)
        x = x.reshape(n_samples * n_vertices, n_feats)
        debug("input", x)
        x = self.weight(x)
        debug("weighted input", x)
        x = x.view(n_samples, n_vertices, self.neigh_size, self.out_feats)
        debug("weighted input", x)
        x = x.view(n_samples, n_vertices * self.neigh_size, self.out_feats)
        x1 = x[:, self.argsort_2occ_12neigh_indices]
        x1 = x1.view(n_samples, 12, 2, self.out_feats)
        debug("12 first 2 occ output", x1)
        x2 = x[:, self.argsort_1occ_neigh_indices]
        debug("1 occ output", x2)
        x3 = x[:, self.argsort_2occ_neigh_indices]
        x3 = x3.view(n_samples, -1, 2, self.out_feats)
        debug("2 occ output", x3)
        x = torch.cat(
            (torch.mean(x1, dim=2), x2, torch.mean(x3, dim=2)), dim=1)
        x = x.permute(0, 2, 1)
        debug("output", x)
        return x


class IcoGenericUpConvLayer(nn.Module):
    """ The transposed convolution layer on icosahedron discretized sphere
    using n-ring filter (slow).
    """
    def __init__(self, in_feats, out_feats, up_neigh_indices, down_indices):
        """ Init.

        Parameters
        ----------
        in_feats: int
            input features/channels.
        out_feats: int
            output features/channels.
        up_neigh_indices: array
            upsampling neighborhood indices at sampling i + 1.
        down_indices: array
            downsampling indices at sampling i
        """
        super(IcoUpConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.up_neigh_indices = up_neigh_indices
        self.neigh_indices = up_neigh_indices[down_indices]
        self.down_indices = down_indices
        self.n_vertices, self.neigh_size = self.up_neigh_indices.shape

        self.flat_neigh_indices = self.neigh_indices.reshape(-1)
        self.argsort_neigh_indices = np.argsort(self.flat_neigh_indices)
        self.sorted_neigh_indices = self.flat_neigh_indices[
            self.argsort_neigh_indices]
        assert(np.unique(self.sorted_neigh_indices).tolist() ==
               list(range(self.n_vertices)))
        count = collections.Counter(self.sorted_neigh_indices)
        self.count = sorted(count.items(), key=lambda item: item[0])

        self.weight = nn.Linear(in_feats, self.neigh_size * out_feats)

    def _check_occurence(self, data, occ):
        count = collections.Counter(data)
        unique_count = np.unique(list(count.values()))
        assert len(unique_count) == 1
        assert unique_count[0] == occ

    def forward(self, x):
        logger.debug("UpSampleLayer: transpose conv...")
        debug("input", x)
        n_samples, n_feats, n_vertices = x.size()
        logger.debug(" weight: {0}".format(self.weight))
        logger.debug(" neighbors indices: {0}".format(
            self.neigh_indices.shape))
        x = x.permute(0, 2, 1)
        x = x.reshape(n_samples * n_vertices, n_feats)
        debug("input", x)
        x = self.weight(x)
        debug("weighted input", x)
        x = x.view(n_samples, n_vertices, self.neigh_size, self.out_feats)
        debug("weighted input", x)
        x = x.view(n_samples, n_vertices * self.neigh_size, self.out_feats)
        out = torch.zeros(n_samples, self.out_feats, self.n_vertices)
        start = 0
        for idx in range(self.n_vertices):
            _idx, _count = self.count[idx]
            assert(_idx == idx)
            stop = start + _count
            _x = x[:, self.argsort_neigh_indices[start: stop]]
            out[..., idx] = torch.mean(_x, dim=1)
            start = stop
        debug("output", out)
        return out


class IcoUpSampleLayer(nn.Module):
    """ The upsampling layer on icosahedron discretized sphere using
    interpolation.
    """
    def __init__(self, in_feats, out_feats, up_neigh_indices):
        """ Init.

        Parameters
        ----------
        in_feats: int
            input features/channels.
        out_feats: int
            output features/channels.
        up_neigh_indices: array
            upsampling neighborhood indices.
        """
        super(IcoUpSampleLayer, self).__init__()
        self.up_neigh_indices = up_neigh_indices
        self.n_vertices, self.neigh_size = up_neigh_indices.shape
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        logger.debug("UpSampleLayer: interp...")
        debug("input", x)
        n_vertices = x.size(2) * 4 - 6
        assert self.n_vertices == n_vertices
        n_features = x.size(1)
        logger.debug(" up neighbors indices: {0}".format(
            self.up_neigh_indices.shape))
        x = x[:, :, self.up_neigh_indices.reshape(-1)].view(
            len(x), n_features, n_vertices, self.neigh_size)
        debug("neighbors", x)
        x = torch.mean(x, dim=-1)
        debug("interp", x)
        n_samples = len(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(n_samples * self.n_vertices, self.in_feats)
        x = self.fc(x)
        x = x.view(n_samples, self.n_vertices, self.out_feats)
        x = x.permute(0, 2, 1)
        debug("output", x)
        return x


class IcoUpSampleFixIndexLayer(nn.Module):
    """ The upsampling layer on icosahedron discretized sphere using fixed
    indices 0 (padding new vertices with 0).
    """
    def __init__(self, in_feats, out_feats, up_neigh_indices):
        """ Init.

        Parameters
        ----------
        in_feats: int
            input features/channels.
        out_feats: int
            output features/channels.
        up_neigh_indices: array
            upsampling neighborhood indices.
        """
        super(IcoUpSampleFixIndexLayer, self).__init__()
        self.up_neigh_indices = up_neigh_indices
        self.n_vertices, self.neigh_size = up_neigh_indices.shape
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats)
        self.new_indices = []
        for idx, row in enumerate(self.up_neigh_indices):
            if len(np.unique(row)) > 1:
                self.new_indices.append(idx)

    def forward(self, x):
        logger.debug("UpSampleLayer: zero padding...")
        debug("input", x)
        n_vertices = x.size(2) * 4 - 6
        assert self.n_vertices == n_vertices
        n_features = x.size(1)
        logger.debug(" up neighbors indices: {0}".format(
            self.up_neigh_indices.shape))
        x = x[:, :, self.up_neigh_indices[:, 0]]
        debug("neighbors", x)
        x[:, :, self.new_indices] = 0
        debug("interp", x)
        n_samples = len(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(n_samples * self.n_vertices, self.in_feats)
        x = self.fc(x)
        x = x.view(n_samples, self.n_vertices, self.out_feats)
        x = x.permute(0, 2, 1)
        debug("output", x)
        return x


class IcoUpSampleMaxIndexLayer(nn.Module):
    """ The upsampling layer on icosahedron discretized sphere using
    max indices.
    """
    def __init__(self, in_feats, out_feats, up_neigh_indices, down_indices):
        """ Init.

        Parameters
        ----------
        in_feats: int
            input features/channels.
        out_feats: int
            output features/channels.
        up_neigh_indices: array
            upsampling neighborhood indices at sampling i + 1.
        down_indices: array
            downsampling indices at sampling i.
        """
        super(IcoUpSampleMaxIndexLayer, self).__init__()
        self.up_neigh_indices = up_neigh_indices
        self.neigh_indices = up_neigh_indices[down_indices]
        self.down_indices = down_indices
        self.n_vertices, self.neigh_size = up_neigh_indices.shape
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats)

    def forward(self, x, max_pool_indices):
        logger.debug("UpSampleLayer: max pooling driven zero padding...")
        debug("input", x)
        logger.debug(" neighbors indices: {0}".format(
            self.neigh_indices.shape))
        logger.debug(" max pool indices: {0}".format(max_pool_indices.shape))
        debug("input", x)
        n_samples, n_feats, n_raw_vertices = x.size()
        x = x.permute(0, 2, 1)
        x = x.reshape(n_samples * n_raw_vertices, self.in_feats)
        x = self.fc(x)
        x = x.view(n_samples, n_raw_vertices, self.out_feats)
        x = x.permute(0, 2, 1)
        debug("fc", x)
        n_samples, n_feats, n_raw_vertices = x.size()
        x = x.reshape(n_samples, -1)
        y = torch.zeros(n_samples, n_feats, self.n_vertices)
        vertices_indices = np.zeros((n_samples, n_feats, n_raw_vertices))
        # TODO: how to deal with different channels count
        for idx in range(n_raw_vertices):
            vertices_indices[..., idx] = self.neigh_indices[idx][
                max_pool_indices[..., idx]]
        vertices_indices = torch.from_numpy(vertices_indices).long()
        logger.debug(" vertices indices: {0}".format(vertices_indices.shape))
        vertices_indices = vertices_indices.view(n_samples, -1)
        logger.debug(" vertices indices: {0}".format(vertices_indices.shape))
        feats_indices = np.floor(
            np.linspace(0.0, float(n_feats), num=(n_raw_vertices * n_feats)))
        feats_indices[-1] -= 1
        feats_indices = torch.from_numpy(feats_indices).long()
        logger.debug(" features indices: {0}".format(feats_indices.shape))
        y[:, feats_indices, vertices_indices] = x
        debug("interp", y)
        return y
