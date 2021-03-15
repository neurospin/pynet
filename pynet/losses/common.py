# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides common losses.
"""

# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from pynet.utils import Losses


# Global parameters
logger = logging.getLogger("pynet")


@Losses.register
class MSELoss(object):
    """ Calculate the Mean Square Error loss between I and J.
    """
    def __init__(self, concat=False):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        """
        super(MSELoss, self).__init__()
        self.concat = concat

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute MSE loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        self.debug("I", arr_i)
        self.debug("J", arr_j)
        loss = torch.mean((arr_i - arr_j) ** 2)
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss

    def debug(self, name, tensor):
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


@Losses.register
class PCCLoss(object):
    """ Calculate the Pearson correlation coefficient between I and J.
    """
    def __init__(self, concat=False):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        """
        super(PCCLoss, self).__init__()
        self.concat = concat

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute PCC loss...")
        nb_channels = arr_j.shape[1]
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        logger.debug("  channels: {0}".format(nb_channels))
        self.debug("I", arr_i)
        self.debug("J", arr_j)
        centered_arr_i = arr_i - torch.mean(arr_i)
        centered_arr_j = arr_j - torch.mean(arr_j)
        pearson_loss = torch.sum(
            centered_arr_i * centered_arr_j) / (
                torch.sqrt(torch.sum(centered_arr_i ** 2) + 1e-6) *
                torch.sqrt(torch.sum(centered_arr_j ** 2) + 1e-6))
        loss = 1. - pearson_loss
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss

    def debug(self, name, tensor):
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


@Losses.register
class NCCLoss(object):
    """ Calculate the normalize cross correlation between I and J.
    """
    def __init__(self, concat=False, win=None):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        win: list of in, default None
            the window size to compute the correlation, default 9.
        """
        super(NCCLoss, self).__init__()
        self.concat = concat
        self.win = win

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute NCC loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        ndims = len(list(arr_i.size())) - 2
        if ndims not in [1, 2, 3]:
            raise ValueError("Volumes should be 1 to 3 dimensions, not "
                             "{0}.".format(ndims))
        if self.win is None:
            self.win = [9] * ndims
        device = arr_i.device
        sum_filt = torch.ones([1, 1, *self.win]).to(device)
        pad_no = math.floor(self.win[0] / 2)
        stride = tuple([1] * ndims)
        padding = tuple([pad_no] * ndims)
        logger.debug("  ndims: {0}".format(ndims))
        logger.debug("  stride: {0}".format(stride))
        logger.debug("  padding: {0}".format(padding))
        logger.debug("  filt: {0} - {1}".format(
            sum_filt.shape, sum_filt.get_device()))
        logger.debug("  win: {0}".format(self.win))
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))

        var_arr_i, var_arr_j, cross = self._compute_local_sums(
            arr_i, arr_j, sum_filt, stride, padding)
        cc = cross * cross / (var_arr_i * var_arr_j + 1e-5)
        loss = -1 * torch.mean(cc)
        logger.debug("  loss: {0}".format(loss))
        logger.info("Done.")

        return loss

    def _compute_local_sums(self, arr_i, arr_j, filt, stride, padding):
        conv_fn = getattr(func, "conv{0}d".format(len(self.win)))
        logger.debug("  conv: {0}".format(conv_fn))

        arr_i2 = arr_i * arr_i
        arr_j2 = arr_j * arr_j
        arr_ij = arr_i * arr_j

        sum_arr_i = conv_fn(arr_i, filt, stride=stride, padding=padding)
        sum_arr_j = conv_fn(arr_j, filt, stride=stride, padding=padding)
        sum_arr_i2 = conv_fn(arr_i2, filt, stride=stride, padding=padding)
        sum_arr_j2 = conv_fn(arr_j2, filt, stride=stride, padding=padding)
        sum_arr_ij = conv_fn(arr_ij, filt, stride=stride, padding=padding)

        win_size = np.prod(self.win)
        logger.debug("  win size: {0}".format(win_size))
        u_arr_i = sum_arr_i / win_size
        u_arr_j = sum_arr_j / win_size

        cross = (sum_arr_ij - u_arr_j * sum_arr_i - u_arr_i * sum_arr_j +
                 u_arr_i * u_arr_j * win_size)
        var_arr_i = (sum_arr_i2 - 2 * u_arr_i * sum_arr_i + u_arr_i *
                     u_arr_i * win_size)
        var_arr_j = (sum_arr_j2 - 2 * u_arr_j * sum_arr_j + u_arr_j *
                     u_arr_j * win_size)

        return var_arr_i, var_arr_j, cross


@Losses.register
class RCNetLoss(object):
    """ RCNet Loss function.

    This loss needs intermediate layers outputs.
    Use a callback function to set the 'layer_outputs' class parameter before
    each evaluation of the loss function.
    If you use an interface this parameter is updated automatically?

    PCCLoss
    """
    def __init__(self):
        self.similarity_loss = PCCLoss(concat=True)
        self.layer_outputs = None

    def __call__(self, moving, fixed):
        logger.debug("Compute RCNet loss...")
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        stem_results = self.layer_outputs["stem_results"]
        for stem_result in stem_results:
            params = stem_result["stem_params"]
            if params["raw_weight"] > 0:
                stem_result["raw_loss"] = self.similarity_loss(
                    stem_result["warped"], fixed) * params["raw_weight"]
        loss = sum([
            stem_result["raw_loss"] * stem_result["stem_params"]["weight"]
            for stem_result in stem_results if "raw_loss" in stem_result])
        self.layer_outputs = None
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss


@Losses.register
class VMILoss(object):
    """ Variational Mutual information loss function.

    Reference: http://bayesiandeeplearning.org/2018/papers/136.pdf -
               https://discuss.pytorch.org/t/help-with-histogram-and-loss-
               backward/44052/5
    """
    def get_positive_expectation(self, p_samples, average=True):
        log_2 = math.log(2.)
        Ep = log_2 - F.softplus(-p_samples)
        # Note JSD will be shifted
        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, average=True):
        log_2 = math.log(2.)
        Eq = F.softplus(-q_samples) + q_samples - log_2
        # Note JSD will be shifted
        if average:
            return Eq.mean()
        else:
            return Eq

    def __call__(self, lmap, gmap):
        """ The fenchel_dual_loss from the DIM code
        Reshape tensors dims to (N, Channels, chunks).

        Parameters
        ----------
        lmap: Tensor
            the moving data.
        gmap: Tensor
            the fixed data.
        """
        lmap = lmap.reshape(2, 128, -1)
        gmap = gmap.squeeze()

        N, units, n_locals = lmap.size()
        n_multis = gmap.size(2)

        # First we make the input tensors the right shape.
        l = lmap.view(N, units, n_locals)
        l = lmap.permute(0, 2, 1)
        l = lmap.reshape(-1, units)

        m = gmap.view(N, units, n_multis)
        m = gmap.permute(0, 2, 1)
        m = gmap.reshape(-1, units)

        u = torch.mm(m, l.t())
        u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

        mask = torch.eye(N).to(l.device)
        n_mask = 1 - mask

        E_pos = get_positive_expectation(u, average=False).mean(2).mean(2)
        E_neg = get_negative_expectation(u, average=False).mean(2).mean(2)

        E_pos = (E_pos * mask).sum() / mask.sum()
        E_neg = (E_neg * n_mask).sum() / n_mask.sum()
        loss = E_neg - E_pos

        return loss
