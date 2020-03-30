# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from pynet.utils import Losses


# Global parameters
logger = logging.getLogger("pynet")


def dice_loss_1(logits, true, eps=1e-7):
    """ Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    true: a tensor of shape [B, 1, H, W].
    logits: a tensor of shape [B, C, H, W]. Corresponds to
    the raw output or logits of the model.
    eps: added to the denominator for numerical stability.
    dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def dice_loss_2(output, target, weights=1):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    """
    output = func.softmax(output, dim=1)
    target = torch.argmax(target, dim=1).type(torch.LongTensor)
    encoded_target = output.data.clone().zero_()
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = Variable(encoded_target)

    assert output.size() == encoded_target.size(), "Input sizes must be equal."
    assert output.dim() == 4, "Input must be a 4D Tensor."

    num = (output * encoded_target).sum(dim=3).sum(dim=2)
    den1 = output.pow(2).sum(dim=3).sum(dim=2)
    den2 = encoded_target.pow(2).sum(dim=3).sum(dim=2)

    dice = (2 * num / (den1 + den2)) * weights
    return dice.sum() / dice.size(0)


@Losses.register
class MultiDiceLoss(object):
    """ Define a multy classes dice loss.

    Note that PyTorch optimizers minimize a loss. In this case, we would like
    to maximize the dice loss so we return the negated dice loss.
    """
    def __init__(self, weight=None, ignore_index=None, nb_batch=None):
        """ Class instanciation.

        Parameters
        ----------
        weight: FloatTensor (C), default None
             a manual rescaling weight given to each class.
        ignore_index: int, default None
            specifies a target value that is ignored and does not contribute
            to the input gradient.
        nb_batch: int, default None
            the number of mini batch to rescale loss between 0 and 1.
        """
        self.weight = weight or 1
        self.ignore_index = ignore_index
        self.nb_batch = nb_batch or 1

    def __call__(self, output, target):
        """ Compute the loss.

        Note that this criterion is performing nn.Softmax() on the model
        outputs.

        Parameters
        ----------
        output: Variable (NxCxHxW)
            unnormalized scores for each class (the model output) where C is
            the number of classes.
        target: LongTensor (NxCxHxW)
            the class indices.
        """
        eps = 1  # 0.0001
        n_classes = output.size(1) * self.nb_batch

        output = func.softmax(output, dim=1)
        target = torch.argmax(target, dim=1).type(torch.LongTensor)
        # output = output.exp()

        encoded_target = output.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
        denominator = output + encoded_target
        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = self.weight * (1 - (numerator / denominator))
        logger.info(loss_per_channel)

        return loss_per_channel.sum() / n_classes


@Losses.register
class SoftDiceLoss(_Loss):
    """ Soft Dice Loss.
    """
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = (torch.sum(torch.mul(y_pred, y_pred)) +
                 torch.sum(torch.mul(y_true, y_true)) + eps)
        dice = 2 * intersection / union
        dice_loss = 1 - dice
        return dice_loss


@Losses.register
class CustomKLLoss(_Loss):
    """ KL Loss.
    """
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return (torch.mean(torch.mul(mean, mean)) +
                torch.mean(torch.mul(std, std)) -
                torch.mean(torch.log(torch.mul(std, std))) - 1)


@Losses.register
class NvNetCombinedLoss(_Loss):
    """ Combined Loss.

    Diceloss + k1 * L2loss + k2 * KLloss
    Since the output of the segmentation decoder has N channels (prediction
    for each tumor subregion), we simply add the N dice loss functions.
    A hyper-parameter weight of k1=0.1, k2=0.1 was found empirically in the
    paper.
    """
    def __init__(self, num_classes, k1=0.1, k2=0.1):
        super(NvNetCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, outputs, y_true):
        y_pred, y_mid = outputs
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred = y_pred[:, :self.num_classes]
        seg_truth = y_true[:, :self.num_classes]
        vae_pred = y_pred[:, self.num_classes:]
        vae_truth = y_true[:, self.num_classes:]
        dice_loss = None
        for idx in range(self.num_classes):
            if dice_loss is None:
                dice_loss = self.dice_loss(
                    seg_pred[:, idx], seg_truth[:, idx])
            else:
                dice_loss += self.dice_loss(
                    seg_pred[:, idx], seg_truth[:, idx])
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        logger.debug(
            "dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:"
            "%.4f" % (dice_loss, l2_loss, kl_div, combined_loss))
        return combined_loss


@Losses.register
class MSELoss(_Loss):
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

    def forward(self, arr_i, arr_j):
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
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))
        loss = torch.mean((arr_i - arr_j) ** 2)
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss


@Losses.register
class PCCLoss(_Loss):
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

    def forward(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute PCC loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        logger.debug("  channels: {0}".format(nb_channels))
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))
        centered_arr_i = arr_i - torch.mean(arr_i)
        centered_arr_j = arr_j - torch.mean(arr_j)
        pearson_loss = torch.sum(
            centered_arr_i * centered_arr_j) / (
                torch.sqrt(torch.sum(centered_arr_i ** 2) + 1e-6) *
                torch.sqrt(torch.sum(centered_arr_j ** 2) + 1e-6))
        loss = 1. - pearson_loss
        logger.debug("  loss: {0}".format(loss))
        logger.info("Done.")
        return loss


@Losses.register
class NCCLoss(_Loss):
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

    def forward(self, arr_i, arr_j):
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
        device = arr_i.get_device()
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
