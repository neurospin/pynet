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
from pynet.utils import Losses


# Global parameters
logger = logging.getLogger("pynet")


@Losses.register
class FocalLoss(object):
    """ Define a Focal Loss.

    Loss(pt) = −αt(1−pt)γlog(pt)

    where pt is the model's estimated probability for each class.

    When an example is misclassified and pt is small, the modulating factor
    is near 1 and the loss is unaffected. As pt goes to 1, the factor goes to
    0 and the loss for well-classified examples is down-weighted.
    The focusing parameter γ smoothly adjusts the rate at which easy examples
    are down-weighted. When γ= 0, the loss is equivalent to cross entropy, and
    as γ isincreased the effect of the modulating factor is likewise increased.
    For instance, with γ= 2, an example classified with pt= 0.9 would have
    100×lower loss compared with cross entropy and with pt≈0.968 it would have
    1000×lower loss.
    Then we use an α-balanced variant of the focal loss for addressing class
    imbalance with a weighting factor α∈[0,1]. In practice α may be set by
    inverse class frequency.

    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, gamma=2, alpha=None, reduction="mean", with_logit=True):
        """ Class instanciation.

        Parameters
        ----------
        gamma: float, default 2
            the focusing parameter >=0.
        alpha: float or list of float, default None
            if set use alpha-balanced variant of the focal loss.
        reduction: str, default 'mean'
            specifies the reduction to apply to the output: 'none' - no
            reduction will be applied, 'mean' - the sum of the output
            will be divided by the number of elements in the output, 'sum'
            - the output will be summed.
        with_logit: bool, default True
            apply the softmax logit function to the result.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.with_logit = with_logit
        self.eps = 1e-6
        self.layer_outputs = None

    def __call__(self, output, target):
        """ Compute the loss.

        An update of the alpha parameter can be released with the layer outputs
        that is expected to be a dict.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤targets[i]≤C−1.
        """
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        if (isinstance(self.layer_outputs, dict) and
                "alpha" in self.layer_outputs):
            self.alpha = self.layer_outputs["alpha"]
        if self.with_logit:
            output = func.softmax(output, dim=1)
        output = output + self.eps

        # Create the labels one hot encoded tensor
        one_hot = torch.zeros(n_batch, n_classes, *target.shape[1:],
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(
            1, target.unsqueeze(1), 1.) + self.eps

        # Compute the focal loss
        weight = torch.pow(1 - output, self.gamma)
        self.alpha = self.alpha or 1.
        if not isinstance(self.alpha, list):
            self.alpha = [self.alpha] * n_classes
        if len(self.alpha) != n_classes:
            raise ValueError("Invalid alphas.")
        focal = weight * torch.log(output)
        for idx, alpha in enumerate(self.alpha):
            focal[:, idx] = -alpha * focal[:, idx]
        tmp_loss = torch.sum(target_one_hot * focal, dim=1)

        # Reduction
        if self.reduction == "none":
            loss = tmp_loss
        elif self.reduction == "mean":
            loss = torch.mean(tmp_loss)
        elif reduction == "sum":
            self.loss = torch.sum(tmp_loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        return loss


@Losses.register
class DiceLoss(object):
    """ Define a multy classes Dice Loss.

    Dice = (2 |X| intersec |Y|) / (|X| + |Y|)

    Note that PyTorch optimizers minimize a loss. In this case, we would like
    to maximize the dice loss so we return 1 - Dice.
    """
    def __init__(self, with_logit=True):
        """ Class instanciation.

        Parameters
        ----------
        with_logit: bool, default True
            apply the softmax logit function to the result.
        """
        self.with_logit = with_logit
        self.eps = 1e-6

    def __call__(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤targets[i]≤C−1.
        """
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        if self.with_logit:
            output = func.softmax(output, dim=1)

        # Create the labels one hot encoded tensor
        one_hot = torch.zeros(n_batch, n_classes, *target.shape[1:],
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(
            1, target.unsqueeze(1), 1.)

        # Compute the dice score
        dims = tuple(range(1, len(target.shape) + 1))
        intersection = torch.sum(output * target_one_hot, dims)
        cardinality = torch.sum(output + target_one_hot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)

        return torch.mean(1. - dice_score)


@Losses.register
class SoftDiceLoss(object):
    """ Soft Dice Loss.
    """
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def __call__(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = (torch.sum(torch.mul(y_pred, y_pred)) +
                 torch.sum(torch.mul(y_true, y_true)) + eps)
        dice = 2 * intersection / union
        dice_loss = 1 - dice
        return dice_loss


@Losses.register
class CustomKLLoss(object):
    """ KL Loss.
    """
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def __call__(self, mean, std):
        return (torch.mean(torch.mul(mean, mean)) +
                torch.mean(torch.mul(std, std)) -
                torch.mean(torch.log(torch.mul(std, std))) - 1)


@Losses.register
class NvNetCombinedLoss(object):
    """ Combined Loss.

    Diceloss + k1 * L2loss + k2 * KLloss
    Since the output of the segmentation decoder has N channels (prediction
    for each tumor subregion), we simply add the N dice loss functions.
    A hyper-parameter weight of k1=0.1, k2=0.1 was found empirically in the
    paper.
    """
    def __init__(self, num_classes, k1=0.1, k2=0.1):
        super(NvNetCombinedLoss, self).__init__()
        self.layer_outputs = None
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def __call__(self, outputs, y_true):
        y_pred = outputs
        y_mid = self.layer_outputs
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
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))
        loss = torch.mean((arr_i - arr_j) ** 2)
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss


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
